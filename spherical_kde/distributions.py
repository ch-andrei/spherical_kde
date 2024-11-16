"""
The spherical kernel density estimator class and the kernel functions for the spherical KDE.

For more detail, see:
https://en.wikipedia.org/wiki/Von_Mises-Fisher_distribution

Note AC:
Modified from the original implementation.
There may be some inconsistensies for return shapes, i.e., (N,3) or (3,N) vectors, due to mixing implementations.
Original code for VMF uses 3,N but N,3 is better for vector operations...
https://github.com/williamjameshandley/spherical_kde

"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import brentq
from scipy.special import logsumexp

from common.vector_math import normalize, magnitude
from spherical_kde.spherical_kde.utils import cartesian_from_polar, polar_from_cartesian, logsinh


class SphericalKDE(object):
    """ Spherical kernel density estimator
    """
    def __init__(self, mu: np.ndarray, weights: np.ndarray = None):
        # print('mu', mu.shape)
        if mu.shape[0] != 3:
            raise ValueError()

        self.mu = mu.reshape(3, -1)  # 3, N
        self.num_mu = mu.shape[1]

        if weights is None:
            weights = np.ones(self.num_mu, float)
        self.weights = weights / weights.sum()  # normalize to sum to 1.0

        self.bandwidth = 1.06 * VonMises_std(self.mu) * self.num_mu ** -0.2

        # precompute kde at centers
        self.density = np.exp(self.evaluate_log_prob_density(self.mu))  # pdf at input vectors
        self.density = self.density / np.sum(self.density)  # softmax (exp / sum exp)
        self.density_cumsum = np.cumsum(self.density)

    def sample_distribution(self, num_samples):
        raise NotImplementedError()  # TODO  verify this
        # Note: we can only sample from the VMF distribution for a single center (mu) vector.
        # Given N center vectors on which KDE is computed (self.mu), compute density at each center.
        # Then, use the density map as the sampling probability map, and sample N centers.
        # Then draw a single sample from the VMF distribution around the picked centers.
        sample_mu = np.random.rand(num_samples)  # uniform samples
        sample_mu_inds = (np.searchsorted(self.density_cumsum, sample_mu)).astype(int)  # pick N mu vectors
        mu = self.mu[:, sample_mu_inds]
        x = VonMisesFisher_sample(mu, self.bandwidth, num_samples)  # draw VMF samples and rotate them towards mu
        log_prob = self.evaluate_log_prob_density(x)
        return x, log_prob

    def evaluate_prob_density(self, x):
        return np.exp(self.evaluate_log_prob_density(x))

    def evaluate_log_prob_density(self, x):
        # x shape (3,N)
        return logsumexp(
            VonMisesFisher_log_prob(x, self.mu, self.bandwidth),
            axis=-1,  # sum over N from shape (3,N)
            b=self.weights  # if uniform weights, this is the same as dividing output of logsumexp by self.num_mu
        )


def VonMisesFisher_log_prob_polar(phi, theta, phi0, theta0, sigma0):
    x = cartesian_from_polar(phi, theta)
    x0 = cartesian_from_polar(phi0, theta0)
    return VonMisesFisher_log_prob(x, x0, sigma0)


def VonMisesFisher_log_prob(x, x0, sigma0):
    """ Von-Mises Fisher distribution function.
    Returns log-probability of the vonmises fisher distribution.
    """
    norm = -np.log(4 * np.pi * sigma0 ** 2) - logsinh(1. / sigma0 ** 2)
    tdot = np.tensordot(x, x0, axes=[[0], [0]])
    return norm + tdot / sigma0 ** 2


# this function is from scipy.stats import vonmises_fisher
def sample_vonmisesfisher_scipy(kappa, num_samples):
    """
    Generate samples from a von Mises-Fisher distribution
    with mu = [1, 0, 0] and kappa. Samples then have to be
    rotated towards the desired mean direction mu.
    This method is much faster than the general rejection
    sampling based algorithm.
    Reference: https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf

    """
    # compute w coordinate acc. to equation from section 3.1
    w = np.random.rand(num_samples)
    w = 1. + np.log(w + (1. - w) * np.exp(-2 * kappa)) / kappa

    # (y, z) are random 2D vectors that only have to be
    # normalized accordingly. Then (w, y, z) follow a VMF distribution
    t = np.sqrt(1. - np.square(w))
    uniformcircle = sample_uniform_circle(num_samples)
    samples = np.stack([w, t * uniformcircle[..., 0], t * uniformcircle[..., 1]], axis=-1)
    return samples


def sample_uniform_circle(num_samples):
    """
    method to generate uniform directions
    Reference: Marsaglia, G. (1972). "Choosing a Point from the Surface of a
               Sphere". Annals of Mathematical Statistics. 43 (2): 645-646.
    """
    length = np.sqrt(np.random.uniform(1, size=num_samples))
    angle = np.pi * np.random.uniform(2, size=num_samples)
    x = length * np.cos(angle)
    y = length * np.sin(angle)
    return np.stack([x, y], axis=-1)


def sample_uniform_sphere(num_samples):
    uv = np.random.rand(2, num_samples)
    z = 1.0 - 2.0 * uv[0]
    r = np.sqrt(np.clip(1 - z * z, 0.0, None))
    phi = 2.0 * np.pi * uv[1]

    return np.stack([
        r * np.cos(phi),
        r * np.sin(phi),
        z
    ], axis=0)  # 3, N


def make_fibonacci_unit_sphere(num_samples):
    indices = np.arange(num_samples, dtype=float)
    phi = np.pi * (np.sqrt(5) - 1.)  # golden angle in radians

    # Compute y values linearly spaced from 1 to -1
    y = 1 - 2 * (indices / (num_samples - 1))
    radius = np.sqrt(1 - y ** 2)

    # Theta based on golden angle increment
    theta = phi * indices
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    points = np.stack((x, y, z), axis=0)    # 3, N

    return points


def rotation_matrix(fw, eps=1e-9):
    # returns N forward transform matrices for forward vectors fw
    up = np.zeros_like(fw)
    up[..., 2] = 1.0  # assume up is along axis=2
    rt = normalize(np.cross(fw, up, axis=1))
    rs = normalize(np.cross(fw, rt, axis=1))
    transform = np.concatenate([fw, rt, rs], axis=-1)
    return transform.reshape(-1, 3, 3) + eps * np.eye(3).reshape(1, 3, 3)  # add diagonal eps else rotations cause nans


def VonMisesFisher_sample_polar(phi0, theta0, sigma0, num_samples):
    x0 = cartesian_from_polar(phi0, theta0)
    x = VonMisesFisher_sample(x0, sigma0, num_samples)
    phi, theta = polar_from_cartesian(x)
    return phi, theta


def VonMisesFisher_sample(x0, sigma0, num_samples):
    # generate VMS samples aligned to direction (1,0,0)
    kappa = 1 / sigma0 ** 2
    x = sample_vonmisesfisher_scipy(kappa, num_samples)

    # create rotation matrices and rotate
    x0 = np.transpose(x0)  # from 3,N to N,3
    M = rotation_matrix(x0)
    xshape = x.shape
    x = x.reshape(-1, 1, 3) @ M
    x = np.transpose(x.reshape(xshape))  # back to 3, N

    return x


def VonMises_mean_polar(phi, theta):
    x = cartesian_from_polar(phi, theta)
    x0 = VonMises_mean(x)
    phi, theta = polar_from_cartesian(x0)
    return phi, theta


def VonMises_mean(x):
    """ Von-Mises sample mean.
    Wikipedia:
        https://en.wikipedia.org/wiki/Von_Mises-Fisher_distribution#Estimation_of_parameters
    """
    x0 = np.sum(x, axis=-1)
    return x0


def VonMises_std_polar(phi, theta):
    x = cartesian_from_polar(phi, theta)
    return VonMises_std(x)


def VonMises_std(x):
    """ Von-Mises sample standard deviation.
    Wikipedia:
        https://en.wikipedia.org/wiki/Von_Mises-Fisher_distribution#Estimation_of_parameters
        but re-parameterised for sigma rather than kappa.
    """
    if x.shape[-1] == 1:
        return 1.0

    S = np.sum(x, axis=-1)
    R = S.dot(S)**0.5 / x.shape[-1]

    def f(s):
        return 1 / np.tanh(s) - 1. / s - R

    kappa = brentq(f, 1e-8, 1e8)
    sigma = kappa**-0.5
    return sigma

