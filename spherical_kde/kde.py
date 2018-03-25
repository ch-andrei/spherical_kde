import matplotlib
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import numpy
from spherical_kde.distributions import VMF
from spherical_kde.utils import polar_to_decra, decra_to_polar
import cartopy.crs as ccrs


class SphericalKDE(object):
    def __init__(self, phi_samples, theta_samples, weights=None, bandwidth=0.2):

        if len(phi_samples) != len(theta_samples):
            raise ValueError("phi_samples must be the same"
                             "length as theta_samples ({}!={})".format(
                                 len(phi_samples),len(theta_samples)))

        self.phi = numpy.array(phi_samples)
        self.theta = numpy.array(theta_samples)
        self.bandwidth = bandwidth
        if weights is None:
            self.weights = numpy.ones_like(theta_samples)
        else:
            self.weights = weights

        self.weights /= self.weights.sum()

        self.palefactor=0.6
        self.nphi = 100
        self.ntheta = 100

    def __call__(self, phi, theta):
        return logsumexp(VMF(phi, theta, self.phi, self.theta, self.bandwidth),
                         axis=-1, b=self.weights)

    def plot(self, ax, colour='g'):
        if not isinstance(ax.projection, ccrs.Projection):
            raise TypeError("ax.projection must be of type ccrs.Projection "
                            "({})".format(type(ax.projection)))

        # Compute the kernel density estimate of the probability on an equiangular grid
        ra = numpy.linspace(-180, 180, self.nphi)
        dec = numpy.linspace(-90, 90, self.ntheta)
        X, Y = numpy.meshgrid(ra, dec)
        phi, theta = decra_to_polar(X, Y)
        P = numpy.exp(self(phi, theta)) 

        # Find 2- and 1-sigma contours
        Ps = numpy.exp(self(self.phi, self.theta))
        i = numpy.argsort(Ps)
        cdf = self.weights[i].cumsum()
        levels = [Ps[i[numpy.argmin(cdf<f)]] for f in [0.05, 0.33]] + [numpy.inf]

        # Plot the countours on a suitable equiangular projection
        ax.contourf(X, Y, P, levels=levels, colors=self.colours(colour), 
                    transform=ccrs.PlateCarree())

    def colours(self, colour):
        cols = [matplotlib.colors.colorConverter.to_rgb(colour)]
        for _ in range(1, 2):
            cols = [[c * (1 - self.palefactor) + self.palefactor for c in cols[0]]] + cols
        return cols

    def decra_samples(self):
        weights = self.weights / self.weights.max()
        i_ = weights > numpy.random.rand(len(weights))
        phi = self.phi[i_]
        theta = self.theta[i_]
        ra, dec = polar_to_decra(phi, theta)
        return ra, dec

