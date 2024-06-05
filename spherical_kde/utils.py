""" Utilities

* General stable functions
* Transforming coordinates
* Computing rotations
* Performing spherical integrals
"""

import numpy as np
from scipy.integrate import dblquad


def cartesian_from_polar(phi, theta):
    """ Embedded 3D unit vector from spherical polar coordinates.

    Parameters
    ----------
    phi, theta : float or np.array
        azimuthal and polar angle in radians.

    Returns
    -------
    nhat : np.array
        unit vector(s) in direction (phi, theta).
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])


def polar_from_cartesian(x):
    """ Embedded 3D unit vector from spherical polar coordinates.

    Parameters
    ----------
    x : array_like
        cartesian coordinates

    Returns
    -------
    phi, theta : float or np.array
        azimuthal and polar angle in radians.
    """
    x = np.array(x)
    r = (x*x).sum(axis=0)**0.5
    x, y, z = x
    theta = np.arccos(z / r)
    phi = np.mod(np.arctan2(y, x), np.pi*2)
    return phi, theta


def polar_from_decra(ra, dec):
    """ Convert from spherical polar coordinates to ra and dec.

    Parameters
    ----------
    ra, dec : float or np.array
        Right ascension and declination in degrees.

    Returns
    -------
    phi, theta : float or np.array
        Spherical polar coordinates in radians
    """
    phi = np.mod(ra / 180. * np.pi, 2 * np.pi)
    theta = np.pi / 2 - dec / 180. * np.pi
    return phi, theta


def decra_from_polar(phi, theta):
    """ Convert from ra and dec to spherical polar coordinates.

    Parameters
    ----------
    phi, theta : float or np.array
        azimuthal and polar angle in radians

    Returns
    -------
    ra, dec : float or np.array
        Right ascension and declination in degrees.
    """
    ra = phi * (phi < np.pi) + (phi - 2 * np.pi) * (phi > np.pi)
    dec = np.pi / 2 - theta
    return ra / np.pi * 180, dec / np.pi * 180


def logsinh(x):
    """ Compute log(sinh(x)), stably for large x.

    Parameters
    ----------
    x : float or np.array
        argument to evaluate at, must be positive

    Returns
    -------
    float or np.array
        log(sinh(x))
    """
    return x + np.log(1-np.exp(-2*x)) - np.log(2)


def rotation_matrix(a, b):
    """ The rotation matrix that takes a onto b.

    Parameters
    ----------
    a, b : np.array
        Three dimensional vectors defining the rotation matrix

    Returns
    -------
    M : np.array
        Three by three rotation matrix

    Notes
    -----
    StackExchange post:
        https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    """
    v = np.cross(a, b)
    s = v.dot(v)**0.5
    if s == 0:
        return np.identity(3)
    c = np.dot(a, b)
    Id = np.identity(3)
    v1, v2, v3 = v
    vx = np.array([[0, -v3, v2],
                      [v3, 0, -v1],
                      [-v2, v1, 0]])
    vx2 = np.matmul(vx, vx)
    R = Id + vx + vx2 * (1-c)/s**2
    return R


def spherical_integrate(f, log=False):
    r""" Integrate an area density function over the sphere.

    Parameters
    ----------
    f : callable
        function to integrate  (phi, theta) -> float

    log : bool
        Should the function be exponentiated?

    Returns
    -------
    float
        Spherical area integral

        .. math::
            \int_0^{2\pi}d\phi\int_0^\pi d\theta
            f(\phi, \theta) \sin(\theta)
    """
    if log:
        def g(phi, theta):
            return np.exp(f(phi, theta))
    else:
        g = f
    ans, _ = dblquad(lambda phi, theta: g(phi, theta)*np.sin(theta),
                     0, np.pi, lambda x: 0, lambda x: 2*np.pi)
    return ans


def spherical_kullback_liebler(logp, logq):
    r""" Compute the spherical Kullback-Liebler divergence.

    Parameters
    ----------
    logp, logq : callable
        log-probability distributions  (phi, theta) -> float

    Returns
    -------
    float
        Kullback-Liebler divergence

            .. math::
                \int P(x)\log \frac{P(x)}{Q(x)} dx

    Notes
    -----
    Wikipedia post:
        https://en.wikipedia.org/wiki/Kullback-Leibler_divergence
    """
    def KL(phi, theta):
        return (logp(phi, theta)-logq(phi, theta))*np.exp(logp(phi, theta))
    return spherical_integrate(KL)
