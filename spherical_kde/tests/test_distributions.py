import numpy
from numpy.testing import assert_allclose
from scipy.integrate import dblquad
import spherical_kde.distributions as dxns
from spherical_kde.utils import cartesian_from_polar, polar_from_cartesian


def random_VonMisesFisher_distribution():
    phi0 = numpy.random.rand()*numpy.pi*2
    theta0 = numpy.random.rand()*numpy.pi
    sigma = numpy.random.rand()

    def f(phi, theta):
        return numpy.exp(dxns.VonMisesFisher_distribution(phi, theta,
                                                          phi0, theta0, sigma))
    return f, phi0, theta0, sigma


def spherical_integrate(f):
    ans, _ = dblquad(lambda phi, theta: f(phi, theta)*numpy.sin(theta),
                     0, numpy.pi, lambda x: 0, lambda x: 2*numpy.pi)
    return ans


def test_VonMisesFisher_distribution_normalisation():
    numpy.random.seed(seed=0)
    for _ in range(3):
        f, phi0, theta0, sigma = random_VonMisesFisher_distribution()
        N = spherical_integrate(f)
        assert_allclose(N, 1)


def test_VonMisesFisher_distribution_mean():
    numpy.random.seed(seed=0)
    for _ in range(3):
        f, phi0, theta0, sigma = random_VonMisesFisher_distribution()
        x = []
        for i in range(3):
            def g(phi, theta):
                return f(phi, theta) * cartesian_from_polar(phi, theta)[i]
            x.append(spherical_integrate(g))
        phi, theta = polar_from_cartesian(x)
        assert_allclose([phi0, theta0], [phi, theta])


def test_VonMisesFisher_mean():
    numpy.random.seed(seed=0)
    for _ in range(3):
        _, phi0, theta0, sigma0 = random_VonMisesFisher_distribution()
        N = 10000
        phi, theta = dxns.VonMisesFisher_sample(phi0, theta0, sigma0, N)
        phi, theta = dxns.VonMises_mean(phi, theta)
        assert_allclose((phi0, theta0), (phi, theta), 1e-2)


def test_VonMisesFisher_standarddeviation():
    numpy.random.seed(seed=0)
    for _ in range(3):
        _, phi0, theta0, sigma0 = random_VonMisesFisher_distribution()
        N = 10000
        phi, theta = dxns.VonMisesFisher_sample(phi0, theta0, sigma0, N)
        sigma = dxns.VonMises_standarddeviation(phi, theta)
        assert_allclose(sigma0, sigma, 1e-2)
