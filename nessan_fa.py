"""
    Functions in Fundamental astronomy
"""

import numpy as np
from numpy.polynomial import Polynomial
from astropy.time import Time
from astropy.coordinates.angles import Angle
from astropy.coordinates.earth_orientation import nutation_components2000B
import astropy.units as u

DEFAULT_OBSTIME_J2K = Time('J2000', scale='utc')

def greenwich_sidereal_time(ut1_time=None):
    """
    Get Greenwich Mean Sidereal Time (GMST) and Greenwich Apparent Sidereal Time
    (GAST).

    Parameters
    ----------
    ut1_time : `~astropy.time.Time` or None
        If None, the default ``obstime`` of J2000.0 will be used.
    Returns
    -------
    gmst, gast : `~astropy.coordinates.angles.Angle`
    """

    if ut1_time is None:
        ut1_time = DEFAULT_OBSTIME_J2K
    else:
        assert isinstance(ut1_time, Time)

    julian_centuries_from_j2k = (ut1_time.jyear - 2000.0) / 100.0
    poly_gmst = Polynomial((18.6973746, 879000.0513367,
                            0.093104/3600.0, 6.2e-6/3600.0))
    gmst = poly_gmst(julian_centuries_from_j2k)
    gmst = Angle(gmst, unit='hour')

    eps, dpsi, deps = nutation_components2000B(ut1_time.jd)
    dmu = dpsi * np.cos(eps)
    gast = gmst + Angle(dmu, unit='radian')

    return gmst, gast



def precession_angle(ut1_time=None):
    """
    Generates precession angles.

    Parameters
    ----------
    ut1_time : `~astropy.time.Time` or None
        If None, the default ``obstime`` of J2000.0 will be used.
    Returns
    -------
    z, zeta, theta: `astropy.coordinates.angles.Angle`
    """

    if ut1_time is None:
        ut1_time = DEFAULT_OBSTIME_J2K
    else:
        assert isinstance(ut1_time, Time)

    julian_centuries_from_j2k = (ut1_time.jyear - 2000.0) / 100.0

    poly_z = Polynomial((-2.650545, 2306.077181, 1.0927348,
                         0.01826837, -0.000028596, -2.904e-7))
    poly_zeta = Polynomial((2.650545, 2306.083227, 0.2988499,
                            0.01801828, -5.971e-6, -3.173e-7))
    poly_theta = Polynomial((0.0, 2004.191903, -0.4294934,
                             -0.04182264, -7.089e-6, -1.274e-7))

    z = poly_z(julian_centuries_from_j2k) / 3600.0 * u.degree
    zeta = poly_zeta(julian_centuries_from_j2k) / 3600.0 * u.degree
    theta = poly_theta(julian_centuries_from_j2k) / 3600.0 * u.degree

    return Angle(z), Angle(zeta), Angle(theta)


def matrix_teme_to_tod(obstime=None):
    """
    Generates a 3 * 3 `numpy.matrix` which transforms a vector from true equator
    true equinox (teme) frame to true of date (tod) frame, i.e.
        [tod] = [matrix][teme]
    where [true] and [teme] are 3 * 1 `numpy.matrix`

    Parameters
    ----------
    obstime : `~astropy.time.Time` or None
        If None, the default ``obstime`` of J2000.0 will be used.

    Returns
    -------
    matrix : numpy.matrix of shape [3, 3]
    """

    if obstime is None:
        obstime = DEFAULT_OBSTIME_J2K
    else:
        assert isinstance(obstime, Time)

    angles = precession_angle(obstime)
    precession_ra = angles['z'] + angles['zeta']
    c = np.cos(np.deg2rad(precession_ra))
    s = np.sin(np.deg2rad(precession_ra))
    mat = np.matrix([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    return mat
