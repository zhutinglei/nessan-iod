"""
    Coordinate transformations
"""

def cart2sph(x, y, z):
    """
    Transformation from cartesian to spherical coordinates.

    Parameters
    ----------
    x, y, z: real values

    Returns
    -------
    [r, lon, lat]: list of size 3, real value
    r: range, real value
    lon: longitude (or alpha), real value, in radians
    lat: latitude (or phi), real value, in radians

    """
    from numpy import sqrt, arctan2, arcsin, pi
    r = sqrt(x*x + y*y + z*z)
    lon = arctan2(y, x)
    if lon < 0:
        lon += 2 * pi
    lat = arcsin(z / r)
    return [lon, lat, r]


def sph2cart(lon, lat, r):
    """
    Transformation from spherical to cartesian coordinates.

    Parameters
    -------
    r: range, real value
    lon: longitude (or alpha), real value, in radians
    lat: latitude (or phi), real values, in radians

    Returns
    -------
    (x, y, z): tuple of size 3, real values

    """
    from numpy import cos, sin
    x = cos(lon)*cos(lat)*r
    y = sin(lon)*cos(lat)*r
    z = sin(lat)*r
    import numpy as np
    return np.array([x, y, z]).transpose()


def jac_sph2cart(lon, lat, r):
    """
    Generate jacobian matrix of transformation from spherical to cartesian
    coordinate, i.e.
        [ partial (x, y, z) / partial (r, lon, lat) ]

    Parameters
    ----------
    r : range, float or ndarray
    lon: longitude (or alpha), float or ndarray, in radians
    lat: latitude (or phi), float or ndarray, in radians

    Returns
    -------
    jacobi: list of shape [3, 3]
    """
    from numpy import sin, cos
    jacobi = [
        [-r*cos(lat)*sin(lon), -r*sin(lat)*cos(lon), cos(lat)*cos(lon)],
        [r*cos(lat)*cos(lon), -r*sin(lat)*sin(lon), cos(lat)*sin(lon)],
        [0, r*cos(lat), sin(lat)]
    ]
    return jacobi


def jac_cart2sph(lon, lat, r):
    """
    Generate jacobian matrix of transformation from cartesian to spherical
    coordinate, i.e.
        [ partial (r, lon, lat) /  partial (x, y, z) ]

    Parameters
    ----------
    r : range, float
    lon: longitude (or alpha), float, in radians
    lat: latitude (or phi), float, in radians

    Returns
    -------
    jacobi: list of shape [3, 3]
    """
    from numpy import sin, cos
    jacobi = [
        [-sin(lon)/cos(lat)/r, cos(lon)/cos(lat)/r, 0],
        [-cos(lon)*sin(lat)/r, -sin(lon)*sin(lat)/r, cos(lat)/r],
        [cos(lon)*cos(lat), sin(lon)*cos(lat), sin(lat)]
    ]
    return jacobi


def posvel2sph(pv):
    """
    Transform from position-velocity vector (in cartesian coordinate) to
    spherical coordinate.

    Parameters
    ----------
    pv: [px, py, pz, vx, vy, vz] list of size 6, real values

    Returns
    -------
    sph: (r, lon, lat, dr, dlon, dlat)

    """
    from numpy import matrix
    [lon, lat, r] = cart2sph(pv[0], pv[1], pv[2])
    jacobi = jac_cart2sph(lon, lat, r)
    v = matrix(jacobi) * matrix(pv[3:6]).transpose()
    dlon, dlat, dr = v[0, 0], v[1, 0], v[2, 0]
    return [lon, lat, r, dlon, dlat, dr]


def sph2posvel(sph):
    """
    Transform from angle-range and angle-range-rate to position-velocity vector.

    Parameters
    ----------
    sph: [r, lon, lat, dr, dlon, dlat]

    Returns
    -------
    pv: [px, py, pz, vx, vy, vz] list of size 6, real values

    """
    from numpy import matrix
    [lon, lat, r] = sph[0:3]
    [px, py, pz] = sph2cart(lon, lat, r)
    jacobi = jac_sph2cart(lon, lat, r)
    v = matrix(jacobi) * matrix(sph[3:6]).transpose()
    vx, vy, vz = v[0, 0], v[1, 0], v[2, 0]
    return [px, py, pz, vx, vy, vz]

