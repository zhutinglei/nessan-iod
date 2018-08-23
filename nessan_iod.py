# Standard Library
import numpy as np
# Astropy
import astropy.coordinates as coord
from astropy.coordinates.angles import Angle
from astropy.coordinates import EarthLocation
from astropy.time import Time
import astropy.units as u
from astropy.coordinates.earth_orientation import rotation_matrix
from astropy.coordinates.earth_orientation import matrix_product
from astropy.utils.iers import IERS
# Poliastro
from poliastro.twobody import Orbit
from poliastro.bodies import Earth

# Global parameters
iers_table = IERS.open('eopc04_08_IAU2000.62-now')

# Functions
def iod_obs_tod(location: EarthLocation, epoch: Time):
    """
    Get the true of date (tod) position-velocity vector of the observatory

    Parameters
    ----------
    location: `~astropy.coordinates.EarthLocation`
        Position vector in body fixed reference frame.
    epoch: `~astropy.time.Time`

    Returns
    -------
    pos, vel: ~astropy.units.Quantity
        Position and velocity vector of the observation, in tod frame

    """
    from astropy.coordinates.earth import OMEGA_EARTH
    from astropy import _erfa as erfa
    assert isinstance(location, EarthLocation)
    assert isinstance(epoch, Time)

    dut = epoch.get_delta_ut1_utc(iers_table=iers_table)
    ut1 = Time(epoch.mjd + dut.to_value('day'), format='mjd', scale='ut1')
    earth_rotation_angle = erfa.era00(ut1.jd1, ut1.jd2) * u.rad
    loc = location.get_itrs()
    unit = loc.x.unit
    loc = np.array([loc.x.value, loc.y.value, loc.z.value])
    pos = np.matmul(rotation_matrix(-earth_rotation_angle, 'z'), loc)
    vel = [-OMEGA_EARTH.value * pos[1], OMEGA_EARTH.value * pos[0], 0.0]
    acc = [-OMEGA_EARTH.value**2 * pos[0], -OMEGA_EARTH.value**2 * pos[1], 0.0]

    return pos * unit, \
           vel * unit * OMEGA_EARTH.unit, \
           acc * unit * OMEGA_EARTH.unit * OMEGA_EARTH.unit


def iod_obs_gcrs(location: EarthLocation, epoch: Time):
    """
    Get the GCRS position-velocity vector of the observatory, earth orientation
    model includes only precession and nutation (no polar motion)

    Parameters
    ----------
    location: `~astropy.coordinates.EarthLocation`
        Position vector in body fixed reference frame.
    epoch: `~astropy.time.Time`

    Returns
    -------
    pos, vel: ~astropy.units.Quantity
        Position and velocity vector of the observation, in tod frame
    """

    from astropy.coordinates.earth_orientation import nutation_matrix
    from astropy.coordinates.earth_orientation import precession_matrix_Capitaine

    pos, vel, acc = iod_obs_tod(location, epoch)

    j2000 = Time(2000.0, format='jyear', scale='utc')
    pmat = precession_matrix_Capitaine(j2000, epoch)
    nmat = nutation_matrix(epoch)
    gmat = matrix_product(nmat, pmat)

    pos = np.matmul(gmat.transpose(), pos.value) * pos.unit
    vel = np.matmul(gmat.transpose(), vel.value) * vel.unit
    acc = np.matmul(gmat.transpose(), acc.value) * acc.unit

    return pos, vel, acc


def iod_with_angles(epochs: Time, angles: Angle, location: EarthLocation):
    """
    Solve initial orbit determination problem, with observation data be angles
    only

    Parameters
    ----------
    epochs: `~astropy.time.Time`, utc
        epochs.size should be N
    angles: `~astropy.coordinates.angles.Angle`
        angles.shape should be (N, 2), with the first column be RA and the
        second column be DE.
    location: `~astropy.coordinates.EarthLocation`
        location of the observatory

    Returns
    -------
    orbit: `~poliastro.twobody.Orbit`

    """

    u_time = 13.446852063738202 * u.minute
    u_length = 6378.137 * u.kilometer
    times = [((epoch - epochs[0])/u_time).to_value(u.one) for epoch in epochs]

    tmp = coord.UnitSphericalRepresentation(angles[:, 0], angles[:, 1])
    directions = np.array([
        [d.x.to_value(), d.y.to_value(), d.z.to_value()]
        for d in tmp.to_cartesian()
    ])

    obsvectors = np.array([
        (iod_obs_tod(location, epoch)[0]/u_length).to_value(u.one)
        for epoch in epochs
    ])

    # Add a first guess
    indices = np.array([10, 20], dtype=int)
    times = np.array(times)
    range = _iod_laplace_first_guess(times[indices],
                                     directions[indices],
                                     obsvectors[indices])
    print(range*u_length)
    time, r, v, residual = _iod_laplace(times, directions, obsvectors, range)

    epoch = epochs[0] + time * u_time
    orbit = Orbit.from_vectors(Earth, r * u_length, v * u_length/u_time, epoch)
    residual = residual * u.rad

    return orbit, Angle(residual, unit='deg')


def _iod_cal_fg(tau, posvel=None):
    if posvel is None:
        f = 1.0
        g = tau
        return f, g
    elif isinstance(posvel, float):
        r = posvel
        f = 1.0 - 0.5 * tau * tau / (r**3)
        g = tau - 1.0 / 6.0 * (tau**3) / (r**3)
        return f, g
    else:
        assert np.size(posvel) == 6

    pos, vel = posvel[0:3], posvel[3:6]
    sigma = np.inner(pos, vel)
    r, v = np.linalg.norm(pos), np.linalg.norm(vel)
    f = 1.0 - 0.5 / (r**3) * (tau**2) \
        + 0.5 / (r**5) * sigma * (tau**3) \
        + 1.0/24.0 / (r**5) * (
            3.0 * v * v - 2.0 / r - 15.0 / r / r * sigma * sigma
        ) * (tau**4) \
        + 1.0/8.0 / (r**7) * sigma * (
            -3.0 * v * v + 2.0 / r + 7.0 / r / r * sigma * sigma
        ) * (tau**5) \
        + 1.0/720.0 / (r**7) * (
            sigma * sigma / r / r * (
                630.0 * v * v - 420.0 / r - 945.0 / r / r * sigma * sigma
            ) - (22.0 / r / r - 66.0 / r * v * v + 45.0 * (v**4))
        ) * (tau**6)
    g = tau - 1.0/6.0 / (r**3) * (tau**3) + 0.25 / (r**5)*sigma*(tau**4) \
        + 1.0/120.0 / (r**5) * (
            9.0 * v * v - 8.0 / r - 45 / r / r * sigma * sigma
        ) * (tau**5) \
        + 1.0 / 24.0 / (r**7) * sigma * (
            -6.0 * v * v + 5.0 / r + 14.0 / r / r * sigma * sigma
        ) * (tau**6)
    return f, g


def _iod_laplace_first_guess(times, directions, obsvectors):

    def pos(direction, obsvector, semimajor):
        R = np.linalg.norm(obsvector)

        theta = np.arccos(np.inner(direction, obsvector)/R)
        tmp = semimajor**2 - (R*np.sin(theta))**2
        range = -np.inner(direction, obsvector) + np.sqrt(tmp)
        return range * direction + obsvector

    def eq_of_semimajor(semimajor):
        pos_1 = pos(directions[0], obsvectors[0], semimajor)
        pos_2 = pos(directions[1], obsvectors[1], semimajor)
        delta_f = np.arccos(np.inner(pos_1, pos_2) /
                            np.linalg.norm(pos_1) /
                            np.linalg.norm(pos_2))
        tmp = semimajor ** (-1.5) * (times[1] - times[0])
        n_2pi = int(tmp / np.pi / 2)
        tmp -= n_2pi * np.pi * 2.0
        if tmp < 0:
            tmp += 2*np.pi
        return delta_f - tmp

    bounds = [1.05, 10.0]
    while bounds[1] - bounds[0] > 1.0e-9:
        low = eq_of_semimajor(bounds[0]) > 0
        up = eq_of_semimajor(bounds[1]) > 0
        if low == up:
            # print("fail")
            return 0.0
        b_mid = 0.5 * (bounds[0] + bounds[1])
        mid = eq_of_semimajor(b_mid) > 0
        if mid == low:
            bounds[0] = b_mid
        else:
            bounds[1] = b_mid

    return 0.5 * (bounds[0] + bounds[1])


def _iod_laplace(times, directions, obsvectors, guess_semimajor=None):
    time = 0.5 * (times[0] + times[-1])
    skew_mats = [
        np.cross(np.identity(3), directions[i])
        for i in np.arange(directions.shape[0])
    ]

    f, g = _iod_cal_fg(times[0]-time, guess_semimajor)
    left_mat = np.append(f*skew_mats[0], g*skew_mats[0], axis=1)
    right_vec = np.cross(directions[0], obsvectors[0])
    for i in np.arange(1, np.size(times)):
        f, g = _iod_cal_fg(times[i] - time)
        m = np.append(f * skew_mats[i], g * skew_mats[i], axis=1)
        left_mat = np.append(left_mat, m, axis=0)
        right_vec = np.append(right_vec, np.cross(directions[i], obsvectors[i]))
    posvel, residual = np.linalg.lstsq(left_mat, right_vec, rcond=-1)[0:2]
    r, v = posvel[0:3], posvel[3:6]

    max_iter = 100
    for iter in np.arange(max_iter):
        f, g = _iod_cal_fg(times[0]-time, posvel)
        left_mat = np.append(f*skew_mats[0], g*skew_mats[0], axis=1)
        right_vec = np.cross(directions[0], obsvectors[0])
        for i in np.arange(1, np.size(times)):
            f, g = _iod_cal_fg(times[i] - time, posvel)
            m = np.append(f * skew_mats[i], g * skew_mats[i], axis=1)
            left_mat = np.append(left_mat, m, axis=0)
            right_vec = np.append(right_vec,
                                  np.cross(directions[i], obsvectors[i]))
        posvel = np.linalg.lstsq(left_mat, right_vec, rcond=-1)[0]
        err_r = np.linalg.norm(r-posvel[0:3])
        err_v = np.linalg.norm((v-posvel[3:6]))
        if err_r < 1e-7 and err_v < 1e-7:
            break
        r, v = posvel[0:3], posvel[3:6]
        # print(iter, r, v)
    if iter == max_iter - 1:
        print('_iod_laplace fail')
    else:
        print('iod success after', iter, 'times of iterations')

    residual = []
    for i in np.arange(np.size(times)):
        f, g = _iod_cal_fg(times[i] - time, posvel)
        pos = f * r + g * v - obsvectors[i]
        pos = pos / np.linalg.norm(pos)
        dir_t = np.cross(np.array([0, 0, 1]), pos)
        dir_t /= np.linalg.norm(dir_t)
        dir_n = np.cross(pos, dir_t)
        err = directions[i] - pos
        theta = np.arccos(np.inner(directions[i], pos))
        err *= theta / np.linalg.norm(err)
        residual.append([np.inner(err, dir_t), np.inner(err, dir_n)])

    return time, r, v, residual


