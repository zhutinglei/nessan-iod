import numpy as np
from datetime import datetime
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation
from astropy.coordinates.angles import Angle


import nessan_iod as iod


def load_data(filename):
    with open(filename) as f:
        line = f.readline()
    [x, y, z] = [float(a) for a in line.split()]
    site = EarthLocation.from_geocentric(x, y, z, unit='meter').to('kilometer')

    yr, mon, day, hr, min = np.loadtxt(filename, dtype=int,
                                       usecols=(0, 1, 2, 3, 4),
                                       skiprows=1,
                                       unpack=True)
    sec, ra, de = np.loadtxt(filename,
                             dtype=float,
                             usecols=(5, 6, 7),
                             skiprows=1,
                             unpack=True)
    angles = np.array([ra, de])
    angles = angles.transpose()
    epochs = []
    for i in np.arange(np.size(yr)):
        s = int(sec[i])
        ms = int((sec[i]-s)*1e6)
        epoch = Time(datetime(yr[i], mon[i], day[i], hr[i], min[i], s, ms),
                     scale='utc')
        epochs = np.append(epochs, epoch)
    return site, epochs, Angle(angles * u.deg)


def demo():

    print()
    print('obs1:')
    site, epochs, angles = load_data('./Week2/obs1.dat')
    indices = np.arange(0, epochs.size, 1)

    orbit, residual = iod.iod_with_angles(epochs[indices],
                                          angles[indices],
                                          site)

    print('Orbital elements:')
    print('   a: ', orbit.coe()[0])
    print(' ecc: ', orbit.coe()[1])
    print(' inc: ', orbit.coe()[2])
    print('raan: ', orbit.coe()[3])
    print('   w: ', orbit.coe()[4])
    print('   M: ', orbit.coe()[5])

    residual = residual.to('arcsec')
    print('mean and std of residuals:')
    print([residual.mean(), residual.std()])

    import scipy.stats as stats
    import matplotlib.pyplot as plt
    plt.figure(figsize=(16, 7))
    ax = plt.subplot(1, 2, 1)
    ax.scatter(residual[:, 0], residual[:, 1], marker='.', edgecolor=None)

    ax.set_aspect('equal')

    bound = 1.1 * np.max(abs(residual.to_value()))
    ax.set_xlim([-bound, bound])
    ax.set_ylim([-bound, bound])
    ax.set_xlabel('error along RA / arcsecond')
    ax.set_ylabel('error along DEC / arcsecond')
    ax.grid(True)
    
    x = np.arange(-bound, bound, bound/500.0)
    m, s = stats.norm.fit(residual[:, 0])
    y = stats.norm.pdf(x, m, s)
    ax = plt.subplot(2, 2, 2)
    ax.plot(x, y, '-.', label='PDF fitted by Gaussian')
    ax.hist(residual[:, 0].to_value(), bins=30, density=True, label='histogram')
    ax.grid(True)
    ax.set_title('error along RA')

    m, s = stats.norm.fit(residual[:, 1].to_value())
    y = stats.norm.pdf(x, m, s)
    ax = plt.subplot(2, 2, 4)
    ax.plot(x, y, '-.', label='PDF fitted by Gaussian')
    ax.hist(residual[:, 1].to_value(), bins=30, density=True, label='histogram')
    ax.grid(True)
    ax.legend()
    ax.set_title('error along DEC')
    ax.set_xlabel('arcsecond')


    plt.show()
    

if __name__ == "__main__":
    demo()


