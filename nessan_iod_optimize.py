#
# --*-- coding: utf-8 --*--



class iod_optimize:
    """
    Initial orbit determination based on optimization

    Parameter:
    ==========
    time: `numpy.ndarray`, shape=(M,)
        Time of observation, seconds of the data
    direction: `numpy.ndarray`, shape=(M, 3)
        Observation value, a series of unit vectors
    site_xyz: `numpy.ndarray`, shape=(M,3)
        Position vectors of the site
    """
    def __init__(self, time, direction, site_xyz):
        self.time = time
        self.direction = direction
        self.site_xyz = site_xyz

    def __fitness_function_1(self, element)):
        # Step 1: calculate direction by element = (a, e, i, O, w, M)
        # use a, e, M only

        # Step 2: calculate fitness
        #   F = sqrt(2/M/(M-1)*sum(j=0 : M-1)sum(k=j+1 : M) (f[k] - f[j] - theta(k, j))^2)
        m = len(time)
        lst = np.arange(m)
        idx = [(i,j) for i in lst for j in np.arange(i+1, m)]

