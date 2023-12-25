import numpy as np
import math

def HymanCalibration(self, eps):

    ###############################
    # GIVEN: T*
    # FIND: BEST BETA
    ###############################

    c_star = np.sum(self.T * self.c)
    betta_0 = 1.5 / c_star
    # self.TripDistribution(detterence_func=lambda x: math.exp(-1 * betta_0 * x))

    betta = betta_0
    betta_prev = betta_0
    c_m = 0
    c_prev = 0
    ITERATION_M = 0
    while ITERATION_M < 1000:

        ITERATION_M += 1
        self.TripDistribution(detterence_func=lambda x: math.exp(-betta * x))

        c_prev = c_m
        c_m = np.sum(self.T * self.c)

        betta_prev = betta

        # if ITERATION_M > 1:
        #     betta = ((c_star - c_prev) * betta - (c_star - c_m) * betta_prev) / (c_m - c_prev)
        # else:
        #     betta = betta_0 * c_m / c_star
        betta = betta * c_m / c_star

        if abs(c_m - c_star) < eps:
            print('ITERATION: ', ITERATION_M)
            print('beta', betta)
            print('c*, c_m: ', c_star, c_m)
            return betta


