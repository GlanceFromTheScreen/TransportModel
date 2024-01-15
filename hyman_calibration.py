import numpy as np
import math

def HymanCalibration(self, eps):

    ###############################
    # GIVEN: T*
    # FIND: BEST BETA
    ###############################

    self.T = self.T / np.sum(self.T)

    c_star = np.sum(self.T * self.c)
    c_m = 0
    c_prev = 0

    betta_0 = 1.5 / c_star
    betta = betta_0
    betta_prev = betta_0

    ITERATION_M = 0
    while ITERATION_M < 5000:

        if c_m == None:
            print('cat')

        ITERATION_M += 1
        self.TripDistribution(detterence_func=lambda x: math.exp(-betta * x))
        self.T = self.T / np.sum(self.T)

        c_prev = c_m
        c_m = np.sum(self.T * self.c)

        if ITERATION_M > 1:
            buffer = betta
            betta = ((c_star - c_prev) * betta - (c_star - c_m) * betta_prev) / (c_m - c_prev)
            betta_prev = buffer
        else:
            betta = betta_0 * c_m / c_star

        if abs(c_m - c_star) < eps:
            # print('ITERATION: ', ITERATION_M)
            # print('beta', betta)
            # print('c*, c_m: ', c_star, c_m)
            return {'ITERATION_M': ITERATION_M, 'betta': betta, 'c_star': c_star, 'c_m': c_m}

    # print('cat')


