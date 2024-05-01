import numpy as np
import math


def HymanCalibration(self, eps, c_star=None, detterence_func=lambda x, beta: math.exp(-betta * x)):

    ###############################
    # GIVEN: T* (or c*)
    # FIND: BEST BETA
    ###############################

    if not c_star:  # заглушка для тестирования калибровки
        c_star = np.sum(self.T * self.c) / np.sum(self.T)

    c_m = 0

    betta_0 = 1.5 / c_star
    betta = betta_0
    betta_prev = betta_0

    ITERATION_M = 0
    while ITERATION_M < 5000:

        ITERATION_M += 1
        tmp_res = self.TripDistribution(detterence_func=lambda x: detterence_func(x, betta))

        c_prev = c_m
        c_m = np.sum(self.T * self.c) / np.sum(self.T)

        if ITERATION_M > 1:
            buffer = betta
            betta = ((c_star - c_prev) * betta - (c_star - c_m) * betta_prev) / (c_m - c_prev)
            betta_prev = buffer
        else:
            betta = betta_0 * c_m / c_star

        if abs(c_m - c_star) < eps:
            return {'ITERATION_M': ITERATION_M, 'beta': betta, 'c_star': c_star, 'c_m': c_m, 'target_function_value': abs(c_m - c_star)}


