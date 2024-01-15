import numpy as np
import math
import random
from MainModel import FourStepModel

def GenerateSyntheticData(O_len, D_len, beta, noise=1):
    O = np.array([random.randint(10, 1000) for i in range(O_len)])
    D = np.array([random.randint(10, 1000) for i in range(D_len)])
    postfix = random.randint(10, 1000)

    if np.sum(O[:-1]) > np.sum(D[:-1]):
        O[-1] = postfix
        D[-1] = np.sum(O[:-1]) - np.sum(D[:-1]) + postfix
    else:
        O[-1] = -np.sum(O[:-1]) + np.sum(D[:-1]) + postfix
        D[-1] = postfix

    c = np.random.rand(O_len, D_len)

    model = FourStepModel()
    model.O = O
    model.D = D
    model.c = np.round(c,2)

    model.TripDistribution(detterence_func=lambda x: math.exp(-beta * x), eps=0.1)

    ###############################
    # NOISE U(0,1)
    # noise - percent of mean value of T, in range of with we may see random shift
    ###############################
    mean_val = np.mean(model.T)
    percent = noise * mean_val
    if noise:
        for i in range(model.T.shape[0]):
            for j in range(model.T.shape[1]):
                model.T[i, j] += (2 * random.random() - 1) * percent

    return model



