import numpy as np
import math
import random
from MainModel import FourStepModel

def GenerateSyntheticData(O_len, D_len, beta):
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

    ###
    # рудимент
    ###
    for i in range(O_len):
        for j in range(i, D_len):
            c[i, j] = c[j, i]

    model = FourStepModel()
    model.O = O
    model.D = D
    model.c = np.round(c,2)

    model.TripDistribution(detterence_func=lambda x: math.exp(-beta * x), eps=0.1)

    return model



