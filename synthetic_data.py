import numpy as np
import math
import random
from MainModel import FourStepModel

def get_random_O_D(O_len, D_len):
    O = np.array([random.randint(10, 1000) for i in range(O_len)])
    D = np.array([random.randint(10, 1000) for i in range(D_len)])
    postfix = random.randint(10, 1000)

    if np.sum(O[:-1]) > np.sum(D[:-1]):
        O[-1] = postfix
        D[-1] = np.sum(O[:-1]) - np.sum(D[:-1]) + postfix
    else:
        O[-1] = -np.sum(O[:-1]) + np.sum(D[:-1]) + postfix
        D[-1] = postfix

    return O, D


def add_noise(model, noise):
    mean_val = np.mean(model.T)
    percent = noise * mean_val
    if noise:
        for i in range(model.T.shape[0]):
            for j in range(model.T.shape[1]):
                model.T[i, j] += (2 * random.random() - 1) * percent


def GenerateSyntheticData(O_len, D_len, beta, noise=0):

    c_scale_ratio = 4

    O, D = get_random_O_D(O_len, D_len)
    c = np.random.rand(O_len, D_len) * c_scale_ratio

    model = FourStepModel()
    model.O = O
    model.D = D
    model.c = np.round(c,2)

    model.TripDistribution(detterence_func=lambda x: math.exp(-beta * x), eps=0.1)

    ###############################
    # NOISE U(0,1)
    # noise - percent of mean value of T, in range of with we may see random shift
    ###############################
    add_noise(model, noise)

    return model

def GenerateSyntheticDataReversed(O_len, D_len, beta, rev_det_fun_type='exp', noise=0):

    deterrence_functions = {'exp': lambda t: -1 * math.log(t) / beta if t != 0 else None,
                            'power': lambda t: 1 / t**(1/beta)}

    T = np.random.rand(O_len, D_len) * 100
    O = np.array([sum(T[i,:]) for i in range(O_len)])
    D = np.array([sum(T[:,j]) for j in range(D_len)])

    model = FourStepModel()
    model.O = O
    model.D = D
    model.T = T

    c = np.zeros([O_len, D_len])
    for i in range(O_len):
        for j in range(D_len):
            c[i][j] = deterrence_functions[rev_det_fun_type](T[i][j] / (O[i] * D[j]))
    model.c = c

    add_noise(model, noise)

    return model






