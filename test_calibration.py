from MainModel import FourStepModel
from synthetic_data import GenerateSyntheticData, GenerateSyntheticDataReversed
import matplotlib.pyplot as plt
import statistics
import numpy as np

def get_c_star(m):
    distr = np.concatenate(m.c)
    distr_T = np.concatenate(m.T) / np.sum(m.T)
    c_star = 0
    for i in range(len(distr)):
        c_star += distr[i] * distr_T[i]
    return c_star


if __name__ == '__main__':

    error_beta = []
    beta = 0.04
    N = 10
    M = 100

    ###############################
    # k - percent of mean value of T, in range of with we may see random shift
    # 10%, 20% ... 50%
    ###############################
    for k in range(0, N, 1):
        buff = []
        for i in range(M):
            m = GenerateSyntheticData(5, 5, beta, noise=0.01 * k)
            c_star = get_c_star(m)
            est_beta = m.HymanCalibration(0.01, c_star)['betta']
            buff += [abs(est_beta - beta) / beta * 100]
        f = statistics.mean(buff)
        error_beta += [statistics.mean(buff)]

    print(error_beta)
    plt.plot(range(0, N, 1), error_beta)
    plt.xlabel('величина шума (% от среднего значения матрицы корреспонденций)')
    plt.ylabel('относительная ошибка (%)')
    plt.show()

