from MainModel import FourStepModel
from synthetic_data import GenerateSyntheticData, GenerateSyntheticDataReversed
import matplotlib.pyplot as plt
import statistics

if __name__ == '__main__':

    error_beta = []
    beta = 0.04
    N = 10
    M = 1000

    ###############################
    # k - percent of mean value of T, in range of with we may see random shift
    # 10%, 20% ... 50%
    ###############################
    for k in range(0, N, 1):
        buff = []
        for i in range(M):
            m = GenerateSyntheticData(5, 5, beta, noise=0.01 * k)
            est_beta = m.HymanCalibration(0.01)['betta']
            buff += [abs(est_beta - beta) / beta * 100]
        f = statistics.mean(buff)
        error_beta += [statistics.mean(buff)]

    print(error_beta)
    plt.plot(range(0, N, 1), error_beta)
    plt.xlabel('величина шума (% от среднего значения матрицы корреспонденций)')
    plt.ylabel('ошибка')
    plt.show()

