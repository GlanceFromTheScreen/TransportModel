from MainModel import FourStepModel
from synthetic_data import GenerateSyntheticData
import matplotlib.pyplot as plt
import statistics

if __name__ == '__main__':

    error_beta = []
    beta = 3.543
    N = 70
    M = 50

    ###############################
    # k - percent of mean value of T, in range of with we may see random shift
    # 10%, 20% ... 60%
    ###############################
    for k in range(0, N, 10):
        buff = []
        for i in range(M):
            m = GenerateSyntheticData(5, 5, beta, noise=0.01 * k)
            est_beta = m.HymanCalibration(0.01)['betta']
            buff += [abs(est_beta - beta)]
        f = statistics.mean(buff)
        error_beta += [statistics.mean(buff)]

    print(error_beta)
    plt.plot(range(0, N, 10), error_beta)
    plt.xlabel('величина шума')
    plt.ylabel('ошибка')
    plt.show()

