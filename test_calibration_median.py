from MainModel import FourStepModel
from synthetic_data import GenerateSyntheticData, GenerateSyntheticDataReversed
import matplotlib.pyplot as plt
import statistics as stat
import numpy as np

if __name__ == '__main__':
    beta = 0.04
    m = GenerateSyntheticDataReversed(20, 20, beta)
    distr = np.concatenate(m.c)

    ###############################
    # CASE 1
    ###############################
    # m.MedianCalibration(distr)

    ###############################
    # CASE 2
    ###############################
    distr_T = np.concatenate(m.T)
    hist_data = []
    for i in range(len(distr)):
        hist_data += [distr[i]] * int(distr_T[i])

    plt.hist(hist_data, bins=20, label='hist')
    # plt.hist(distr, label='distr')
    plt.legend()
    plt.show()

    m.MedianCalibration(stat.median(distr), mini=np.min(distr), maxi=np.max(distr))

    print('cat')

