import numpy as np
import math
import statistics as stat
from lib_1d_minimization.One_D_Problem_file import One_D_Problem
import matplotlib.pyplot as plt


def MedianCalibration(self, distr, eps=0.00001):

    ###############################
    # GIVEN: dictribution
    # FIND: MEDIAN BETA ESTIMATION
    ###############################

    TIMESTAMPS = 180

    MED = stat.median(distr)
    MED_IND = 0

    time_arr = []
    for i in range(TIMESTAMPS):
        curr = np.min(distr) + i * (np.max(distr) - np.min(distr)) / (TIMESTAMPS-1)
        time_arr += [curr]
        if i > 0 and curr > MED > time_arr[-2]:
            MED_IND = i


    print(MED_IND)

    ###############################
    # STEP 1: fill table1
    ###############################

    table1 = np.zeros((len(self.O), TIMESTAMPS))
    for i in range(self.c.shape[0]):
        for j in range(self.c.shape[1]):
            k = 1
            while k < len(time_arr) and time_arr[k] <= self.c[i][j]:
                k += 1
            table1[i][k-1] += self.D[j]

    # table1 /= c_scale

    ###############################
    # STEP 2: fill table2
    ###############################

    table2 = np.zeros(TIMESTAMPS)
    for t in range(table1.shape[1]):
        s = 0
        for i in range(table1.shape[0]):
            s += self.O[i] * table1[i][t]
        table2[t] = s / np.sum(self.O)

    plt.plot(range(TIMESTAMPS), table2, '.')
    plt.hist(table2)
    plt.xlabel('minutes')
    plt.ylabel('waighted population count')
    plt.title('table2')
    plt.show()

    ###############################
    # STEP 3: 1D minimization
    ###############################

    # left = lambda x: sum([table2[t] * 1 / t**x for t in range(1, MED_IND)])
    # right = lambda x: sum([table2[t] * 1 / t**x for t in range(MED_IND, TIMESTAMPS, 1)])
    left = lambda x: sum([table2[t] * math.exp(-time_arr[t] * x) for t in range(1, MED_IND)])
    right = lambda x: sum([table2[t] * math.exp(-time_arr[t] * x) for t in range(MED_IND, TIMESTAMPS, 1)])

    p1 = One_D_Problem()
    p1.target_function = lambda x: abs(left(x) - right(x))
    p1.left_border = 0
    p1.right_border = 1

    ans, n = p1.uniform_search_method(6, eps)
    print('Метод равномерного поиска:\n', n, ' ', ((ans / eps) // 1) * eps, '+-', eps / 2)

    print()
    x_axe = np.linspace(0, 0.5, 100)
    plt.plot(x_axe, [p1.target_function(x) for x in x_axe])
    plt.show()



