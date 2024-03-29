import numpy as np
import math
import statistics as stat
from lib_1d_minimization.One_D_Problem_file import One_D_Problem
import matplotlib.pyplot as plt
import scipy
from scipy import optimize


def MedianCalibration(self, MED, mini, maxi, eps=0.000001,):

    ###############################
    # GIVEN: dictribution
    # FIND: MEDIAN BETA ESTIMATION
    ###############################

    TIMESTAMPS = 90
    MED_IND = 0

    time_arr = []
    for i in range(1, TIMESTAMPS+1):
        curr = mini + i * (maxi - mini) / (TIMESTAMPS)
        time_arr += [curr]
        if i > 1 and time_arr[-1] >= MED > time_arr[-2]:
            MED_IND = i-1


    print(MED_IND)

    ###############################
    # STEP 1: fill table1
    ###############################

    table1 = np.zeros((len(self.O), TIMESTAMPS))
    for i in range(self.c.shape[0]):
        for j in range(self.c.shape[1]):
            k = 0
            while k < len(time_arr)-1 and time_arr[k] < self.c[i][j]:
                k += 1
            table1[i][k] += self.D[j]

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
    # plt.hist(table2)
    plt.xlabel('minutes')
    plt.ylabel('waighted population count')
    plt.title('table2')
    plt.show()

    ###############################
    # STEP 3: 1D minimization
    ###############################

    left = lambda x: sum([table2[t] / time_arr[t]**x for t in range(0, MED_IND)])
    right = lambda x: sum([table2[t] / time_arr[t]**x for t in range(MED_IND, TIMESTAMPS, 1)])
    # left = lambda x: sum([table2[t] * math.exp(-time_arr[t] * x) for t in range(0, MED_IND+1)])
    # right = lambda x: sum([table2[t] * math.exp(-time_arr[t] * x) for t in range(MED_IND+1, TIMESTAMPS, 1)])

    p1 = One_D_Problem()
    p1.target_function = lambda x: abs(left(x) - right(x))
    p1.left_border = 0
    p1.right_border = 1

    ans, n = p1.trial_point_method(eps)
    print('Метод равномерного поиска:\n', n, ' ', ((ans / eps) // 1) * eps, '+-', eps / 2, p1.target_function(ans))
    # print(optimize.basinhopping(lambda x: abs(left(x) - right(x)), 0))
    print(optimize.shgo(lambda x: abs(left(x) - right(x)), bounds=[(0, 1)]).x)

    print()
    x_axe = np.linspace(0, 50, 1000)
    plt.semilogy(x_axe, [p1.target_function(x) for x in x_axe])
    plt.show()

    return {'beta': ans}



