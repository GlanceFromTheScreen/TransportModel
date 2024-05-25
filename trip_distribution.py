import numpy as np
import math


def TripDistribution(self, detterence_func=lambda x: 1 / x**2, eps=0.1):
    """
    Takes: origins O, destinations D, cost matrix c
    Generates: the list of correspondence matrices T (for each layer)
    """

    eps = 0.01

    ###############################
    # INITIALIZE
    ###############################

    def detterence_func_preload(x):
        try:
            return detterence_func(x)
        except OverflowError:
            if x < 0:
                return 0
            else:
                return 999999.9

    detterence_func2 = np.vectorize(detterence_func_preload)
    detterence_matrix = detterence_func2(self.c)
    B = np.array([1.0 for i in range(self.D.shape[0])])
    Error = eps + 1
    ITERATIONS = 0

    ###############################
    # MAIN LOOP
    ###############################

    while Error > eps and ITERATIONS < 1000:
        ITERATIONS += 1
        BDf = np.array(B * self.D * detterence_matrix)


        tmp = np.array([sum(BDf[i, :]) for i in range(self.O.shape[0])])

        A = np.zeros(self.O.shape[0])
        for i in range(self.O.shape[0]):
            if tmp[i] != 0:
                A[i] = 1. / tmp[i]
            else:
                A[i] = 1
                if self.O[i] != 0:
                    print('i=',i,"\nO_i=",self.O[i])
        # A = np.array([1. / tmp[i] for i in range(self.O.shape[0])])
        AOf = np.array((A * self.O) * detterence_matrix.T).T

        for j in range(self.D.shape[0]):
            tmp2 = sum(AOf[:, j])
            if tmp2 != 0:
                B[j] = 1. / tmp2
            else:
                B[j] = 1
                if self.D[j] != 0:
                    print('j=', j, "\nD_j=", self.D[j])


        # B = np.array([1. / sum(AOf[:, j]) for j in range(self.D.shape[0])])
        self.T = np.outer(A * self.O, B * self.D) * detterence_matrix

        O_ = np.array([sum(self.T[i, :]) for i in range(self.O.shape[0])])
        D_ = np.array([sum(self.T[:, j]) for j in range(self.D.shape[0])])

        Error = sum(abs(self.O - O_)) + sum(abs(self.D - D_))

    return {'ITERATIONS': ITERATIONS, 'Error': Error, 'O_': O_, 'D_': D_}






