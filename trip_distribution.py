import numpy as np
import math

def TripDistribution(self, detterence_func=lambda x: math.exp(-1 * x), eps=0.1):
    """
    Takes: origins O, destinations D, cost matrix c
    Generates: the list of correspondence matrices T (for each layer)
    """

    ###############################
    # INITIALIZE
    ###############################

    detterence_func = np.vectorize(detterence_func)
    detterence_matrix = detterence_func(self.c)
    B = np.array([1.0 for i in range(self.D.shape[0])])
    Error = eps + 1
    ITERATIONS = 0

    ###############################
    # MAIN LOOP
    ###############################

    while Error > eps and ITERATIONS < 1000:
        ITERATIONS += 1
        BDf = np.array(B * self.D * detterence_matrix)

        A = np.array([1. / sum(BDf[i, :]) for i in range(self.O.shape[0])])
        AOf = np.array(A * self.O * detterence_matrix)

        B = np.array([1. / sum(AOf[i, :]) for i in range(self.D.shape[0])])
        self.T = np.outer(A * self.O, B * self.D) * detterence_matrix

        O_ = np.array([sum(self.T[i, :]) for i in range(self.O.shape[0])])
        D_ = np.array([sum(self.T[:, j]) for j in range(self.D.shape[0])])

        Error = sum(abs(self.O - O_)) + sum(abs(self.D - D_))

    # print("ITERATIONS: ", ITERATIONS)
    # print("ERROR: ", Error)
    # print("O_ ", O_)
    # print("D_ ", D_)
    # print("T ", self.T)



