import numpy as np
import math
import matplotlib.pyplot as plt

def PlotDistribution(self):
    gr_x = []
    gr_y = []

    for row in range(self.T.shape[0]):
        for col in range(self.T.shape[1]):
            gr_x.append(self.c[row][col])
            gr_y.append(self.T[row][col])

    gr_x = np.array(gr_x)
    gr_y = np.array(gr_y)

    sorted_indices = np.argsort(gr_x)
    gr_x = gr_x[sorted_indices]
    gr_y = gr_y[sorted_indices]

    plt.plot(gr_x, gr_y, '.')
    plt.xlabel('travel time')
    plt.ylabel('workers number')
    plt.show()

    ###############################
    # each element in self.T implies that self.T_i_j workers
    # spend self.c_i_j time for traveling. So, initial
    # data for histogram - each element of c_i_j used T_i_j times
    ###############################
    hist_data = []
    for i in range(len(gr_x)):
        hist_data += [gr_x[i]] * int(gr_y[i])

    plt.hist(hist_data)
    plt.xlabel('travel time')
    plt.ylabel('workers number')
    plt.show()


