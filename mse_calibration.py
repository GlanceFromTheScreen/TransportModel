import numpy as np
import math
import statistics as stat
from lib_1d_minimization.One_D_Problem_file import One_D_Problem
import matplotlib.pyplot as plt
import scipy
from scipy import optimize
from graphic_distribution import pf, PlotDistribution, get_bins_for_chjsquare


def MSECalibration(self, detterance_func, hist_to_compare, eps=0.01, target_function = 'trial', aux_data = {'MSE': None}):

    # aux_data = {'aux_data_type like c_star. e.g mode': value of aux_data - double or list}

    START = -10
    END = 100
    FREQ = 100

    x_axe = np.linspace(START, END, FREQ)
    pf_current = np.array([pf(x, hist_to_compare) for x in x_axe])

    def get_mse(beta):
        self.TripDistribution(detterence_func=lambda x: detterance_func(x, beta), eps=eps)

        gr_x = []
        gr_y = []

        for row in range(self.T.shape[0]):
            for col in range(self.T.shape[1]):
                gr_x.append(self.c[row][col])
                gr_y.append(self.T[row][col])

        gr_x = np.array(gr_x)
        gr_y = np.array(gr_y)

        hist_data = []
        for i in range(len(gr_x)):
            hist_data += [gr_x[i]] * int(gr_y[i])

        if list(aux_data.keys())[0] == 'MSE':
            error = 1. / FREQ * np.sum(
                (pf_current - np.array([pf(x, hist_data) for x in x_axe])) ** 2
            )
        elif list(aux_data.keys())[0] == 'c_star_ORTUZAR':
            c_star = aux_data['c_star_ORTUZAR']
            error = abs(c_star - np.sum(self.T * self.c) / np.sum(self.T))
        elif list(aux_data.keys())[0] == 'c_star':
            c_star = aux_data['c_star']
            error = abs(c_star - stat.mean(hist_data))
        elif list(aux_data.keys())[0] == 'histogram_ERROR1':
            data_distance = aux_data['histogram_ERROR1']
            milestones = [0.]
            for key in data_distance.keys():
                milestones.append(key[1])
            bins1 = get_bins_for_chjsquare(hist_data, milestones)
            bins2 = get_bins_for_chjsquare(hist_to_compare, milestones)
            error = sum(abs((np.array(bins1) - np.array(bins2))))
        elif list(aux_data.keys())[0] == 'histogram_ERROR2':
            data_distance = aux_data['histogram_ERROR2']
            milestones = [0.]
            for key in data_distance.keys():
                milestones.append(key[1])
            bins1 = get_bins_for_chjsquare(hist_data, milestones)
            bins2 = get_bins_for_chjsquare(hist_to_compare, milestones)
            error = sum(((np.array(bins1) - np.array(bins2)))**2)

        return error

    # beta_values = np.arange(0, 1, 0.1)
    # mse_values = [get_mse(x) for x in beta_values]
    # plt.plot(beta_values, mse_values)
    # plt.xlabel('beta')
    # plt.ylabel('mse')
    # plt.title('mse')
    # plt.show()

    # print(optimize.shgo(lambda x: get_mse(x), bounds=[(0, 1.2)]).x)
    # print(optimize.minimize(lambda x: get_mse(x), 0.5, bounds=[(0, 1.2)]))

    p1 = One_D_Problem()
    p1.target_function = lambda x: get_mse(x)
    p1.left_border = 0
    p1.right_border = 2

    if target_function == 'trial':
        ans, n = p1.trial_point_method(eps)
    elif target_function == 'gold':
        ans, n = p1.golden_search(eps)
    elif target_function == 'uniform':
        ans, n = p1.uniform_search_method(accuracy=eps, n=6)
    print('1d min: ', ans, '\nfunction evaluations:', n, '\nerror: ', get_mse(ans))


    return {'beta': ans, 'target_function_value': p1.target_function(ans), 'error': get_mse(ans)}

