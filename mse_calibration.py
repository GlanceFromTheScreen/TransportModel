import numpy as np
import math
import statistics as stat
from lib_1d_minimization.One_D_Problem_file import One_D_Problem
import matplotlib.pyplot as plt
import scipy
from scipy import optimize
from graphic_distribution import pf, PlotDistribution, get_bins_for_chjsquare
from scipy.optimize import minimize, Bounds, LinearConstraint
from sko.GA import GA
from aux_data_tmp import TMP_N_ARR


def MSECalibration(self, detterance_func, hist_to_compare, eps=0.01, minimization_method = 'trial', aux_data = {'MSE': None}):

    # aux_data = {'aux_data_type like c_star. e.g mode': value of aux_data - double or list}

    START = -10
    END = 100
    FREQ = 100

    x_axe = np.linspace(START, END, FREQ)
    pf_current = np.array([pf(x, hist_to_compare) for x in x_axe])

    one_dimention = True
    if detterance_func.__code__.co_varnames[1] == 'params':
        one_dimention = False

    def get_mse(beta):
        beta = [beta] if type(beta).__name__ in ['int', 'float'] else list(beta)

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
            tt = max(hist_data)
            error = sum(abs((np.array(bins1) - np.array(bins2))))
        elif list(aux_data.keys())[0] == 'histogram_ERROR1_relative':
            data_distance = aux_data['histogram_ERROR1_relative']
            milestones = [0.]
            for key in data_distance.keys():
                milestones.append(key[1])
            bins1 = get_bins_for_chjsquare(hist_data, milestones)
            bins2 = get_bins_for_chjsquare(hist_to_compare, milestones)
            error = sum(abs((np.array(bins1[:-1]) - np.array(bins2[:-1]))) / np.array(bins2[:-1]))
        elif list(aux_data.keys())[0] == 'histogram_ERROR2':
            data_distance = aux_data['histogram_ERROR2']
            milestones = [0.]
            for key in data_distance.keys():
                milestones.append(key[1])
            bins1 = get_bins_for_chjsquare(hist_data, milestones)
            bins2 = get_bins_for_chjsquare(hist_to_compare, milestones)
            error = sum(((np.array(bins1) - np.array(bins2)))**2)

        print("....... ", beta, " -> ", error)

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        TMP_N_ARR.append(error)

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
    p1.left_border = -0.5
    p1.right_border = 1.5

    # bb = Bounds(lb=[0], ub=[2])
    # bb2 = Bounds(lb=[-1,-1], ub=[2,2])

    if minimization_method == 'trial':
        if not one_dimention:
            return
        ans, n = p1.trial_point_method(eps)
    elif minimization_method == 'gold':
        if not one_dimention:
            return
        ans, n = p1.golden_search(eps)
    elif minimization_method == 'uniform':
        if not one_dimention:
            return
        ans, n = p1.uniform_search_method(accuracy=eps, n=6)
    elif minimization_method == 'nelder-mead':
        if one_dimention:
            res = minimize(lambda x: get_mse(x), [0.3], method='nelder-mead', tol=eps, bounds=Bounds(lb=[-0.5], ub=[1.5]))
        else:
            res = minimize(lambda x: get_mse(x), [0.3, 0.3], method='nelder-mead', tol=eps, bounds=Bounds(lb=[-0.5, -0.5], ub=[1.5, 1.5]))
        ans = res.x
        n = res.nfev
    elif minimization_method == 'powell':
        if one_dimention:
            res = minimize(lambda x: get_mse(x), [0.3], method='powell', tol=eps, bounds=Bounds(lb=[-0.5], ub=[1.5]), options={'maxiter': 3})
        else:
            res = minimize(lambda x: get_mse(x), [0.3, 0.3], method='powell', tol=eps, bounds=Bounds(lb=[-0.5,-0.5], ub=[1.5,1.5]), options={'maxiter': 3})
        ans = res.x
        n = res.nfev
    elif minimization_method == 'SLSQP':
        if one_dimention:
            res = minimize(lambda x: get_mse(x), [0.3], method='SLSQP', bounds=Bounds(lb=[-0.5], ub=[1.5]), options={'eps': eps})
        else:
            res = minimize(lambda x: get_mse(x), [0.3, 0.3], method='SLSQP', bounds=Bounds(lb=[-0.5,-0.5], ub=[1.5,1.5]), options={'eps': eps})
        ans = res.x
        n = res.nfev
    elif minimization_method == 'GA':
        max_iter = 8
        size_pop = 8
        if one_dimention:
            ga = GA(func=lambda x: get_mse(x), n_dim=1, size_pop=size_pop, max_iter=max_iter, prob_mut=0.01, lb=[-0.5], ub=[1.5],
                    precision=eps)
        else:
            ga = GA(func=lambda x: get_mse(x), n_dim=2, size_pop=size_pop, max_iter=max_iter, prob_mut=0.01, lb=[-0.5, -0.5], ub=[1.5, 1.5],
                    precision=eps)
        best_x, best_y = ga.run()
        ans = best_x
        n = max_iter * size_pop
    print('1d min: ', ans, '\nfunction evaluations:', n, '\nerror: ', get_mse(ans))


    return {'beta': [ans] if type(ans).__name__ in ['int', 'float'] else list(ans), 'target_function_value': get_mse(ans), 'error': get_mse(ans), 'nfev': n}

