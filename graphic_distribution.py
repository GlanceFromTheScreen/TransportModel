import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
from statsmodels.stats.diagnostic import anderson_statistic


def pf(x, distr):
    return 1. / len(distr) * sum(distr <= x)

def get_bins_for_chjsquare(hist_data, milestones):
    milestones.sort()
    bins = [0] * (len(milestones) + 1)
    for item in hist_data:
        for i in range(len(milestones)):
            if item <= milestones[i]:
                bins[i] += 1. / len(hist_data)
                break
        else:
            bins[-1] += 1. / len(hist_data)
    return bins


def PlotDistribution(self, hist_to_compare=None, is_show=True):

    gr_x = []
    gr_y = []

    for row in range(self.T.shape[0]):
        for col in range(self.T.shape[1]):
            gr_x.append(self.c[row][col])
            gr_y.append(self.T[row][col])

    gr_x = np.array(gr_x)
    gr_y = np.array(gr_y)

    # sorted_indices = np.argsort(gr_x)
    # gr_x = gr_x[sorted_indices]
    # gr_y = gr_y[sorted_indices]

    ###############################
    # each element in self.T implies that self.T_i_j workers
    # spend self.c_i_j time for traveling. So, initial
    # data for histogram - each element of c_i_j used T_i_j times
    ###############################
    hist_data = []
    for i in range(len(gr_x)):
        hist_data += [gr_x[i]] * int(gr_y[i])

    BINS = 15
    BINS_explicit = [0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 90]

    if hist_to_compare:
        plt.hist(hist_to_compare, bins=BINS_explicit, density=True, label='initial hist', alpha=0.8)

    plt.hist(hist_data, bins=BINS_explicit, density=True, label='resulted hist', alpha=0.8)
    plt.xlabel('travel time')
    plt.ylabel('workers density')
    plt.title(f'normalized distributions, beta={round(self.beta, 3)}')
    plt.legend()
    if is_show:
        plt.show()

    START = -10
    END = 100
    FREQ = 100

    x_axe = np.linspace(START, END, FREQ)
    plt.plot(x_axe, [pf(x, hist_to_compare) for x in x_axe], label='initial pf')
    plt.plot(x_axe, [pf(x, hist_data) for x in x_axe], label='resulted pf')
    plt.title('probability functions')
    plt.legend()
    if is_show:
        plt.show()

    diff_supremum = np.max(np.abs(
        np.array([pf(x, hist_to_compare) for x in x_axe]) - np.array([pf(x, hist_data) for x in x_axe])
    ))

    diff_mse = 1. / FREQ * np.sum(
        (np.array([pf(x, hist_to_compare) for x in x_axe]) - np.array([pf(x, hist_data) for x in x_axe])) ** 2
    )

    pvalue_kstest = scipy.stats.ks_2samp(hist_to_compare, hist_data).pvalue

    milestones = [10, 20, 30, 40, 50, 60]
    # x1 = np.array(get_bins_for_chjsquare(hist_data, milestones))
    # x2 = np.array(get_bins_for_chjsquare(hist_to_compare, milestones))  / len(hist_to_compare)
    # pvalue_chisquare = scipy.stats.chisquare(x1, x2 * len(hist_data)).pvalue
    #
    # T_chi2 = np.sum((x1 - len(hist_data) * x2) ** 2 / (len(hist_data) * x2))
    # pv_chi2 = 1 - scipy.stats.chi2.cdf(T_chi2, df=len(x1))
    #
    # pvalue_chisquare_2samp = scipy.stats.chi2_contingency([x1, x2 * len(hist_data)]).pvalue

    #  works bad
    pvalue_anderson = scipy.stats.anderson_ksamp([hist_data, hist_to_compare]).pvalue


    print('DIFFERENCE SUPREMUM: ', diff_supremum)
    print('DIFFERENCE MSE: ', diff_mse)
    print('pvalue_kstest: ', pvalue_kstest)
    print('pvalue_chisquare: ', pvalue_chisquare)
    print('pvalue_chisquare_2samp: ', pvalue_chisquare_2samp)
    print('pvalue_chi2: ', pv_chi2, 'T: ', T_chi2)
    print('pvalue_anderson: ', pvalue_anderson)

    return {
        'diff_supremum': diff_supremum,
        'diff_mse': diff_mse,
        'pvalue_kstest': pvalue_kstest,
        'pvalue_chisquare': pvalue_chisquare
    }







