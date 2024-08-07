import aux_data_tmp
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
from statsmodels.stats.diagnostic import anderson_statistic
from aux_data_tmp import TMP_N_ARR


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


def PlotDistribution(self, hist_to_compare=None, is_show=True, is_save=[False, None], mean_c = None, ITERATIONS = None):

    if not ITERATIONS:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig, (ax3, ax1, ax2) = plt.subplots(1, 3, figsize=(17, 5))

    gr_x = []
    gr_y = []

    for row in range(self.T.shape[0]):
        for col in range(self.T.shape[1]):
            gr_x.append(self.c[row][col])
            gr_y.append(self.T[row][col])

    gr_x = np.array(gr_x)
    gr_y = np.array(gr_y)

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

    if ITERATIONS:
        print('TMP_N_ARR ', TMP_N_ARR)
        if 'Hyman' in is_save[1]:
            y_ax = TMP_N_ARR[-(ITERATIONS):]
        else:
            y_ax = TMP_N_ARR[-(ITERATIONS + 3):-3]

        # yticks_tmp = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
        # for item in yticks_tmp[1:]:
        #     if max(y_ax) > item:
        #         yticks_max = item * 10
        #         break
        #
        # for item in yticks_tmp[::-1][1:]:
        #     if min(y_ax) < item:
        #         yticks_min = item / 10
        #         break
        #
        # yticks = [item for item in yticks_tmp if item >= yticks_min and item <= yticks_max]
        # print(yticks, yticks_min, yticks_max, min(y_ax), max(y_ax))

        x_ax = np.arange(1, 200)
        ax3.semilogy([x_ax[i] for i in range(len(y_ax))], y_ax, 'o-', label='значения целевой функции')
        ax3.set_xlabel('номер итерации', fontsize=12)
        ax3.set_ylabel('значения целевой функции', fontsize=12)
        ax3.set_title('значения целевой функции', fontsize=12)
        # ax3.set_yticks(yticks)
        # ax3.set_aspect('equal', 'box')

        TMP_N_ARR.clear()


    if hist_to_compare:
        ax1.hist(hist_to_compare, bins=BINS_explicit, density=True, label='эмпирическая\nплотность распределения', alpha=0.8)

    ax1.hist(hist_data, bins=BINS_explicit, density=True, label='полученная\nплотность распределения', alpha=0.8)
    ax1.set_xlabel('обобщенная цена пути (в минутах)', fontsize=12)
    ax1.set_ylabel('плотности распределений', fontsize=12)
    ax1.set_title(f'Гистограммы распределений, beta={list(map(lambda x: round(x, 3), self.beta))}', fontsize=12)
    # ax1.set_aspect('equal', 'box')
    ax1.legend(loc="upper right", fontsize=12)
    # if is_save[0]:
    #     ax1.savefig(f'images/hist_{is_save[1]}')
    if is_show:
        plt.show()

    START = -10
    END = 100
    FREQ = 100

    x_axe = np.linspace(START, END, FREQ)
    ax2.plot(x_axe, [pf(x, hist_to_compare) for x in x_axe], label='эмпирическая\nфункция распределения')
    ax2.plot(x_axe, [pf(x, hist_data) for x in x_axe], label='полученная функция\nраспределения')
    ax2.set_xlabel('обобщенная цена пути (в минутах)', fontsize=12)
    ax2.set_ylabel('распределение вероятностей', fontsize=12)
    ax2.set_title('функции распределения вероятностей', fontsize=12)
    # ax2.set_aspect('equal', 'box')
    ax2.legend(loc='lower right', fontsize=12)
    # if is_save[0]:
    #     ax2.savefig(f'images/pp_{is_save[1]}')
    if is_save[0]:
        fig.savefig(f'images/{is_save[1]}')
    if is_show:
        plt.show()

    diff_supremum = np.max(np.abs(
        np.array([pf(x, hist_to_compare) for x in x_axe]) - np.array([pf(x, hist_data) for x in x_axe])
    ))

    diff_mse = 1. / FREQ * np.sum(
        (np.array([pf(x, hist_to_compare) for x in x_axe]) - np.array([pf(x, hist_data) for x in x_axe])) ** 2
    )

    pvalue_kstest = scipy.stats.ks_2samp(hist_to_compare, hist_data).pvalue
    pvalue_anderson = scipy.stats.anderson_ksamp([hist_data, hist_to_compare]).pvalue
    ttest = scipy.stats.ttest_ind(a=hist_data, b=hist_to_compare, equal_var=True)

    milestones = [10, 20, 30, 40, 50, 60]
    x1 = np.array(get_bins_for_chjsquare(hist_data, milestones))
    x2 = np.array(get_bins_for_chjsquare(hist_to_compare, milestones))
    T_chi2 = len(hist_data) * np.sum((x1 - x2) ** 2 / (x2))
    pv_chi2 = 1 - scipy.stats.chi2.cdf(T_chi2, df=len(x1))

    if mean_c:
        mean_diff = abs(np.mean(hist_data) - mean_c)
        print('mean_diff: ', mean_diff)

    print('DIFFERENCE SUPREMUM: ', diff_supremum)
    print('DIFFERENCE MSE: ', diff_mse)
    print('pvalue_kstest: ', pvalue_kstest)
    print('pvalue_chi2: ', pv_chi2, 'T: ', T_chi2)
    print('pvalue_anderson: ', pvalue_anderson)
    print('pvalue ttest: ', ttest.pvalue)

    plt.close()

    return {
        'diff_supremum': diff_supremum,
        'diff_mse': diff_mse,
        'pvalue_kstest': pvalue_kstest,
        'pvalue_chisquare': pv_chi2,
        'pvalue_anderson': pvalue_anderson,
        'pvalue_ttest': ttest.pvalue,
        'mean_diff': mean_diff if mean_c else None
    }







