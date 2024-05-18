import numpy as np
import pandas as pd
import math
from MainModel import FourStepModel
from synthetic_data import GenerateSyntheticData, GenerateSyntheticDataReversed
import matplotlib.pyplot as plt
import statistics
import pickle
from data_preprocessing import exclude_excess_rows_and_cols
from operator import itemgetter


m = FourStepModel()
eps = 0.01
minimization_method = 'nelder-mead'

with open("data/c_matrix_distance.pickle", "rb") as file:
    m.c = pickle.load(file)
with open("data/origin.pickle", "rb") as file:
    m.O = O_vector = pickle.load(file)
with open("data/destination.pickle", "rb") as file:
    m.D = pickle.load(file)
with open("data/c_star_distance.pickle", "rb") as file:
    c_star = pickle.load(file)
with open("data/hist_data_distance.pickle", "rb") as file:
    hist_data_distance = pickle.load(file)
with open("data/min_max_distance.pickle", "rb") as file:
    mini, maxi = pickle.load(file)
with open("data/data_distance.pickle", "rb") as file:
    data_distance = pickle.load(file)

exclude_excess_rows_and_cols(m)

print('OK.......')

detterence_func_pow=lambda x, beta: 1 / x**beta[0] if x != 0 else 999999.9
detterence_func_exp=lambda x, beta: math.exp(-beta[0] * x)
detterence_func_combined=lambda x, params: (x ** (-params[1])) * math.exp(-params[0] * x) if x != 0 else 999999.9

beta_df = pd.DataFrame(columns=['Method', 'Exp', 'Exp TFV', 'Pow', 'Pow TFV'])
mse_df = pd.DataFrame(columns=['Method', 'Exp', 'Pow'])
sup_df = pd.DataFrame(columns=['Method', 'Exp', 'Pow'])
pvalue_df = pd.DataFrame(columns=['Method', 'Exp', 'Pow'])
pvalue_chisquare_df = pd.DataFrame(columns=['Method', 'Exp', 'Pow'])
pvalue_anderson_df = pd.DataFrame(columns=['Method', 'Exp', 'Pow'])
pvalue_ttest_df = pd.DataFrame(columns=['Method', 'Exp', 'Mean Diff Exp', 'Pow', 'Mean Diff Pow'])
n_df = pd.DataFrame(columns=['Method', 'Exp', 'Pow'])

print('\nHyman pow:')

beta_h_p, tfv_h_p, n_h_p = itemgetter('beta', 'target_function_value', 'nfev')(m.HymanCalibration(
    0.0001,
    c_star,
    detterence_func=detterence_func_pow
))
m.beta = beta_h_p
m.TripDistribution(detterence_func=lambda x: detterence_func_pow(x, m.beta), eps=eps)
diff_supremum_h_p, diff_mse_h_p, pvalue_h_p, pvalue_chisquare_h_p, pvalue_anderson_h_p, pvalue_ttest_h_p, mean_diff_h_p = (
    itemgetter('diff_supremum', 'diff_mse', 'pvalue_kstest', 'pvalue_chisquare', 'pvalue_anderson', 'pvalue_ttest', 'mean_diff')
    (m.PlotDistribution(hist_data_distance, is_show=False, mean_c=c_star)))

print('\nHyman exp:')

beta_h_e, tfv_h_e, n_h_e = itemgetter('beta', 'target_function_value', 'nfev')(m.HymanCalibration(
    0.0001,
    c_star,
    detterence_func=detterence_func_exp
))
m.beta = beta_h_e
m.TripDistribution(detterence_func=lambda x: detterence_func_exp(x, m.beta), eps=eps)
diff_supremum_h_e, diff_mse_h_e, pvalue_h_e, pvalue_chisquare_h_e, pvalue_anderson_h_e, pvalue_ttest_h_e, mean_diff_h_e = (
    itemgetter('diff_supremum', 'diff_mse', 'pvalue_kstest', 'pvalue_chisquare', 'pvalue_anderson', 'pvalue_ttest', 'mean_diff')
    (m.PlotDistribution(hist_data_distance, is_show=False, mean_c=c_star)))

print('\nMED pow:')

beta_m_p, tfv_m_p, n_m_p = itemgetter('beta', 'target_function_value', 'nfev')(m.MedianCalibration(
    statistics.median(hist_data_distance),
    mini,
    maxi,
    detterence_function_type='POW'
))
m.beta = beta_m_p
m.TripDistribution(detterence_func=lambda x: detterence_func_pow(x, m.beta), eps=eps)
diff_supremum_m_p, diff_mse_m_p, pvalue_m_p, pvalue_chisquare_m_p, pvalue_anderson_m_p, pvalue_ttest_m_p, mean_diff_m_p = (
    itemgetter('diff_supremum', 'diff_mse', 'pvalue_kstest', 'pvalue_chisquare', 'pvalue_anderson', 'pvalue_ttest', 'mean_diff')
    (m.PlotDistribution(hist_data_distance, is_show=False, mean_c=c_star)))

def ErrorMethodCompute(model, det_func, hist_compare, minimize_method, eps, c_star):
    beta, tfv, n = itemgetter('beta', 'target_function_value', 'nfev')(m.MSECalibration(
        detterance_func=det_func,
        hist_to_compare=hist_compare,
        minimization_method=minimize_method,
        eps=eps
    ))
    model.beta = beta
    model.TripDistribution(detterence_func=lambda x: det_func(x, m.beta), eps=eps)
    diff_supremum, diff_mse, pvalue, pvalue_chisquare, pvalue_anderson, pvalue_ttest, mean_diff = (
        itemgetter('diff_supremum', 'diff_mse', 'pvalue_kstest', 'pvalue_chisquare', 'pvalue_anderson', 'pvalue_ttest',
                   'mean_diff')
        (m.PlotDistribution(hist_compare, is_show=False, mean_c=c_star)))
    return None


print('\nError MSE pow:')

beta_e_p, tfv_e_p, n_e_p = itemgetter('beta', 'target_function_value', 'nfev')(m.MSECalibration(
    detterance_func=detterence_func_pow,
    hist_to_compare=hist_data_distance,
    minimization_method = minimization_method,
    eps=eps
))
m.beta = beta_e_p
m.TripDistribution(detterence_func=lambda x: detterence_func_pow(x, m.beta), eps=eps)
diff_supremum_e_p, diff_mse_e_p, pvalue_e_p, pvalue_chisquare_e_p, pvalue_anderson_e_p, pvalue_ttest_e_p, mean_diff_e_p = (
    itemgetter('diff_supremum', 'diff_mse', 'pvalue_kstest', 'pvalue_chisquare', 'pvalue_anderson', 'pvalue_ttest', 'mean_diff')
    (m.PlotDistribution(hist_data_distance, is_show=False, mean_c=c_star)))

print('\nError MSE exp:')

beta_e_e, tfv_e_e, n_e_e = itemgetter('beta', 'target_function_value', 'nfev')(m.MSECalibration(
    detterance_func=detterence_func_exp,
    hist_to_compare=hist_data_distance,
    minimization_method = minimization_method,
    eps=eps
))
m.beta = beta_e_e
m.TripDistribution(detterence_func=lambda x: detterence_func_exp(x, m.beta), eps=eps)
diff_supremum_e_e, diff_mse_e_e, pvalue_e_e, pvalue_chisquare_e_e, pvalue_anderson_e_e, pvalue_ttest_e_e, mean_diff_e_e = (
    itemgetter('diff_supremum', 'diff_mse', 'pvalue_kstest', 'pvalue_chisquare', 'pvalue_anderson', 'pvalue_ttest', 'mean_diff')
    (m.PlotDistribution(hist_data_distance, is_show=False, mean_c=c_star)))

print('\nError C_STAR ORTUZAR exp:')

beta_eco_e, tfv_eco_e, n_eco_e = itemgetter('beta', 'target_function_value', 'nfev')(m.MSECalibration(
    detterance_func=detterence_func_exp,
    hist_to_compare=hist_data_distance,
    aux_data={'c_star_ORTUZAR': c_star},
    minimization_method = minimization_method,
    eps=eps
))
m.beta = beta_eco_e
m.TripDistribution(detterence_func=lambda x: detterence_func_exp(x, m.beta), eps=eps)
diff_supremum_eco_e, diff_mse_eco_e, pvalue_eco_e, pvalue_chisquare_eco_e, pvalue_anderson_eco_e, pvalue_ttest_eco_e, mean_diff_eco_e = (
    itemgetter('diff_supremum', 'diff_mse', 'pvalue_kstest', 'pvalue_chisquare', 'pvalue_anderson', 'pvalue_ttest', 'mean_diff')
    (m.PlotDistribution(hist_data_distance, is_show=False, mean_c=c_star)))

print('\nError C_STAR ORTUZAR pow:')

beta_eco_p, tfv_eco_p, n_eco_p = itemgetter('beta', 'target_function_value', 'nfev')(m.MSECalibration(
    detterance_func=detterence_func_pow,
    hist_to_compare=hist_data_distance,
    aux_data={'c_star_ORTUZAR': c_star},
    minimization_method = minimization_method,
    eps=eps
))
m.beta = beta_eco_p
m.TripDistribution(detterence_func=lambda x: detterence_func_pow(x, m.beta), eps=eps)
diff_supremum_eco_p, diff_mse_eco_p, pvalue_eco_p, pvalue_chisquare_eco_p, pvalue_anderson_eco_p, pvalue_ttest_eco_p, mean_diff_eco_p = (
    itemgetter('diff_supremum', 'diff_mse', 'pvalue_kstest', 'pvalue_chisquare', 'pvalue_anderson', 'pvalue_ttest', 'mean_diff')
    (m.PlotDistribution(hist_data_distance, is_show=False, mean_c=c_star)))

print('\nError C_STAR NOT ORTUZAR exp:')

beta_ec_e, tfv_ec_e, n_ec_e = itemgetter('beta', 'target_function_value', 'nfev')(m.MSECalibration(
    detterance_func=detterence_func_exp,
    hist_to_compare=hist_data_distance,
    aux_data={'c_star': c_star},
    minimization_method = minimization_method,
    eps=eps
))
m.beta = beta_ec_e
m.TripDistribution(detterence_func=lambda x: detterence_func_exp(x, m.beta), eps=eps)
diff_supremum_ec_e, diff_mse_ec_e, pvalue_ec_e, pvalue_chisquare_ec_e, pvalue_anderson_ec_e, pvalue_ttest_ec_e, mean_diff_ec_e = (
    itemgetter('diff_supremum', 'diff_mse', 'pvalue_kstest', 'pvalue_chisquare', 'pvalue_anderson', 'pvalue_ttest', 'mean_diff')
    (m.PlotDistribution(hist_data_distance, is_show=False, mean_c=c_star)))

print('\nError C_STAR NOT ORTUZAR pow:')

beta_ec_p, tfv_ec_p, n_ec_p = itemgetter('beta', 'target_function_value', 'nfev')(m.MSECalibration(
    detterance_func=detterence_func_pow,
    hist_to_compare=hist_data_distance,
    aux_data={'c_star': c_star},
    minimization_method = minimization_method,
    eps=eps
))
m.beta = beta_ec_p
m.TripDistribution(detterence_func=lambda x: detterence_func_pow(x, m.beta), eps=eps)
diff_supremum_ec_p, diff_mse_ec_p, pvalue_ec_p, pvalue_chisquare_ec_p, pvalue_anderson_ec_p, pvalue_ttest_ec_p, mean_diff_ec_p = (
    itemgetter('diff_supremum', 'diff_mse', 'pvalue_kstest', 'pvalue_chisquare', 'pvalue_anderson', 'pvalue_ttest', 'mean_diff')
    (m.PlotDistribution(hist_data_distance, is_show=False, mean_c=c_star)))


print('\nError histogram 1 pow:')

beta_hist_p, tfv_hist_p, n_hist_p = itemgetter('beta', 'target_function_value', 'nfev')(m.MSECalibration(
    detterance_func=detterence_func_pow,
    hist_to_compare=hist_data_distance,
    aux_data={'histogram_ERROR1': data_distance},
    minimization_method = minimization_method,
    eps=eps
))
m.beta = beta_hist_p
m.TripDistribution(detterence_func=lambda x: detterence_func_pow(x, m.beta), eps=eps)
diff_supremum_hist_p, diff_mse_hist_p, pvalue_hist_p, pvalue_chisquare_hist_p, pvalue_anderson_hist_p, pvalue_ttest_hist_p, mean_diff_hist_p = (
    itemgetter('diff_supremum', 'diff_mse', 'pvalue_kstest', 'pvalue_chisquare', 'pvalue_anderson', 'pvalue_ttest', 'mean_diff')
    (m.PlotDistribution(hist_data_distance, is_show=False, mean_c=c_star)))


print('\nError histogram 1 exp:')

beta_hist_e, tfv_hist_e, n_hist_e = itemgetter('beta', 'target_function_value', 'nfev')(m.MSECalibration(
    detterance_func=detterence_func_exp,
    hist_to_compare=hist_data_distance,
    aux_data={'histogram_ERROR1': data_distance},
    minimization_method = minimization_method,
    eps=eps
))
m.beta = beta_hist_e
m.TripDistribution(detterence_func=lambda x: detterence_func_exp(x, m.beta), eps=eps)
diff_supremum_hist_e, diff_mse_hist_e, pvalue_hist_e, pvalue_chisquare_hist_e, pvalue_anderson_hist_e, pvalue_ttest_hist_e, mean_diff_hist_e = (
    itemgetter('diff_supremum', 'diff_mse', 'pvalue_kstest', 'pvalue_chisquare', 'pvalue_anderson', 'pvalue_ttest', 'mean_diff')
    (m.PlotDistribution(hist_data_distance, is_show=False, mean_c=c_star)))


print('\nError histogram 1 relative exp:')

beta_hist_e_re, tfv_hist_e_re, n_hist_e_re = itemgetter('beta', 'target_function_value', 'nfev')(m.MSECalibration(
    detterance_func=detterence_func_exp,
    hist_to_compare=hist_data_distance,
    aux_data={'histogram_ERROR1_relative': data_distance},
    minimization_method = minimization_method,
    eps=eps
))
m.beta = beta_hist_e_re
m.TripDistribution(detterence_func=lambda x: detterence_func_exp(x, m.beta), eps=eps)
diff_supremum_hist_e_re, diff_mse_hist_e_re, pvalue_hist_e_re, pvalue_chisquare_hist_e_re, pvalue_anderson_hist_e_re, pvalue_ttest_hist_e_re, mean_diff_hist_e_re = (
    itemgetter('diff_supremum', 'diff_mse', 'pvalue_kstest', 'pvalue_chisquare', 'pvalue_anderson', 'pvalue_ttest', 'mean_diff')
    (m.PlotDistribution(hist_data_distance, is_show=False, mean_c=c_star)))


print('\nError histogram 1 relative pow:')

beta_hist_p_re, tfv_hist_p_re, n_hist_p_re = itemgetter('beta', 'target_function_value', 'nfev')(m.MSECalibration(
    detterance_func=detterence_func_pow,
    hist_to_compare=hist_data_distance,
    aux_data={'histogram_ERROR1_relative': data_distance},
    minimization_method = minimization_method,
    eps=eps
))
m.beta = beta_hist_p_re
m.TripDistribution(detterence_func=lambda x: detterence_func_pow(x, m.beta), eps=eps)
diff_supremum_hist_p_re, diff_mse_hist_p_re, pvalue_hist_p_re, pvalue_chisquare_hist_p_re, pvalue_anderson_hist_p_re, pvalue_ttest_hist_p_re, mean_diff_hist_p_re = (
    itemgetter('diff_supremum', 'diff_mse', 'pvalue_kstest', 'pvalue_chisquare', 'pvalue_anderson', 'pvalue_ttest', 'mean_diff')
    (m.PlotDistribution(hist_data_distance, is_show=False, mean_c=c_star)))


print('\nError histogram 2 pow:')

beta_hist2_p, tfv_hist2_p, n_hist2_p = itemgetter('beta', 'target_function_value', 'nfev')(m.MSECalibration(
    detterance_func=detterence_func_pow,
    hist_to_compare=hist_data_distance,
    aux_data={'histogram_ERROR2': data_distance},
    minimization_method = minimization_method,
    eps=eps
))
m.beta = beta_hist2_p
m.TripDistribution(detterence_func=lambda x: detterence_func_pow(x, m.beta), eps=eps)
diff_supremum_hist2_p, diff_mse_hist2_p, pvalue_hist2_p, pvalue_chisquare_hist2_p, pvalue_anderson_hist2_p, pvalue_ttest_hist2_p, mean_diff_hist2_p = (
    itemgetter('diff_supremum', 'diff_mse', 'pvalue_kstest', 'pvalue_chisquare', 'pvalue_anderson', 'pvalue_ttest', 'mean_diff')
    (m.PlotDistribution(hist_data_distance, is_show=False, mean_c=c_star)))


print('\nError histogram 2 exp:')

beta_hist2_e, tfv_hist2_e, n_hist2_e = itemgetter('beta', 'target_function_value', 'nfev')(m.MSECalibration(
    detterance_func=detterence_func_exp,
    hist_to_compare=hist_data_distance,
    aux_data={'histogram_ERROR2': data_distance},
    minimization_method = minimization_method,
    eps=eps
))
m.beta = beta_hist2_e
m.TripDistribution(detterence_func=lambda x: detterence_func_exp(x, m.beta), eps=eps)
diff_supremum_hist2_e, diff_mse_hist2_e, pvalue_hist2_e, pvalue_chisquare_hist2_e, pvalue_anderson_hist2_e, pvalue_ttest_hist2_e, mean_diff_hist2_e = (
    itemgetter('diff_supremum', 'diff_mse', 'pvalue_kstest', 'pvalue_chisquare', 'pvalue_anderson', 'pvalue_ttest', 'mean_diff')
    (m.PlotDistribution(hist_data_distance, is_show=False, mean_c=c_star)))

print('\nOK.......')

beta_df = beta_df._append({'Method': 'Hyman', 'Exp': beta_h_e, 'Exp TFV': tfv_h_e, 'Pow': beta_h_p, 'Pow TFV': tfv_h_p}, ignore_index=True)
beta_df = beta_df._append({'Method': 'Median', 'Exp': None, 'Exp TFV': None, 'Pow': beta_m_p, 'Pow TFV': tfv_m_p}, ignore_index=True)
beta_df = beta_df._append({'Method': 'Error MSE', 'Exp': beta_e_e, 'Exp TFV': tfv_e_e, 'Pow': beta_e_p, 'Pow TFV': tfv_e_p}, ignore_index=True)
beta_df = beta_df._append({'Method': 'Error ORTUZAR', 'Exp': beta_eco_e, 'Exp TFV': tfv_eco_e, 'Pow': beta_eco_p, 'Pow TFV': tfv_eco_p}, ignore_index=True)
beta_df = beta_df._append({'Method': 'Error NOT ORTUZAR', 'Exp': beta_ec_e, 'Exp TFV': tfv_ec_e, 'Pow': beta_ec_p, 'Pow TFV': tfv_ec_p}, ignore_index=True)
beta_df = beta_df._append({'Method': 'Error Hist 1', 'Exp': beta_hist_e, 'Exp TFV': tfv_hist_e, 'Pow': beta_hist_p, 'Pow TFV': tfv_hist_p}, ignore_index=True)
beta_df = beta_df._append({'Method': 'Error Hist 1 relative', 'Exp': beta_hist_e_re, 'Exp TFV': tfv_hist_e_re, 'Pow': beta_hist_p_re, 'Pow TFV': tfv_hist_p_re}, ignore_index=True)
beta_df = beta_df._append({'Method': 'Error Hist 2', 'Exp': beta_hist2_e, 'Exp TFV': tfv_hist2_e, 'Pow': beta_hist2_p, 'Pow TFV': tfv_hist2_p}, ignore_index=True)

mse_df = mse_df._append({'Method': 'Hyman', 'Exp': diff_mse_h_e, 'Pow': diff_mse_h_p}, ignore_index=True)
mse_df = mse_df._append({'Method': 'Median', 'Exp': None, 'Pow': diff_mse_m_p}, ignore_index=True)
mse_df = mse_df._append({'Method': 'Error MSE', 'Exp': diff_mse_e_e, 'Pow': diff_mse_e_p}, ignore_index=True)
mse_df = mse_df._append({'Method': 'Error ORTUZAR', 'Exp': diff_mse_eco_e, 'Pow': diff_mse_eco_p}, ignore_index=True)
mse_df = mse_df._append({'Method': 'Error NOT ORTUZAR', 'Exp': diff_mse_ec_e, 'Pow': diff_mse_ec_p}, ignore_index=True)
mse_df = mse_df._append({'Method': 'Error Hist 1', 'Exp': diff_mse_hist_e, 'Pow': diff_mse_hist_p}, ignore_index=True)
mse_df = mse_df._append({'Method': 'Error Hist 1 relative', 'Exp': diff_mse_hist_e_re, 'Pow': diff_mse_hist_p_re}, ignore_index=True)
mse_df = mse_df._append({'Method': 'Error Hist 2', 'Exp': diff_mse_hist2_e, 'Pow': diff_mse_hist2_p}, ignore_index=True)

sup_df = sup_df._append({'Method': 'Hyman', 'Exp': diff_supremum_h_e, 'Pow': diff_supremum_h_p}, ignore_index=True)
sup_df = sup_df._append({'Method': 'Median', 'Exp': None, 'Pow': diff_supremum_m_p}, ignore_index=True)
sup_df = sup_df._append({'Method': 'Error MSE', 'Exp': diff_supremum_e_e, 'Pow': diff_supremum_e_p}, ignore_index=True)
sup_df = sup_df._append({'Method': 'Error ORTUZAR', 'Exp': diff_supremum_eco_e, 'Pow': diff_supremum_eco_p}, ignore_index=True)
sup_df = sup_df._append({'Method': 'Error NOT ORTUZAR', 'Exp': diff_supremum_ec_e, 'Pow': diff_supremum_ec_p}, ignore_index=True)
sup_df = sup_df._append({'Method': 'Error Hist 1', 'Exp': diff_supremum_hist_e, 'Pow': diff_supremum_hist_p}, ignore_index=True)
sup_df = sup_df._append({'Method': 'Error Hist 1 relative', 'Exp': diff_supremum_hist_e_re, 'Pow': diff_supremum_hist_p_re}, ignore_index=True)
sup_df = sup_df._append({'Method': 'Error Hist 2', 'Exp': diff_supremum_hist2_e, 'Pow': diff_supremum_hist2_p}, ignore_index=True)

pvalue_df = pvalue_df._append({'Method': 'Hyman', 'Exp': pvalue_h_e, 'Pow': pvalue_h_p}, ignore_index=True)
pvalue_df = pvalue_df._append({'Method': 'Madian', 'Exp': None, 'Pow': pvalue_m_p}, ignore_index=True)
pvalue_df = pvalue_df._append({'Method': 'Error MSE', 'Exp': pvalue_e_e, 'Pow': pvalue_e_p}, ignore_index=True)
pvalue_df = pvalue_df._append({'Method': 'ERROR ORTUZAR', 'Exp': pvalue_eco_e, 'Pow': pvalue_eco_p}, ignore_index=True)
pvalue_df = pvalue_df._append({'Method': 'ERROR NOT ORTUZAR', 'Exp': pvalue_ec_e, 'Pow': pvalue_ec_p}, ignore_index=True)
pvalue_df = pvalue_df._append({'Method': 'Error Hist 1', 'Exp': pvalue_hist_e, 'Pow': pvalue_hist_p}, ignore_index=True)
pvalue_df = pvalue_df._append({'Method': 'Error Hist 1 relative', 'Exp': pvalue_hist_e_re, 'Pow': pvalue_hist_p_re}, ignore_index=True)
pvalue_df = pvalue_df._append({'Method': 'Error Hist 2', 'Exp': pvalue_hist2_e, 'Pow': pvalue_hist2_p}, ignore_index=True)

pvalue_chisquare_df = pvalue_chisquare_df._append({'Method': 'Hyman', 'Exp': pvalue_chisquare_h_e, 'Pow': pvalue_chisquare_h_p}, ignore_index=True)
pvalue_chisquare_df = pvalue_chisquare_df._append({'Method': 'Madian', 'Exp': None, 'Pow': pvalue_chisquare_m_p}, ignore_index=True)
pvalue_chisquare_df = pvalue_chisquare_df._append({'Method': 'Error MSE', 'Exp': pvalue_chisquare_e_e, 'Pow': pvalue_chisquare_e_p}, ignore_index=True)
pvalue_chisquare_df = pvalue_chisquare_df._append({'Method': 'ERROR ORTUZAR', 'Exp': pvalue_chisquare_eco_e, 'Pow': pvalue_chisquare_eco_p}, ignore_index=True)
pvalue_chisquare_df = pvalue_chisquare_df._append({'Method': 'ERROR NOT ORTUZAR', 'Exp': pvalue_chisquare_ec_e, 'Pow': pvalue_chisquare_ec_p}, ignore_index=True)
pvalue_chisquare_df = pvalue_chisquare_df._append({'Method': 'Error Hist 1', 'Exp': pvalue_chisquare_hist_e, 'Pow': pvalue_chisquare_hist_p}, ignore_index=True)
pvalue_chisquare_df = pvalue_chisquare_df._append({'Method': 'Error Hist 1 relative', 'Exp': pvalue_chisquare_hist_e_re, 'Pow': pvalue_chisquare_hist_p_re}, ignore_index=True)
pvalue_chisquare_df = pvalue_chisquare_df._append({'Method': 'Error Hist 2', 'Exp': pvalue_chisquare_hist2_e, 'Pow': pvalue_chisquare_hist2_p}, ignore_index=True)

pvalue_anderson_df = pvalue_anderson_df._append({'Method': 'Hyman', 'Exp': pvalue_anderson_h_e, 'Pow': pvalue_anderson_h_p}, ignore_index=True)
pvalue_anderson_df = pvalue_anderson_df._append({'Method': 'Madian', 'Exp': None, 'Pow': pvalue_anderson_m_p}, ignore_index=True)
pvalue_anderson_df = pvalue_anderson_df._append({'Method': 'Error MSE', 'Exp': pvalue_anderson_e_e, 'Pow': pvalue_anderson_e_p}, ignore_index=True)
pvalue_anderson_df = pvalue_anderson_df._append({'Method': 'ERROR ORTUZAR', 'Exp': pvalue_anderson_eco_e, 'Pow': pvalue_anderson_eco_p}, ignore_index=True)
pvalue_anderson_df = pvalue_anderson_df._append({'Method': 'ERROR NOT ORTUZAR', 'Exp': pvalue_anderson_ec_e, 'Pow': pvalue_anderson_ec_p}, ignore_index=True)
pvalue_anderson_df = pvalue_anderson_df._append({'Method': 'Error Hist 1', 'Exp': pvalue_anderson_hist_e, 'Pow': pvalue_anderson_hist_p}, ignore_index=True)
pvalue_anderson_df = pvalue_anderson_df._append({'Method': 'Error Hist 1 relative', 'Exp': pvalue_anderson_hist_e_re, 'Pow': pvalue_anderson_hist_p_re}, ignore_index=True)
pvalue_anderson_df = pvalue_anderson_df._append({'Method': 'Error Hist 2', 'Exp': pvalue_anderson_hist2_e, 'Pow': pvalue_anderson_hist2_p}, ignore_index=True)

pvalue_ttest_df = pvalue_ttest_df._append({'Method': 'Hyman', 'Exp': pvalue_ttest_h_e, 'Mean Diff Exp': mean_diff_h_p, 'Pow': pvalue_ttest_h_p, 'Mean Diff Pow': mean_diff_h_p}, ignore_index=True)
pvalue_ttest_df = pvalue_ttest_df._append({'Method': 'Madian', 'Exp': None, 'Mean Diff Exp': None, 'Pow': pvalue_ttest_m_p, 'Mean Diff Pow': mean_diff_m_p}, ignore_index=True)
pvalue_ttest_df = pvalue_ttest_df._append({'Method': 'Error MSE', 'Exp': pvalue_ttest_e_e, 'Mean Diff Exp': mean_diff_e_e, 'Pow': pvalue_ttest_e_p, 'Mean Diff Pow': mean_diff_e_p}, ignore_index=True)
pvalue_ttest_df = pvalue_ttest_df._append({'Method': 'ERROR ORTUZAR', 'Exp': pvalue_ttest_eco_e, 'Mean Diff Exp': mean_diff_eco_e, 'Pow': pvalue_ttest_eco_p, 'Mean Diff Pow': mean_diff_eco_p}, ignore_index=True)
pvalue_ttest_df = pvalue_ttest_df._append({'Method': 'ERROR NOT ORTUZAR', 'Exp': pvalue_ttest_ec_e, 'Mean Diff Exp': mean_diff_ec_e, 'Pow': pvalue_ttest_ec_p, 'Mean Diff Pow': mean_diff_ec_p}, ignore_index=True)
pvalue_ttest_df = pvalue_ttest_df._append({'Method': 'Error Hist 1', 'Exp': pvalue_ttest_hist_e, 'Mean Diff Exp': mean_diff_hist_e, 'Pow': pvalue_ttest_hist_p, 'Mean Diff Pow': mean_diff_hist_p}, ignore_index=True)
pvalue_ttest_df = pvalue_ttest_df._append({'Method': 'Error Hist 1 relative', 'Exp': pvalue_ttest_hist_e_re, 'Mean Diff Exp': mean_diff_hist_e_re, 'Pow': pvalue_ttest_hist_p_re, 'Mean Diff Pow': mean_diff_hist_p_re}, ignore_index=True)
pvalue_ttest_df = pvalue_ttest_df._append({'Method': 'Error Hist 2', 'Exp': pvalue_ttest_hist2_e, 'Mean Diff Exp': mean_diff_hist2_e, 'Pow': pvalue_ttest_hist2_p, 'Mean Diff Pow': mean_diff_hist2_p}, ignore_index=True)

n_df = n_df._append({'Method': 'Hyman', 'Exp': n_h_e, 'Pow': n_h_p}, ignore_index=True)
n_df = n_df._append({'Method': 'Madian', 'Exp': None, 'Pow': n_m_p}, ignore_index=True)
n_df = n_df._append({'Method': 'Error MSE', 'Exp': n_e_e, 'Pow': n_e_p}, ignore_index=True)
n_df = n_df._append({'Method': 'ERROR ORTUZAR', 'Exp': n_eco_e, 'Pow': n_eco_p}, ignore_index=True)
n_df = n_df._append({'Method': 'ERROR NOT ORTUZAR', 'Exp': n_ec_e, 'Pow': n_ec_p}, ignore_index=True)
n_df = n_df._append({'Method': 'Error Hist 1', 'Exp': n_hist_e, 'Pow': n_hist_p}, ignore_index=True)
n_df = n_df._append({'Method': 'Error Hist 1 relative', 'Exp': n_hist_e_re, 'Pow': n_hist_p_re}, ignore_index=True)
n_df = n_df._append({'Method': 'Error Hist 2', 'Exp': n_hist2_e, 'Pow': n_hist2_p}, ignore_index=True)

print('\nbeta:')
print(beta_df.to_latex())
print('\nmse error:')
print(mse_df.to_latex())
print('\nsupremum error:')
print(sup_df.to_latex())
print('\npvalue kstest:')
print(pvalue_df.to_latex())
print('\npvalue chisquare:')
print(pvalue_chisquare_df.to_latex())
print('\npvalue anderson:')
print(pvalue_anderson_df.to_latex())
print('\npvalue ttest:')
print(pvalue_ttest_df.to_latex())
print('\nfunctions evaluations:')
print(n_df.to_latex())


