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

exclude_excess_rows_and_cols(m)

print('OK.......')

detterence_func_pow=lambda x, beta: 1 / x**beta if x != 0 else 999999.9
detterence_func_exp=lambda x, beta: math.exp(-beta * x)

beta_df = pd.DataFrame(columns=['Method', 'Exp', 'Exp TFV', 'Pow', 'Pow TFV'])
mse_df = pd.DataFrame(columns=['Method', 'Exp', 'Pow'])
sup_df = pd.DataFrame(columns=['Method', 'Exp', 'Pow'])
pvalue_df = pd.DataFrame(columns=['Method', 'Exp', 'Pow'])
pvalue_chisquare_df = pd.DataFrame(columns=['Method', 'Exp', 'Pow'])

print('\nHyman pow:')

beta_h_p, tfv_h_p = itemgetter('beta', 'target_function_value')(m.HymanCalibration(
    0.0001,
    c_star,
    detterence_func=detterence_func_pow
))
m.beta = beta_h_p
m.TripDistribution(detterence_func=lambda x: detterence_func_pow(x, m.beta), eps=eps)
diff_supremum_h_p, diff_mse_h_p, pvalue_h_p, pvalue_chisquare_h_p = (
    itemgetter('diff_supremum', 'diff_mse', 'pvalue_kstest', 'pvalue_chisquare')
    (m.PlotDistribution(hist_data_distance, is_show=False)))

print('\nHyman exp:')

beta_h_e, tfv_h_e =  itemgetter('beta', 'target_function_value')(m.HymanCalibration(
    0.0001,
    c_star,
    detterence_func=detterence_func_exp
))
m.beta = beta_h_e
m.TripDistribution(detterence_func=lambda x: detterence_func_exp(x, m.beta), eps=eps)
diff_supremum_h_e, diff_mse_h_e, pvalue_h_e, pvalue_chisquare_h_e = (
    itemgetter('diff_supremum', 'diff_mse', 'pvalue_kstest', 'pvalue_chisquare')
    (m.PlotDistribution(hist_data_distance, is_show=False)))

print('\nMED pow:')

beta_m_p, tfv_m_p =  itemgetter('beta', 'target_function_value')(m.MedianCalibration(
    statistics.median(hist_data_distance),
    mini,
    maxi,
    detterence_function_type='POW'
))
m.beta = beta_m_p
m.TripDistribution(detterence_func=lambda x: 1 / x**m.beta if x != 0 else 999999.9, eps=eps)
diff_supremum_m_p, diff_mse_m_p, pvalue_m_p, pvalue_chisquare_m_p = (
    itemgetter('diff_supremum', 'diff_mse', 'pvalue_kstest', 'pvalue_chisquare')
    (m.PlotDistribution(hist_data_distance, is_show=False)))

print('\nError MSE pow:')

beta_e_p, tfv_e_p =  itemgetter('beta', 'target_function_value')(m.MSECalibration(
    detterance_func=lambda x, beta: 1 / x**beta if x != 0 else 999999.9,
    hist_to_compare=hist_data_distance,
    eps=eps
))
m.beta = beta_e_p
m.TripDistribution(detterence_func=lambda x: detterence_func_pow(x, m.beta), eps=eps)
diff_supremum_e_p, diff_mse_e_p, pvalue_e_p, pvalue_chisquare_e_p = (
    itemgetter('diff_supremum', 'diff_mse', 'pvalue_kstest', 'pvalue_chisquare')
    (m.PlotDistribution(hist_data_distance, is_show=False)))

print('\nError MSE exp:')

beta_e_e, tfv_e_e =  itemgetter('beta', 'target_function_value')(m.MSECalibration(
    detterance_func=lambda x, beta: math.exp(-beta * x),
    hist_to_compare=hist_data_distance,
    eps=eps
))
m.beta = beta_e_e
m.TripDistribution(detterence_func=lambda x: detterence_func_exp(x, m.beta), eps=eps)
diff_supremum_e_e, diff_mse_e_e, pvalue_e_e, pvalue_chisquare_e_e = (
    itemgetter('diff_supremum', 'diff_mse', 'pvalue_kstest', 'pvalue_chisquare')
    (m.PlotDistribution(hist_data_distance, is_show=False)))

print('\nError C_STAR ORTUZAR exp:')

beta_eco_e, tfv_eco_e =  itemgetter('beta', 'target_function_value')(m.MSECalibration(
    detterance_func=detterence_func_exp,
    hist_to_compare=hist_data_distance,
    aux_data={'c_star_ORTUZAR': c_star},
    eps=eps
))
m.beta = beta_eco_e
m.TripDistribution(detterence_func=lambda x: detterence_func_exp(x, m.beta), eps=eps)
diff_supremum_eco_e, diff_mse_eco_e, pvalue_eco_e, pvalue_chisquare_eco_e = (
    itemgetter('diff_supremum', 'diff_mse', 'pvalue_kstest', 'pvalue_chisquare')
    (m.PlotDistribution(hist_data_distance, is_show=False)))

print('\nError C_STAR ORTUZAR pow:')

beta_eco_p, tfv_eco_p =  itemgetter('beta', 'target_function_value')(m.MSECalibration(
    detterance_func=detterence_func_pow,
    hist_to_compare=hist_data_distance,
    aux_data={'c_star_ORTUZAR': c_star},
    eps=eps
))
m.beta = beta_eco_p
m.TripDistribution(detterence_func=lambda x: detterence_func_pow(x, m.beta), eps=eps)
diff_supremum_eco_p, diff_mse_eco_p, pvalue_eco_p, pvalue_chisquare_eco_p = (
    itemgetter('diff_supremum', 'diff_mse', 'pvalue_kstest', 'pvalue_chisquare')
    (m.PlotDistribution(hist_data_distance, is_show=False)))

print('\nError C_STAR NOT ORTUZAR exp:')

beta_ec_e, tfv_ec_e =  itemgetter('beta', 'target_function_value')(m.MSECalibration(
    detterance_func=detterence_func_exp,
    hist_to_compare=hist_data_distance,
    aux_data={'c_star': c_star},
    eps=eps
))
m.beta = beta_ec_e
m.TripDistribution(detterence_func=lambda x: detterence_func_exp(x, m.beta), eps=eps)
diff_supremum_ec_e, diff_mse_ec_e, pvalue_ec_e, pvalue_chisquare_ec_e = (
    itemgetter('diff_supremum', 'diff_mse', 'pvalue_kstest', 'pvalue_chisquare')
    (m.PlotDistribution(hist_data_distance, is_show=False)))

print('\nError C_STAR NOT ORTUZAR pow:')

beta_ec_p, tfv_ec_p =  itemgetter('beta', 'target_function_value')(m.MSECalibration(
    detterance_func=detterence_func_pow,
    hist_to_compare=hist_data_distance,
    aux_data={'c_star': c_star},
    eps=eps
))
m.beta = beta_ec_p
m.TripDistribution(detterence_func=lambda x: detterence_func_pow(x, m.beta), eps=eps)
diff_supremum_ec_p, diff_mse_ec_p, pvalue_ec_p, pvalue_chisquare_ec_p = (
    itemgetter('diff_supremum', 'diff_mse', 'pvalue_kstest', 'pvalue_chisquare')
    (m.PlotDistribution(hist_data_distance, is_show=False)))

print('\nOK.......')

beta_df = beta_df._append({'Method': 'Hyman', 'Exp': beta_h_e, 'Exp TFV': tfv_h_e, 'Pow': beta_h_p, 'Pow TFV': tfv_h_p}, ignore_index=True)
beta_df = beta_df._append({'Method': 'Median', 'Exp': None, 'Exp TFV': None, 'Pow': beta_m_p, 'Pow TFV': tfv_m_p}, ignore_index=True)
beta_df = beta_df._append({'Method': 'Error MSE', 'Exp': beta_e_e, 'Exp TFV': tfv_e_e, 'Pow': beta_e_p, 'Pow TFV': tfv_e_p}, ignore_index=True)
beta_df = beta_df._append({'Method': 'Error ORTUZAR', 'Exp': beta_eco_e, 'Exp TFV': tfv_eco_e, 'Pow': beta_eco_p, 'Pow TFV': tfv_eco_p}, ignore_index=True)
beta_df = beta_df._append({'Method': 'Error NOT ORTUZAR', 'Exp': beta_ec_e, 'Exp TFV': tfv_ec_e, 'Pow': beta_ec_p, 'Pow TFV': tfv_ec_p}, ignore_index=True)

mse_df = mse_df._append({'Method': 'Hyman', 'Exp': diff_mse_h_e, 'Pow': diff_mse_h_p}, ignore_index=True)
mse_df = mse_df._append({'Method': 'Median', 'Exp': None, 'Pow': diff_mse_m_p}, ignore_index=True)
mse_df = mse_df._append({'Method': 'Error MSE', 'Exp': diff_mse_e_e, 'Pow': diff_mse_e_p}, ignore_index=True)
mse_df = mse_df._append({'Method': 'Error ORTUZAR', 'Exp': diff_mse_eco_e, 'Pow': diff_mse_eco_p}, ignore_index=True)
mse_df = mse_df._append({'Method': 'Error NOT ORTUZAR', 'Exp': diff_mse_ec_e, 'Pow': diff_mse_ec_p}, ignore_index=True)

sup_df = sup_df._append({'Method': 'Hyman', 'Exp': diff_supremum_h_e, 'Pow': diff_supremum_h_p}, ignore_index=True)
sup_df = sup_df._append({'Method': 'Median', 'Exp': None, 'Pow': diff_supremum_m_p}, ignore_index=True)
sup_df = sup_df._append({'Method': 'Error MSE', 'Exp': diff_supremum_e_e, 'Pow': diff_supremum_e_p}, ignore_index=True)
sup_df = sup_df._append({'Method': 'Error ORTUZAR', 'Exp': diff_supremum_eco_e, 'Pow': diff_supremum_eco_p}, ignore_index=True)
sup_df = sup_df._append({'Method': 'Error NOT ORTUZAR', 'Exp': diff_supremum_ec_e, 'Pow': diff_supremum_ec_p}, ignore_index=True)

pvalue_df = pvalue_df._append({'Method': 'Hyman', 'Exp': pvalue_h_e, 'Pow': pvalue_h_p}, ignore_index=True)
pvalue_df = pvalue_df._append({'Method': 'Madian', 'Exp': None, 'Pow': pvalue_m_p}, ignore_index=True)
pvalue_df = pvalue_df._append({'Method': 'Error MSE', 'Exp': pvalue_e_e, 'Pow': pvalue_e_p}, ignore_index=True)
pvalue_df = pvalue_df._append({'Method': 'ERROR ORTUZAR', 'Exp': pvalue_eco_e, 'Pow': pvalue_eco_p}, ignore_index=True)
pvalue_df = pvalue_df._append({'Method': 'ERROR NOT ORTUZAR', 'Exp': pvalue_ec_e, 'Pow': pvalue_ec_p}, ignore_index=True)

pvalue_chisquare_df = pvalue_chisquare_df._append({'Method': 'Hyman', 'Exp': pvalue_chisquare_h_e, 'Pow': pvalue_chisquare_h_p}, ignore_index=True)
pvalue_chisquare_df = pvalue_chisquare_df._append({'Method': 'Madian', 'Exp': None, 'Pow': pvalue_chisquare_m_p}, ignore_index=True)
pvalue_chisquare_df = pvalue_chisquare_df._append({'Method': 'Error MSE', 'Exp': pvalue_chisquare_e_e, 'Pow': pvalue_chisquare_e_p}, ignore_index=True)
pvalue_chisquare_df = pvalue_chisquare_df._append({'Method': 'ERROR ORTUZAR', 'Exp': pvalue_chisquare_eco_e, 'Pow': pvalue_chisquare_eco_p}, ignore_index=True)
pvalue_chisquare_df = pvalue_chisquare_df._append({'Method': 'ERROR NOT ORTUZAR', 'Exp': pvalue_chisquare_ec_e, 'Pow': pvalue_chisquare_ec_p}, ignore_index=True)

print('\nbeta:')
print(beta_df)
print('\nmse error:')
print(mse_df)
print('\nsupremum error:')
print(sup_df)
print('\npvalue kstest:')
print(pvalue_df)
print('\npvalue chisquare:')
print(pvalue_chisquare_df)
