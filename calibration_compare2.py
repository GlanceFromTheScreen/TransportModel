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

def filter_dict(dict, arr):
    return {k: v for k, v in dict.items() if k in arr}

def format_table(str_table):
    return str_table.replace("\\toprule", "\\hline").replace("\\midrule", "\\hline").replace("\\bottomrule", "\\hline")

def short_method(str_method):
    if str_method in ['histogram_ERROR1', 'histogram_ERROR1_relative', 'histogram_ERROR2']:
        return str_method[10:].replace("_", "")
    return str_method.replace("_", "")


m = FourStepModel()

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

DATA = 'distance'

print('OK.......')

detterence_func_pow=lambda x, beta: 1 / x**beta[0] if x != 0 else 999999.9
detterence_func_exp=lambda x, beta: math.exp(-beta[0] * x)
detterence_func_combined=lambda x, params: (x ** (-params[1])) * math.exp(-params[0] * x) if x != 0 else 999999.9

methods = {
    'Hyman': None,
    'MED': None,
    'MSE': None,
    'c_star_ORTUZAR': c_star,
    # 'c_star': c_star,
    'histogram_ERROR1': data_distance,
    'histogram_ERROR1_relative': data_distance,
    'histogram_ERROR2': data_distance
}
# minimization = ['trial', 'nelder-mead', 'powell', 'SLSQP']
minimization = ['trial', 'nelder-mead', 'powell', 'GA', 'SLSQP']
# minimization = ['trial', 'SLSQP']
det_functions = {
    'pow': lambda x, beta: 1 / x**beta[0] if x != 0 else 999999.9,
    'exp': lambda x, beta: math.exp(-beta[0] * x),
    'combined': lambda x, params: (x ** (-params[1])) * math.exp(-params[0] * x) if x != 0 else 999999.9
}

def getMethodDict(model, hist_to_compare, method='Hyman', min_method='trial', c_star=c_star, det_func='pow', eps=0.01, aux_data_value=None):

    m = model

    methodDict = {}
    methodDict['Method'] = short_method(method)
    methodDict['Min Method'] = min_method

    if method == 'Hyman':
        if det_func not in ['pow', 'exp']:
            return
        methodDict['beta'], methodDict['TFV'], methodDict['N'] = itemgetter('beta', 'target_function_value', 'nfev')(m.HymanCalibration(
            eps,
            c_star,
            detterence_func=det_functions[det_func]
        ))
    elif method == 'MED':
        if det_func != 'pow':
            return
        methodDict['beta'], methodDict['TFV'], methodDict['N'] = itemgetter('beta', 'target_function_value', 'nfev')(m.MedianCalibration(
            statistics.median(hist_to_compare),
            mini,
            maxi,
            detterence_function_type=det_func
        ))
    else:
        if det_func == 'combined' and min_method in ['trial', 'gold', 'uniform']:
            return
        methodDict['beta'], methodDict['TFV'], methodDict['N'] = itemgetter('beta', 'target_function_value', 'nfev')(m.MSECalibration(
            detterance_func=det_functions[det_func],
            hist_to_compare=hist_to_compare,
            aux_data={method: aux_data_value},
            minimization_method=min_method,
            eps=eps
        ))

    m.beta = methodDict['beta']
    m.TripDistribution(detterence_func=lambda x: det_functions[det_func](x, m.beta), eps=eps)
    (methodDict['diff sup'],
     methodDict['diff mse'],
     methodDict['pvalue ks'],
     methodDict['pvalue chi'],
     methodDict['pvalue and'],
     methodDict['pvalue tt'],
     methodDict['diff mean']) = (
        itemgetter('diff_supremum', 'diff_mse', 'pvalue_kstest', 'pvalue_chisquare', 'pvalue_anderson',
                   'pvalue_ttest', 'mean_diff')
        (m.PlotDistribution(
            hist_to_compare,
            is_show=False,
            is_save=[True, DATA + "_" + det_func + "_" + method + "_" + min_method],
            mean_c=c_star,
            ITERATIONS=methodDict['N']
        )))

    for key, value in methodDict.items():
        if type(value).__name__ == 'list':
            methodDict[key] = list(map(lambda x: round(x, 5), value))
        elif type(value).__name__ != 'str':
            methodDict[key] = round(value, 5)

    print(methodDict)
    return methodDict


l1 = ['Method', 'Min Method', 'beta', 'TFV', 'N']
df1 = pd.DataFrame(columns=l1)

l2 = ['Method', 'Min Method', 'diff sup', 'diff mse', 'diff mean']
df2 = pd.DataFrame(columns=l2)

l3 = ['Method', 'Min Method', 'pvalue ks', 'pvalue and', 'pvalue tt']
df3 = pd.DataFrame(columns=l3)

# x1 = {'Method': 'MSE', 'Min Method': 'nelder-mead', 'beta': [0.08703, -0.06166], 'TFV': 3e-05, 'N': 74, 'diff sup': 0.02073, 'diff mse': 3e-05, 'pvalue ks': 0.5023, 'pvalue chi': 0.0, 'pvalue and': 0.11836, 'pvalue tt': 0.53493, 'diff mean': 0.21346}
# x2 = {'Method': 'MSE', 'Min Method': 'powell', 'beta': [0.08714, -0.06248], 'TFV': 3e-05, 'N': 83, 'diff sup': 0.02066, 'diff mse': 3e-05, 'pvalue ks': 0.50591, 'pvalue chi': 0.0, 'pvalue and': 0.11832, 'pvalue tt': 0.53051, 'diff mean': 0.21523}
# x3 = {'Method': 'MSE', 'Min Method': 'GA', 'beta': [-0.06033, 1.24988], 'TFV': 0.00274, 'N': 100, 'diff sup': 0.10304, 'diff mse': 0.00274, 'pvalue ks': 0.0, 'pvalue chi': 0.0, 'pvalue and': 0.001, 'pvalue tt': 0.0, 'diff mean': 2.55413}
# x4 = {'Method': 'MSE', 'Min Method': 'SLSQP', 'beta': [0.05745, 0.26016], 'TFV': 9e-05, 'N': 16, 'diff sup': 0.03532, 'diff mse': 9e-05, 'pvalue ks': 0.14516, 'pvalue chi': 0.0, 'pvalue and': 0.01018, 'pvalue tt': 0.89743, 'diff mean': 0.08316}
#
# for item in [x1, x2, x3, x4]:
#     df1 = df1._append(filter_dict(item, l1), ignore_index=True)
#     df2 = df2._append(filter_dict(item, l2), ignore_index=True)
#     df3 = df3._append(filter_dict(item, l3), ignore_index=True)
#
# print("---1---\n")
# print(format_table(df1.to_latex(column_format="|p{1cm}|p{4cm}|p{3cm}|p{4cm}|p{2cm}|p{1cm}|", float_format="%.5f")))
# print("---2---\n")
# print(format_table(df2.to_latex(column_format="|p{1cm}|p{4cm}|p{3cm}|p{2cm}|p{2cm}|p{2cm}|", float_format="%.5f")))
# print("---3---\n")
# print(format_table(df3.to_latex(column_format="|p{1cm}|p{4cm}|p{3cm}|p{2cm}|p{2cm}|p{2cm}|", float_format="%.5f")))

counted_Hyman = False
counted_MED = False
for key, value in methods.items():
    for item in minimization:
        if (key == 'Hyman' and counted_Hyman) or (key=='MED' and counted_MED):
            continue
        d = getMethodDict(model=m, hist_to_compare=hist_data_distance, method=key, min_method=item, aux_data_value=value, det_func='combined', eps=0.001)
        if d:
            df1 = df1._append(filter_dict(d, l1), ignore_index=True)
            df2 = df2._append(filter_dict(d, l2), ignore_index=True)
            df3 = df3._append(filter_dict(d, l3), ignore_index=True)
        if key == 'Hyman':
            counted_Hyman = True
        if key == 'MED':
            counted_MED = True

print("---1---\n")
print(format_table(df1.to_latex(column_format="|p{1cm}|p{4cm}|p{3cm}|p{4cm}|p{2cm}|p{1cm}|", float_format="%.5f")))
print("---2---\n")
print(format_table(df2.to_latex(column_format="|p{1cm}|p{4cm}|p{3cm}|p{2cm}|p{2cm}|p{2cm}|", float_format="%.5f")))
print("---3---\n")
print(format_table(df3.to_latex(column_format="|p{1cm}|p{4cm}|p{3cm}|p{2cm}|p{2cm}|p{2cm}|", float_format="%.5f")))

