import numpy as np
import math
from MainModel import FourStepModel
from synthetic_data import GenerateSyntheticData, GenerateSyntheticDataReversed
import matplotlib.pyplot as plt
import statistics
import pickle
from data_preprocessing import exclude_excess_rows_and_cols

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
with open("data/data_distance.pickle", "rb") as file:
    data_distance = pickle.load(file)

print('OK.......')

exclude_excess_rows_and_cols(m)

# detterence_func=lambda x, beta: 1 / x**beta if x != 0 else 999999.9
detterence_func=lambda x, beta: math.exp(-beta[0] * x)
# detterence_func=lambda x, params: (x ** (-params[1])) * math.exp(-params[0] * x) if x != 0 else 999999.9

# beta = m.MSECalibration(detterance_func=detterence_func, hist_to_compare=hist_data_distance)['beta']
# m.beta = beta
# m.TripDistribution(detterence_func=lambda x: detterence_func(x, beta), eps=0.1)
# m.PlotDistribution(hist_data_distance)

beta = m.MSECalibration(
    detterance_func=detterence_func,
    hist_to_compare=hist_data_distance,
    aux_data={'histogram_ERROR1_relative': data_distance},
    eps=0.01,
    minimization_method='trial'
)['beta']
m.beta = beta
m.TripDistribution(detterence_func=lambda x: detterence_func(x, beta), eps=0.1)
m.PlotDistribution(hist_data_distance, is_show=True, mean_c=c_star)

import matplotlib.pyplot as plt
import numpy as np

# x = [1.0,
#      0.7639320225002102,
#      1.2360679774997896,
#      0.4721359549995794,
#      0.2917960675006309,
#      0.18033988749894847,
#      0.08127447911449198,
#      0.07729229189466173,
#      0.07975341894608227,
#      0.06888370749726609,
#      0.08188387248891559,
#      ]
# y = [0.02190112764314156,
#      0.01972191745693215,
#      0.023215685856634926,
#      0.014432277540195024,
#      0.008310210875672065,
#      0.0031320344495503073,
#      3.181174320983287e-05,
#      3.940545964234242e-05,
#      3.2969011246661164e-05,
#      0.00010536988492145479,
#      3.1985325888689135e-05]
#
# plt.loglog(x, y, '.')
# plt.show()





