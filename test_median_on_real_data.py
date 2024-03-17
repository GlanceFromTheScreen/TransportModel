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
with open("data/min_max_distance.pickle", "rb") as file:
    mini, maxi = pickle.load(file)
with open("data/hist_data_distance.pickle", "rb") as file:
    hist_data_distance = pickle.load(file)

exclude_excess_rows_and_cols(m)
beta = m.MedianCalibration(statistics.median(hist_data_distance), mini, maxi)['beta']
m.beta = beta
m.TripDistribution(detterence_func=lambda x: 1 / x**beta if x != 0 else 999999.9, eps=0.1)
m.PlotDistribution(hist_data_distance)




