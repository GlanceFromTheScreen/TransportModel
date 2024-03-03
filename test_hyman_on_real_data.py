import numpy as np
import math
from MainModel import FourStepModel
from synthetic_data import GenerateSyntheticData, GenerateSyntheticDataReversed
import matplotlib.pyplot as plt
import statistics
import pickle
from data_preprocessing import exclude_excess_rows_and_cols

m = FourStepModel()

with open("data/c_matrix_time_it.pickle", "rb") as file:
    m.c = pickle.load(file)

with open("data/origin.pickle", "rb") as file:
    m.O = O_vector = pickle.load(file)

with open("data/destination.pickle", "rb") as file:
    m.D = pickle.load(file)

with open("data/c_star_time.pickle", "rb") as file:
    c_star = pickle.load(file)

print('OK.......')

exclude_excess_rows_and_cols(m)

print(m.HymanCalibration(0.0001, c_star))


