import pandas as pd
import numpy as np
import pickle

###############################
# GIVEN DATA HAS SQUARE SHAPE,
# SO MxN -> NxN:
# |O_vector| = |D_vector|
###############################

###############################
# TIME MATRIX
###############################

print('READING DATA...')

file_path = 'data/test_data.xlsx'
c_table = pd.read_excel(file_path, sheet_name=0)
size = max(c_table['FROMZONENO'])
c_matrix = np.zeros((size, size))

for index, row in c_table.iterrows():
    c_matrix[int(row['FROMZONENO'] - 1)][int(row['TOZONENO']) - 1] = row['ВРЕМЯ ИТ']

###############################
# ORIGIN AND DESTINATION
###############################

districts = pd.read_excel(file_path, sheet_name=1)
O_vector = np.zeros(size)
D_vector = np.zeros(size)

for index, row in districts.iterrows():
    O_vector[index] = row['PRODUCTION']
    D_vector[index] = row['ATTRACTION']

###############################
# C* - TIME
###############################

distribution_time = pd.read_excel(file_path, sheet_name=2)
c_star_time = 0

for index, row in distribution_time.iterrows():
    c_star_time += (row['UPPERLIMIT'] + row['LOWERLIMIT']) / 2 * row['SHARE']

print('SERIALIZING DATA...')

with open("data/c_matrix.pickle", "wb") as file:
    pickle.dump(c_matrix, file)

with open("data/origin.pickle", "wb") as file:
    pickle.dump(O_vector, file)

with open("data/destination.pickle", "wb") as file:
    pickle.dump(D_vector, file)

with open("data/c_star_time.pickle", "wb") as file:
    pickle.dump(c_star_time, file)











