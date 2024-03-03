import pandas as pd
import numpy as np
import pickle

###############################
# GIVEN DATA HAS SQUARE SHAPE,
# SO MxN -> NxN:
# |O_vector| = |D_vector|
###############################

file_path = 'data/test_data.xlsx'
print('READING DATA...')

###############################
# TIME MATRIX IT, TIME MATRIX PT, TRIP DISTANCE
###############################

c_table = pd.read_excel(file_path, sheet_name=0)
size = max(c_table['FROMZONENO'])
c_matrix_time_it = np.zeros((size, size))
c_matrix_time_pt = np.zeros((size, size))
c_matrix_distance = np.zeros((size, size))

for index, row in c_table.iterrows():
    c_matrix_time_it[int(row['FROMZONENO'] - 1)][int(row['TOZONENO']) - 1] = row['ВРЕМЯ ИТ']
    c_matrix_time_pt[int(row['FROMZONENO'] - 1)][int(row['TOZONENO']) - 1] = row['ВРЕМЯ ОТ']
    c_matrix_distance[int(row['FROMZONENO'] - 1)][int(row['TOZONENO']) - 1] = row['РАССТОЯНИЕ']

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
# C* - TIME and DISTANCE
###############################

distribution_time = pd.read_excel(file_path, sheet_name=2)
c_star_time = 0
for index, row in distribution_time.iterrows():
    c_star_time += (row['UPPERLIMIT'] + row['LOWERLIMIT']) / 2 * row['SHARE']

distribution_distance = pd.read_excel(file_path, sheet_name=3)
c_star_distance = 0
for index, row in distribution_distance.iterrows():
    c_star_distance += (row['UPPERLIMIT'] + row['LOWERLIMIT']) / 2 * row['SHARE']

print('SERIALIZING DATA...')

with open("data/c_matrix_time_it.pickle", "wb") as file:
    pickle.dump(c_matrix_time_it, file)

with open("data/c_matrix_time_pt.pickle", "wb") as file:
    pickle.dump(c_matrix_time_pt, file)

with open("data/c_matrix_distance.pickle", "wb") as file:
    pickle.dump(c_matrix_distance, file)

with open("data/origin.pickle", "wb") as file:
    pickle.dump(O_vector, file)

with open("data/destination.pickle", "wb") as file:
    pickle.dump(D_vector, file)

with open("data/c_star_time.pickle", "wb") as file:
    pickle.dump(c_star_time, file)

with open("data/c_star_distance.pickle", "wb") as file:
    pickle.dump(c_star_distance, file)











