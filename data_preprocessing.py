import numpy as np

def exclude_excess_rows_and_cols(model):
    i = 0
    while i < model.O.shape[0]:
        if model.O[i] == 0.:
            model.c = np.delete(model.c, i, axis=0)
            model.O = np.delete(model.O, i, axis=0)
            i -= 1
        i += 1

    j = 0
    while j < model.D.shape[0]:
        if model.D[j] == 0.:
            model.c = np.delete(model.c, j, axis=1)
            model.D = np.delete(model.D, j, axis=0)
            j -= 1
        j += 1


