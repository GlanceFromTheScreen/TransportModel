from MainModel import FourStepModel
from synthetic_data import GenerateSyntheticData, GenerateSyntheticDataReversed
from lib_1d_minimization.One_D_Problem_file import One_D_Problem

if __name__ == '__main__':

    # p1 = One_D_Problem()
    # p1.target_function = lambda a: (a-1.7)**2
    # item = 0.001
    # ans, n = p1.uniform_search_method(6, item)
    # print('Метод равномерного поиска:\n', n, ' ', ((ans / item) // 1) * item, '+-', item / 2)

    m1 = FourStepModel()
    m1.TripGeneration()
    print(m1.TripDistribution())

    m2 = GenerateSyntheticDataReversed(10, 10, 3.543, noise=0)
    m2.PlotDistribution()
    print(m2.HymanCalibration(0.01))
