from MainModel import FourStepModel
from synthetic_data import GenerateSyntheticData, GenerateSyntheticDataReversed

if __name__ == '__main__':
    m1 = FourStepModel()
    m1.TripGeneration()
    print(m1.TripDistribution())

    m2 = GenerateSyntheticDataReversed(10, 10, 3.543, noise=0)
    m2.PlotDistribution()
    print(m2.HymanCalibration(0.01))
