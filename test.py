from MainModel import FourStepModel
from synthetic_data import GenerateSyntheticData

if __name__ == '__main__':
    m1 = FourStepModel()
    m1.TripGeneration()
    print(m1.TripDistribution())

    m2 = GenerateSyntheticData(5, 10, 3.543, noise=0)
    print(m2.HymanCalibration(0.01))
