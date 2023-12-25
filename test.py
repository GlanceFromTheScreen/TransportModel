from MainModel import FourStepModel
from synthetic_data import GenerateSyntheticData

if __name__ == '__main__':
    m1 = FourStepModel()
    m1.TripGeneration()
    m1.TripDistribution()
    # m1.HymanCalibration()

    m2 = GenerateSyntheticData(3, 3, 6)
    m2.HymanCalibration(0.01)
