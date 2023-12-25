import numpy as np
from trip_distribution import  TripDistribution
from hyman_calibration import HymanCalibration

class FourStepModel:
    def __init__(self):
        self.O = None
        self.D = None
        self.c = None
        self.T = None
        self.betta = 1
        self.T_split = None
        self.Final_flows_distribution = None

    def TripGeneration(self):
        """
        Generates: origins O, destinations D, cost matrix c
        """
        self.O = np.array([98, 106, 122])
        self.D = np.array([102, 118, 106])
        self.c = np.array([[1.0, 6.2, 1.8],
                           [6.2, 1.0, 1.5],
                           [1.8, 1.5, 1.0]])
        pass

    TripDistribution = TripDistribution
    HymanCalibration = HymanCalibration


    def ModalSplit(self):
        """
        Takes: the list of correspondence matrices T (for each layer)
        Generates: the list of correspondence matrices T_split (for each layer and for each transportation method)
        """
        pass

    def TripAssignment(self):
        """
        Takes: the list of correspondence matrices T_split (for each layer and for each transportation method)
        Generates: Final_flows_distribution
        """
        pass
