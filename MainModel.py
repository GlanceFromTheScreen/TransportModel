import numpy as np
from trip_distribution import  TripDistribution
from hyman_calibration import HymanCalibration
from median_calibration import MedianCalibration
from graphic_distribution import PlotDistribution

class FourStepModel:
    def __init__(self):
        self.O = None
        self.D = None
        self.c = None
        self.T = None
        self.beta = 1
        self.T_split = None
        self.Final_flows_distribution = None

    def TripGeneration(self):
        """
        Generates: origins O, destinations D, cost matrix c
        """
        self.O = np.array([98, 106, 122])
        self.D = np.array([102, 224])
        # self.c = np.array([[1.0, 8.2, 1.8],
        #                    [6.2, 6.0, 9.5],
        #                    [11.8, 1.58, 1.0]])
        self.c = np.array([[1.0, 2.0],
                           [3.0, 4.0],
                           [5.0, 6.0]])
        # self.c = np.array([[1., 1.2, 1.8],
        #                    [1.2, 1., 1.5],
        #                    [1.8, 1.5, 1.]])


    TripDistribution = TripDistribution
    HymanCalibration = HymanCalibration
    MedianCalibration = MedianCalibration
    PlotDistribution = PlotDistribution


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
