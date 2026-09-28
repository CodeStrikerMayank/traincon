import pandas as pd
import numpy as np


class extraction:

    def __init__(self):
        self.data = pd.read_csv("data.csv")

    def data_sorting(self):
        featrues = ["Practice Hours (x1​)", "Written Test Score (x2​)"]
        self.x1 = self.data[featrues].to_numpy()
        self.y1 = self.data["Result (y)"].to_numpy()


class logic(extraction):

    def __init__(self):
        super().__init__()
        self.bias = 0
        self.weights = np.zeros((2, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def Training(self):
        self.data_sorting()
        print(self.x1)
        print(self.y1)


p = logic()
p.Training()
