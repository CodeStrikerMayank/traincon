import pandas as pd
import numpy as np


class extraction:

    def __init__(self):
        self.data = pd.read_csv("data.csv")

    def data_sorting(self):
        featrues = ["Practice Hours (x1​)", "Written Test Score (x2​)"]
        self.x1 = self.data[featrues].to_numpy()
        self.y1 = self.data[["Result (y)"]].to_numpy()

    def check_sort(self):
        data = pd.read("info.json")
        self.weights = np.array(data["weights"][0])
        self.bias  = data["bias"][0]

    def save_file(self):
        data = [{"weights": self.weights,"bias": self.bias}]
        df = pd.DataFrame(data)
        df.to_json("info.json",orient="split",indent=4)

class logic(extraction):

    def __init__(self):
        super().__init__()
        self.bias = 0
        self.weights = np.zeros((2, 1))
        self.echoes = 10000
        self.lrt = .01

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def Training(self):
        self.data_sorting()
        n = len(self.x1)
        for echoe in range(self.echoes):
            z = np.dot(self.x1,self.weights)+self.bias
            pred = self.sigmoid(z)
            loss = pred-self.y1
            gradient = loss*pred*(1-pred)*1
            dw = np.dot(self.x1.T,gradient)/n;
            db = np.sum(gradient)/n
            self.weights -= dw* self.lrt
            self.bias -= db* self.lrt
            if echoe%10 ==0:
                print(f"Baised is :{self.bias}")
        print("Weights is what ",self.weights)
        self.save_file()
p = logic()
p.Training()
