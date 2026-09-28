import pandas as pd
import numpy as np


class extraction:

    def __init__(self):
        self.data = pd.read_csv("data.csv")

    def data_sorting(self):
        self.x = np.array(self.data[["Practice Hours (x)"],])
        print(self.x)


p1 = extraction()
p1.data_sorting()
