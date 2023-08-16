import numpy as np
import pandas as pd
from .organ import Organ
from pythae.ssc.variable import Variable


class MUSCLE(Organ):
    def __init__(self):
        super().__init__("MUSCLE")

    def init_variables(self):
        self.variables = [
            Variable(self.time_name, kind="continuous", encoding="mean_var", xyt="t"),
            Variable("Muscle weakness", kind="binary", encoding="one_hot", xyt="x"),
            Variable(
                "Proximal muscle weakness not explainable by other causes",
                kind="binary",
                encoding="one_hot",
                xyt="x",
            ),
            Variable("Muscle atrophy", kind="binary", encoding="one_hot", xyt="x"),
            Variable("Myalgia", kind="binary", encoding="one_hot", xyt="x"),
        ]

        self.variable_names = [var.name for var in self.variables]

        return self.variables

    def init_labels(self):
        self.labels = []
        return

    def create_labels(self, df):
        return
