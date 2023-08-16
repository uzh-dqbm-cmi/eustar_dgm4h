import numpy as np
import pandas as pd
from .organ import Organ
from pythae.ssc.variable import Variable


class KIDNEY(Organ):
    def __init__(self):
        super().__init__("KIDNEY")

    def init_variables(self):
        self.variables = [
            Variable(self.time_name, kind="continuous", encoding="mean_var", xyt="t"),
            Variable("Renal crisis", kind="binary", encoding="one_hot", xyt="x"),
        ]

        self.variable_names = [var.name for var in self.variables]

        return self.variables

    def init_labels(self):
        self.labels = []
        return

    def create_labels(self, df):
        return
