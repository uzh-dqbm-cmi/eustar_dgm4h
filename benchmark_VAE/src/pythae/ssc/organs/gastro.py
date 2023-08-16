import numpy as np
import pandas as pd
from .organ import Organ
from pythae.ssc.variable import Variable


class GASTRO(Organ):
    def __init__(self):
        super().__init__("GASTRO")

    def init_variables(self):
        self.variables = [
            Variable(self.time_name, kind="continuous", encoding="mean_var", xyt="t"),
            Variable(
                "Stomach symptoms (early satiety, vomiting)",
                kind="binary",
                encoding="one_hot",
                xyt="x",
            ),
            Variable(
                "Intestinal symptoms (diarrhea, bloating, constipation)",
                kind="binary",
                encoding="one_hot",
                xyt="x",
            ),
        ]

        self.variable_names = [var.name for var in self.variables]

        return self.variables

    def init_labels(self):
        self.labels = []
        return

    def create_labels(self, df):
        return
