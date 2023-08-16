import numpy as np
import pandas as pd
from .organ import Organ
from pythae.ssc.variable import Variable


class STATIC(Organ):
    def __init__(self):
        super().__init__("STATIC")

    def init_variables(self):
        self.variables = [
            Variable("Sex", kind="binary", encoding="one_hot", xyt="s"),
            Variable("Height", kind="continuous", encoding="mean_var", xyt="s"),
            Variable("Height", kind="continuous", encoding="mean_var", xyt="s"),
            Variable("Race white", kind="binary", encoding="one_hot", xyt="s"),
            Variable("Hispanic", kind="binary", encoding="one_hot", xyt="s"),
            Variable("Any other white", kind="binary", encoding="one_hot", xyt="s"),
            Variable("Race asian", kind="binary", encoding="one_hot", xyt="s"),
            Variable("Race black", kind="binary", encoding="one_hot", xyt="s"),
            Variable(
                "Subsets of SSc according to LeRoy (1988)",
                kind="categorical",
                encoding="one_hot",
                xyt="s",
            ),
            Variable("Date of birth", kind="continuous", encoding="mean_var", xyt="s"),
            Variable(
                "Onset of first non-Raynaud?s of the disease",
                encoding="mean_var",
                xyt="s",
            ),
        ]

        self.variable_names = [var.name for var in self.variables]

        return self.variables

    def init_labels(self):
        self.labels = []
        return

    def create_labels(self, df):
        return
