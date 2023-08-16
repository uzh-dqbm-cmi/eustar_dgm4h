import numpy as np
import pandas as pd
from .organ import Organ
from pythae.ssc.variable import Variable


class SKIN(Organ):
    def __init__(self):
        super().__init__("SKIN")

    def init_variables(self):
        self.variables = [
            Variable(self.time_name, kind="continuous", encoding="mean_var", xyt="t"),
            Variable(
                "Worsening of skin within the last month",
                kind="binary",
                encoding="one_hot",
                xyt="x",
            ),
            Variable(
                "Extent of skin involvement",
                kind="categorical",
                encoding="one_hot",
                xyt="x",
            ),
            Variable(
                "Modified Rodnan Skin Score, only imported value",
                kind="continuous",
                encoding="mean_var",
                xyt="x",
            ),
            Variable(
                "Skin thickening of the fingers of both hands extending proximal to the MCP joints",
                kind="categorical",
                encoding="one_hot",
                xyt="x",
            ),
            Variable(
                "Skin thickening of the whole finger distal to MCP (Sclerodactyly)",
                kind="categorical",
                encoding="one_hot",
                xyt="x",
            ),
            Variable(
                "Skin thickening sparing the fingers",
                kind="categorical",
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
