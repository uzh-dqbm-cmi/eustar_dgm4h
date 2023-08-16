import numpy as np
import pandas as pd
from .organ import Organ
from pythae.ssc.variable import Variable


class LUNG_PH(Organ):
    def __init__(self):
        super().__init__("LUNG_PH")

    def init_variables(self):
        self.variables = [
            Variable(self.time_name, kind="continuous", encoding="mean_var", xyt="t"),
            Variable(
                "PAPsys (mmHg)",
                kind="continuous",
                encoding="mean_var",
                xyt="x",
            ),
            Variable(
                "TAPSE: tricuspid annular plane systolic excursion in cm",
                kind="continuous",
                encoding="mean_var",
                xyt="x",
            ),
            Variable(
                "Right ventricular area (cm2) (right ventricular dilation)",
                kind="continuous",
                encoding="mean_var",
                xyt="x",
            ),
            Variable(
                "Tricuspid regurgitation velocity (m/sec)",
                kind="continuous",
                encoding="mean_var",
                xyt="x",
            ),
            Variable(
                "Pulmonary wedge pressure (mmHg)",
                kind="continuous",
                encoding="mean_var",
                xyt="x",
            ),
            Variable(
                "Pulmonary resistance (dyn.s.cm-5)",
                kind="continuous",
                encoding="mean_var",
                xyt="x",
            ),
            Variable(
                "6 Minute walk test (distance in m)",
                kind="continuous",
                encoding="mean_var",
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
