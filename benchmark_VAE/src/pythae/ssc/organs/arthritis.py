import numpy as np
import pandas as pd
from .organ import Organ
from pythae.ssc.variable import Variable
from pythae.ssc.utils import (
    not_nan,
    OR_nan,
    less_nan,
    equal_nan,
    greater_nan,
    and_nan,
    less_equal_nan,
    greater_equal_nan,
    where_nan,
    less_,
    equal_,
    greater_,
    less_equal_,
    greater_equal_,
)


class ARTHRITIS(Organ):
    def __init__(self):
        super().__init__("ARTHRITIS")

    def init_variables(self):
        self.variables = [
            Variable(self.time_name, kind="continuous", encoding="mean_var", xyt="t"),
            Variable("Joint synovitis", kind="binary", encoding="one_hot", xyt="x"),
            Variable("Joint polyarthritis", kind="binary", encoding="one_hot", xyt="x"),
            Variable("Swollen joints", kind="continuous", encoding="mean_var", xyt="x"),
            Variable(
                "Tendon friction rubs", kind="binary", encoding="one_hot", xyt="x"
            ),
            Variable(
                "DAS 28 (ESR, calculated)",
                kind="continuous",
                encoding="mean_var",
                xyt="x",
            ),
            Variable(
                "DAS 28 (CRP, calculated)",
                kind="continuous",
                encoding="mean_var",
                xyt="x",
            ),
        ]

        self.variable_names = [var.name for var in self.variables]

        return self.variables

    def init_labels(self):
        time_since_str = "time_since_first_" + self.name + "_"

        self.labels = [
            Variable(
                self.name + "_" + "involvement_or",
                kind="binary",
                encoding="one_hot",
                xyt="y",
            ),
            Variable(
                self.name + "_" + "stage_or",
                kind="categorical",
                encoding="one_hot",
                xyt="y",
            ),
            Variable(
                time_since_str + "involvement_or",
                kind="continuous",
                encoding="mean_var",
                xyt="t",
            ),
        ]
        for i in [0, 1, 2, 3]:
            self.labels.append(
                Variable(
                    time_since_str + "stage_or_" + str(i),
                    kind="continuous",
                    encoding="mean_var",
                    xyt="t",
                )
            )
        self.labels_name = [var.name for var in self.labels]
        self.x_name = "Tendon friction rubs"
        self.y_name = "Joint synovitis"
        self.z_name = "DAS 28 (ESR, calculated)"

        return

    def create_labels(self, df):
        inv = self.involvement_or(df[self.x_name], df[self.y_name])
        stage = self.stage_or(df[self.z_name])
        df[self.name + "_" + "stage_or"] = stage

        df[self.name + "_" + "involvement_or"] = inv
        # df[self.labels[1].name] = stage

        nams_new = self.create_since_labels(df)
        return

    def create_since_labels(self, df):
        """
        create new columns indicating the time since the first non-zero and non-nan value appears
        """

        ## hard coded!!
        series = []
        str_nr = []

        # involvement
        # series.append( df[self.labels_name[0]] )
        series.append(df[self.name + "_" + "involvement_or"])

        # # stages
        for i in [0, 1, 2, 3]:
            serie = (df[self.labels_name[1]] == i) * 1.0
            serie.name += "_" + str(i)
            series.append(serie)

        # construct DF
        DF = pd.DataFrame(series).T

        # compute mask indicating which entries are non-zero and non-nan
        Mask = (DF != 0.0) & (~DF.isnull())

        # compute indices for each colum of first appearence (or nan if never)
        inds = Mask.apply(where_nan)

        time = df[self.time_name]

        # add columns for each variable

        nams = []
        for i, ind in enumerate(inds):
            nam = "time_since_first_" + DF.columns[i]
            nams.append(nam)

            if np.isnan(ind):
                df[nam] = np.nan
            else:
                df[nam] = time - time.iloc[int(ind)]

        return nams

    def involvement_or(self, x, y):
        involved = OR_nan([equal_(x, 1.0), equal_(y, 1.0)])
        return involved

    def stage_or(self, z):
        res = np.ones_like(z) * np.nan
        conds = {
            0.0: less_nan(z, 2.7),
            1.0: and_nan(greater_equal_nan(z, 2.7), less_equal_nan(z, 3.2)),
            2.0: and_nan(greater_nan(z, 3.2), less_equal_nan(z, 5.1)),
            3.0: greater_nan(z, 5.1),
        }
        for key, cond in conds.items():
            res[cond == 1.0] = key
        return res
