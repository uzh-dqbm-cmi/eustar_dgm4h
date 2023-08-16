import numpy as np
import pandas as pd

from .organ import Organ
from pythae.ssc.variable import Variable
from pythae.ssc.utils import (
    logical_nan_not_satisfied,
    OR_nan,
    where_nan,
    equal_,
)


class HEART(Organ):
    def __init__(self):
        super().__init__("HEART")

    def init_variables(self):
        self.variables = [
            Variable(self.time_name, kind="continuous", encoding="mean_var", xyt="t"),
            Variable(
                "Left ventricular ejection fraction (%)",
                kind="continuous",
                encoding="mean_var",
                xyt="x",
            ),
            Variable(
                "Worsening of cardiopulmonary manifestations within the last month",
                kind="binary",
                encoding="one_hot",
                xyt="x",
            ),
            Variable(
                "Diastolic function abnormal",
                kind="binary",
                encoding="one_hot",
                xyt="x",
            ),
            Variable(
                "Ventricular arrhythmias", kind="binary", encoding="one_hot", xyt="x"
            ),
            Variable(
                "Arrhythmias requiring therapy",
                kind="binary",
                encoding="one_hot",
                xyt="x",
            ),
            Variable(
                "Pericardial effusion on echo",
                kind="binary",
                encoding="one_hot",
                xyt="x",
            ),
            Variable("Conduction blocks", kind="binary", encoding="one_hot", xyt="x"),
            Variable(
                "NTproBNP (pg/ml)", kind="continuous", encoding="mean_var", xyt="x"
            ),
            Variable(
                "Auricular Arrhythmias", kind="binary", encoding="one_hot", xyt="x"
            ),
            Variable("BNP (pg/ml)", kind="continuous", encoding="mean_var", xyt="x"),
            Variable("Cardiac arrhythmias", kind="binary", encoding="one_hot", xyt="x"),
            Variable(
                "Dyspnea (NYHA-stage)", kind="categorical", encoding="one_hot", xyt="x"
            ),
        ]
        #
        #
        self.variable_names = [var.name for var in self.variables]

        return self.variables

    def init_labels(self):
        """
        init label quantities
        """

        time_since_str = "time_since_first_" + self.name + "_"

        # make sure that we actually create the labels in "create_labels"
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
            # Variable( time_since_str+'involvement_or', kind='continuous', encoding='mean_var'),
        ]

        # ## hard coded!!!!!
        # for i in [1,2,3,4]:
        #     self.labels.append( Variable( time_since_str+'stage_or_'+str(i), kind='continuous', encoding='mean_var', xyt='t') )
        for i in [1, 2, 3, 4]:
            self.labels.append(
                Variable(
                    time_since_str + "stage_or_" + str(i),
                    kind="continuous",
                    encoding="mean_var",
                    xyt="t",
                )
            )
        self.labels_name = [var.name for var in self.labels]

        # varname: {threshold:, logical: }
        self.labels_dict = {
            "Left ventricular ejection fraction (%)": {"threshold": 45, "logical": "<"},
            "Worsening of cardiopulmonary manifestations within the last month": {
                "threshold": 1,
                "logical": "=",
            },
            "Diastolic function abnormal": {"threshold": 1, "logical": "="},
            "Ventricular arrhythmias": {"threshold": 1, "logical": "="},
            "Arrhythmias requiring therapy": {"threshold": 1, "logical": "="},
            "Pericardial effusion on echo": {"threshold": 1, "logical": "="},
            "Conduction blocks": {"threshold": 1, "logical": "="},
            "NTproBNP (pg/ml)": {"threshold": 125, "logical": ">"},
            "BNP (pg/ml)": {"threshold": 35, "logical": ">"},
        }

        # 'Cardiac arrhythmias': {'threshold': 1, 'logical': '='}

        #     self.x_names = ['Left ventricular ejection fraction (%)', 'Worsening of cardiopulmonary manifestations within the last month', 'Diastolic function abnormal', 'Cardiac arrhythmias', 'Ventricular arrhythmias', 'Arrhythmias requiring therapy',
        #    'Pericardial effusion on echo']

        self.x_name = "Dyspnea (NYHA-stage)"

    # self.y_name = 'HRCT: Lung fibrosis'
    # self.z_name = 'Dyspnea (NYHA-stage)'
    # self.q_name = 'Lung fibrosis/ %involvement' # [nan, '>20%', '<20%', 'Indeterminate'] -> [nan, 1, -1, 0]

    # include also mortality
    # include also time since!!

    ##################
    ## 1) organ involvement, 0: not involved, 1: involved, nan
    ## 2) severity stages, 1: mild, 2: moderate, 3: severe, 4: endorgan, nan
    ## 3) end organ, 0: not, 1: end organ, nan (same as 4 in severity stage!)

    def create_labels(self, df):
        """
        create labels based on the data in df
        df is updated with the new labels
        """

        # inv = self.involvement_or(df[self.x_name], df[self.y_name])
        inv = self.involvement_or(df)
        stage = self.stage_or(df[self.x_name])

        df[self.name + "_" + "stage_or"] = stage

        df[self.name + "_" + "involvement_or"] = inv
        # df[self.labels[1].name] = stage

        nams_new = self.create_since_labels(df)

    def involvement_or(self, df):
        # involved = OR_nan( [less_nan(x,70), equal_nan(y,1.)] )

        involved = OR_nan(
            [
                logical_nan_not_satisfied(df[key], value["threshold"], value["logical"])
                for key, value in self.labels_dict.items()
            ]
        )

        return involved

    def stage_or(self, x):

        res = np.ones_like(x) * np.nan

        conds = {
            1.0: equal_(x, 1.0),
            2.0: equal_(x, 2.0),
            3.0: equal_(x, 3.0),
            4.0: equal_(x, 4.0),
        }

        for key, cond in conds.items():
            res[cond == 1.0] = key

        return res

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
        for i in [1, 2, 3, 4]:
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
