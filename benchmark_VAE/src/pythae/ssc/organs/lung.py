import numpy as np
import pandas as pd
import intervals as I
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


class LUNG_ILD(Organ):
    def __init__(self):
        super().__init__("LUNG_ILD")

    def init_variables(self):
        self.variables = [
            Variable(self.time_name, kind="continuous", encoding="mean_var", xyt="t"),
            # Variable(self.time_name, kind="continuous", encoding="mean_var", xyt="x"),
            Variable(
                "Forced Vital Capacity (FVC - % predicted)",
                kind="continuous",
                encoding="mean_var",
                xyt="x",
            ),
            Variable(
                "DLCO/SB (% predicted)", kind="continuous", encoding="mean_var", xyt="x"
            ),
            Variable(
                "DLCOc/VA (% predicted)",
                kind="continuous",
                encoding="mean_var",
                xyt="x",
            ),
            Variable(
                "Lung fibrosis/ %involvement",
                kind="categorical",
                encoding="one_hot",
                xyt="x",
            ),
            # [nan, '>20%', '<20%', 'Indeterminate'] -> [nan, 1, -1, 0]
            # Variable( 'Dyspnea (NYHA-stage)', kind='ordinal', encoding='ordinal', xyt='x'),
            Variable(
                "Dyspnea (NYHA-stage)", kind="categorical", encoding="one_hot", xyt="x"
            ),
            Variable(
                "Worsening of cardiopulmonary manifestations within the last month",
                kind="binary",
                encoding="one_hot",
                xyt="x",
            ),
            Variable("HRCT: Lung fibrosis", kind="binary", encoding="one_hot", xyt="x"),
            Variable(
                "Ground glass opacification", kind="binary", encoding="one_hot", xyt="x"
            ),
            Variable("Honey combing", kind="binary", encoding="one_hot", xyt="x"),
            Variable("Tractions", kind="binary", encoding="one_hot", xyt="x"),
            Variable(
                "Any reticular changes", kind="binary", encoding="one_hot", xyt="x"
            ),
        ]

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
            # Variable(
            #     self.name + "_FVC_progression",
            #     kind="categorical",
            #     encoding="one_hot",
            #     xyt="y",
            # ),
            # Variable( time_since_str+'involvement_or', kind='continuous', encoding='mean_var'),
        ]

        ## hard coded!!!!!
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

        self.x_name = "Forced Vital Capacity (FVC - % predicted)"
        self.y_name = "HRCT: Lung fibrosis"
        self.z_name = "Dyspnea (NYHA-stage)"
        self.q_name = "Lung fibrosis/ %involvement"  # [nan, '>20%', '<20%', 'Indeterminate'] -> [nan, 1, -1, 0]
        # impr, stable, prog in delta_t = t_(i+1) - t_i
        # self.prog_dict = {
        #     "Forced Vital Capacity (FVC - % predicted)": [
        #         I.closedopen(5, I.inf),
        #         I.open(-5, 5),
        #         I.openclosed(-I.inf, -5),
        #     ]
        # }

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

        inv = self.involvement_or(df[self.x_name], df[self.y_name])
        stage = self.stage_or(df[self.x_name], df[self.z_name], df[self.q_name])

        # fvc_prog = self.progression(df[self.x_name], self.prog_dict[self.x_name])

        df[self.name + "_" + "involvement_or"] = inv
        df[self.name + "_" + "stage_or"] = stage
        # df[self.name + "_FVC_progression"] = fvc_prog

        nams_new = self.create_since_labels(df)

    def progression(self, x, cond):
        prog = x[1:].values - x[:-1].values
        res = np.full(len(x), np.nan)
        for i, delta in enumerate(prog):
            match = [index for index, inter in enumerate(cond) if delta in inter]
            if len(match) > 1:
                raise ValueError("Progression has mulitple possibilities")
            if len(match) > 0:
                res[i + 1] = match[0]

        return res

    def involvement_or(self, x, y):
        # involved = OR_nan([less_nan(x, 70), equal_nan(y, 1.0)])
        involved = OR_nan([less_(x, 70), equal_(y, 1.0)])
        return involved

    def stage_or(self, x, z, q):
        res = np.ones_like(x) * np.nan

        # conds = {
        #     1.0: OR_nan([greater_nan(x, 80), equal_nan(z, 2.0)]),
        #     2.0: OR_nan(
        #         [
        #             and_nan(less_equal_nan(x, 80), greater_equal_nan(x, 70)),
        #             equal_nan(z, 3.0),
        #             greater_equal_nan(q, -1),
        #         ]
        #     ),
        #     3.0: OR_nan(
        #         [
        #             and_nan(less_equal_nan(x, 70), greater_equal_nan(x, 50)),
        #             equal_nan(z, 4.0),
        #             greater_equal_nan(q, 1),
        #         ]
        #     ),
        #     4.0: OR_nan([less_equal_nan(x, 50), equal_nan(z, 4.0)]),
        # }
        conds = {
            1.0: OR_nan([greater_(x, 80), equal_(z, 2.0)]),
            2.0: OR_nan(
                [
                    and_nan(less_equal_(x, 80), greater_equal_(x, 70)),
                    equal_(z, 3.0),
                    greater_equal_(q, -1),
                ]
            ),
            3.0: OR_nan(
                [
                    and_nan(less_equal_(x, 70), greater_equal_(x, 50)),
                    equal_(z, 4.0),
                    greater_equal_(q, 1),
                ]
            ),
            4.0: OR_nan([less_equal_(x, 50), equal_(z, 4.0)]),
        }
        # all zeros --> same as nan
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
        series.append(df[self.labels_name[0]])

        # stages
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

        # time vector
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
