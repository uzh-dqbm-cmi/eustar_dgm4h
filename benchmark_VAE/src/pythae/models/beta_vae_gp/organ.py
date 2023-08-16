import numpy as np
import pandas as pd
import torch

# from matplotlib import pyplot as plt
# from tqdm.notebook import tqdm

# from pythae.models.beta_vae_gp.variable import Variable
# from pythae.models.beta_vae_gp.utils import not_nan, OR_nan, less_nan, equal_nan, greater_nan, and_nan, less_equal_nan, greater_equal_nan, where_nan


class Organ:
    def __init__(self, name):
        self.name = name

        self.time_name = "time [years]"

        self.init_variables()
        self.init_labels()

    def fit_encode_variables(self, DF_vis_train):
        """
        fit the encoding to all variables in the training data
        """

        self.encoding_names = []
        encoding_splits = []
        self.encoding_xyt0 = []  # list of xyt of original variables
        self.encoding_xyt1 = []  # list of xyt of transfromed variables

        for var in self.variables + self.labels:
            # fit each variable
            var.fit_encode(DF_vis_train)

            # collect names and lengths
            self.encoding_names += list(var.encoding_names)
            encoding_splits.append(len(var.encoding_names))

            self.encoding_xyt0 += [var.xyt]
            self.encoding_xyt1 += [var.xyt] * encoding_splits[-1]

        # compute splits for the variables
        self.splits = encoding_splits

        return self.encoding_names, self.splits

    def encode_variables(self, DF):
        """
        encode all variables together
        """

        # encode each variable including the nans
        ENC = np.hstack([var.encode(DF) for var in self.variables + self.labels])

        # store the nans
        MISS = np.isnan(ENC)

        # fill and encode
        ENC_FILLED = np.hstack(
            [var.encode(DF, fill_nan=True) for var in self.variables + self.labels]
        )

        return ENC, MISS, ENC_FILLED

    def decode_variables(self, Pat_array):
        # split the output into each variable
        list_of_arrays = np.split(Pat_array, np.cumsum(self.splits[:-1]), axis=1)

        # decode each variable
        DEC = np.hstack(
            [self.variables[i].decode(array) for i, array in enumerate(list_of_arrays)]
        )

        print(DEC.shape)

        return DEC

    def decode_(self, out_array, splits_x0, var_names):
        # split the output into each variable
        list_of_arrays = np.split(out_array, np.cumsum(splits_x0[:-1]), axis=1)

        res_list = []
        for i, arr in enumerate(list_of_arrays):
            Var = [
                var
                for var in (self.variables + self.labels)
                if var.name == var_names[i]
            ][0]
            res_list.append(Var.decode_(arr))

        res_matrix = torch.cat([r_[0] for r_ in res_list], 1)
        prob_matrix = torch.cat([r_[1] for r_ in res_list], 1)

        return res_matrix, prob_matrix, res_list


# class LUNG_ILD(Organ):

#     def __init__(self):

#         super().__init__('LUNG_ILD')


#     def init_variables(self):

#         self.variables = [
#             Variable( self.time_name, kind='continuous', encoding='none', xyt='t'),
#             Variable( 'Forced Vital Capacity (FVC - % predicted)', kind='continuous', encoding='mean_var', xyt='x'),
#             Variable( 'DLCO/SB (% predicted)', kind='continuous', encoding='mean_var', xyt='x'),
#             Variable( 'DLCOc/VA (% predicted)', kind='continuous', encoding='mean_var', xyt='x'),
#             Variable( 'Lung fibrosis/ %involvement', kind='categorical', encoding='one_hot', xyt='x'),
#             # [nan, '>20%', '<20%', 'Indeterminate'] -> [nan, 1, -1, 0]
#             #Variable( 'Dyspnea (NYHA-stage)', kind='ordinal', encoding='ordinal', xyt='x'),
#             Variable( 'Dyspnea (NYHA-stage)', kind='categorical', encoding='one_hot', xyt='x'),
#             Variable( 'Worsening of cardiopulmonary manifestations within the last month', kind='binary', encoding='one_hot', xyt='x'),
#             Variable( 'HRCT: Lung fibrosis', kind='binary', encoding='one_hot', xyt='x'),

#             Variable( 'Ground glass opacification', kind='binary', encoding='one_hot', xyt='x'),
#             Variable( 'Honey combing', kind='binary', encoding='one_hot', xyt='x'),
#             Variable( 'Tractions', kind='binary', encoding='one_hot', xyt='x'),
#             Variable( 'Any reticular changes', kind='binary', encoding='one_hot', xyt='x')
#         ]

#         self.variable_names = [var.name for var in self.variables]

#         return self.variables


#     def init_labels(self):
#         """
#         init label quantities
#         """

#         time_since_str = 'time_since_first_'+self.name+'_'

#         # make sure that we actually create the labels in "create_labels"
#         self.labels = [
#                 Variable( self.name+'_'+'involvement_or', kind='binary', encoding='one_hot', xyt='y'),
#                 Variable( self.name+'_'+'stage_or', kind='categorical', encoding='one_hot', xyt='y'),
#                 Variable( time_since_str+'involvement_or', kind='continuous', encoding='mean_var', xyt='t')
#                 #Variable( time_since_str+'involvement_or', kind='continuous', encoding='mean_var'),
#             ]

#         ## hard coded!!!!!
#         for i in [1,2,3,4]:
#             self.labels.append( Variable( time_since_str+'stage_or_'+str(i), kind='continuous', encoding='mean_var', xyt='t') )

#         self.labels_name = [var.name for var in self.labels]

#         self.x_name = 'Forced Vital Capacity (FVC - % predicted)'
#         self.y_name = 'HRCT: Lung fibrosis'
#         self.z_name = 'Dyspnea (NYHA-stage)'
#         self.q_name = 'Lung fibrosis/ %involvement' # [nan, '>20%', '<20%', 'Indeterminate'] -> [nan, 1, -1, 0]


#         # include also mortality
#         # include also time since!!

#         ##################
#         ## 1) organ involvement, 0: not involved, 1: involved, nan
#         ## 2) severity stages, 1: mild, 2: moderate, 3: severe, 4: endorgan, nan
#         ## 3) end organ, 0: not, 1: end organ, nan (same as 4 in severity stage!)


#     def create_labels(self, df):
#         """
#         create labels based on the data in df
#         df is updated with the new labels
#         """


#         inv = self.involvement_or(df[self.x_name], df[self.y_name])
#         stage = self.stage_or(df[self.x_name], df[self.z_name], df[self.q_name])


#         df[self.labels[0].name] = inv
#         df[self.labels[1].name] = stage


#         nams_new = self.create_since_labels(df)


#     def involvement_or(self, x, y):

#         involved = OR_nan( [less_nan(x,70), equal_nan(y,1.)] )

#         return involved


#     def stage_or(self, x, z, q):

#         res = np.ones_like(x)*np.nan

#         conds = {1.: OR_nan( [ greater_nan(x,80) , equal_nan(z,2.) ]),
#             2.: OR_nan( [ and_nan(less_equal_nan(x,80) , greater_equal_nan(x,70)) , equal_nan(z,3.) , greater_equal_nan(q,-1) ]),
#             3.: OR_nan( [ and_nan(less_equal_nan(x,70) , greater_equal_nan(x,50)) ,  equal_nan(z,4.) , greater_equal_nan(q,1) ]),
#             4.: OR_nan( [ less_equal_nan(x,50) , equal_nan(z,4.) ])
#                 }

#         for key, cond in conds.items():
#             res[cond==1.] = key

#         return res


#     def create_since_labels(self, df):
#         """
#         create new columns indicating the time since the first non-zero and non-nan value appears
#         """


#         ## hard coded!!
#         series = []
#         str_nr = []

#         # involvement
#         series.append( df[self.labels_name[0]] )


#         # stages
#         for i in [1,2,3,4]:
#             serie = (df[self.labels_name[1]]==i)*1.
#             serie.name += '_'+str(i)
#             series.append( serie )


#         # construct DF
#         DF = pd.DataFrame(series).T


#         # compute mask indicating which entries are non-zero and non-nan
#         Mask = (DF!=0.) & (~DF.isnull())

#         # compute indices for each colum of first appearence (or nan if never)
#         inds = Mask.apply(where_nan)

#         # time vetor
#         time = df[ self.time_name ]

#         # add columns for each variable

#         nams = []
#         for i,ind in enumerate(inds):
#             nam = 'time_since_first_'+DF.columns[i]
#             nams.append(nam)

#             if np.isnan(ind):
#                 df[nam] = np.nan
#             else:
#                 df[nam] = time - time.iloc[int(ind)]

#         return nams


# 'First diagnosis of ILD by X-ray or HRCT'
# 'HRCT: Lung fibrosis V2018',
# 'Dyspnea (significant)',
# 'Dyspnea stage',
# 'DLCO/VA (% predicted)'
# 'Plain X-ray: Lung fibrosis'
# 'Hospitalisation due to lung disease',
# 'Planned visit for therapy of lung disease',
# 'Unplanned visit due to lung disease',
# 'Worst Borg dyspnea score during the test',
# 'Unexplained dyspnea'
#  ],
#  'LUNG_PH':
#  ['PAPsys (mmHg)',
#   'Forced Vital Capacity (FVC - % predicted)',
#   'DLCO/SB (% predicted)',
#   'NTproBNP (pg/ml)',
#   'TAPSE: tricuspid annular plane systolic excursion in cm',
#   'Right atrium area (cm2)',
#   'Right ventricular area (cm2) (right ventricular dilation)',
#   'Tricuspid regurgitation velocity (m/sec)',
#   'PAP mean (mmHg)',
#   'Pulmonary wedge pressure (mmHg)',
#   'Pulmonary resistance (dyn.s.cm-5)',
#   '6 Minute walk test (distance in m)'
#
# 'Pulmonary hypertension',
#'Worsening of cardiopulmonary manifestations within the last month',
# 'Dyspnea (NYHA-stage)',
# 'DLCOc/VA (% predicted)',
# 'DLCO/VA (% predicted)'
# 'Dyspnea (significant)',
# 'Dyspnea stage'
# 'Previously diagnosed pulmonary hypertension by rhc',
# 'Lung fibrosis/ %involvement',
# 'Hospitalisation due to lung disease',
# 'Planned visit for therapy of lung disease',
# 'Unplanned visit due to lung disease',
# 'Plain X-ray: Lung fibrosis',
# 'Worst Borg dyspnea score during the test',
# 'Unexplained dyspnea'
#      ]
#    }
