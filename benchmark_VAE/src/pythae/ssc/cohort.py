import numpy as np
import pandas as pd


import sys


from pythae.ssc.patient import Patient

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import os, pickle


from pythae.data.datasets import MissingDataset


class Cohort:
    # def __init__(self, vis, pats, meds):

    #    self.vis = vis
    #    self.pats = pats
    #    self.meds = meds

    def __init__(self, path):
        self.load_data(path)

    def load_data(self, path):
        """
        note that the original data was slightly preprocessed 
        """

        self.pats = pd.read_pickle(path + "pats")
        self.vis = pd.read_pickle(path + "vis")
        self.meds = pd.read_pickle(path + "meds")

    def preprocess(self, ns_visits_drop=[1]):
        """
        Preprocess the data globally on cohort level
        """

        # Replace particular strings in the data
        self._replace_strings()

        # Change format of some columns
        self._make_numeric()

        # remove some outliers
        self._remove_outliers()

        # remove all patients with 0 and ns_visits_drop visits
        self._drop_zero_one_visit_patients(ns_visits_drop=ns_visits_drop)

        # remove visits with no date
        self.vis = self.vis[~self.vis["Visit Date"].isna()]

    def _replace_strings(self):
        """
        Replaces string in vis and pats.
        """

        # remove 'unknown' etc by nan in vis
        nan_strings = ["unknown", "Unknown", "not done", "NaN", "nan", "Not known"]
        for ns in nan_strings:
            self.vis = self.vis.replace(ns, np.nan)

        # remove 'unknown' etc by nan in pats
        nan_strings = ["unknown", "Unknown", "not done", "NaN", "nan", "Not known"]
        for ns in nan_strings:
            self.pats = self.pats.replace(ns, np.nan)

        # replace words in vis
        replace_by = {
            "Yes": 1,
            "yes": 1,
            "No": 0,
            "no": 0,
            "I": 1,
            "II": 2,
            "III": 3,
            "IV": 4,
        }
        # for variable 'Lung fibrosis/ %involvement'
        # [nan, '>20%', '<20%', 'Indeterminate'] -> [nan, 1, -1, 0]
        replace_by2 = {">20%": 1, "<20%": -1, "Indeterminate": 0}
        replace_by3 = {
            "Limited cutaneous involvement": 2,
            "Diffuse cutaneous involvement": 3,
            "Only sclerodactyly": 1,
            "No skin involvement": 0,
        }
        replace_by4 = {"Current": 1, "Never": 0, "Previously": -1}
        replace_by = {**replace_by, **replace_by2, **replace_by3, **replace_by4}

        for rb in replace_by:
            self.vis = self.vis.replace(rb, replace_by[rb])

        # replace words in pats
        replace_by = {
            "Yes": 1,
            "yes": 1,
            "No": 0,
            "no": 0,
            "male": 0.0,
            "female": 1.0,
            "White": 1.0,
            "Hispanic": 1.0,
            "Any other White": 1.0,
            "Asian": 1.0,
            "Black": 1.0,
            "Limited cutaneous SSc": 1.0,
            "Diffuse cutaneous SSc": 2.0,
        }
        for rb in replace_by:
            self.pats = self.pats.replace(rb, replace_by[rb])

    def _make_numeric(self):
        """
        Change some variables to numeric.
        """

        variables = ["Dyspnea (NYHA-stage)"]

        for var in variables:
            self.vis[var] = pd.to_numeric(self.vis[var])

    def _remove_outliers(self):
        # variable name and range
        variables_vis = {
            "BNP (pg/ml)": (0, 10000),
            "6 Minute walk test (distance in m)": (0, 2500),
            "Pulmonary wedge pressure (mmHg)": (0, 200),
            "Pulmonary resistance (dyn.s.cm-5)": (0, 3000),
            "Body weight (kg)": (0, 200),
        }
        variables_pats = {
            "Onset of first non-Raynaud?s of the disease": (
                pd.Timestamp("1930-06-15 00:00:00"),
                pd.Timestamp("2023-06-15 00:00:00"),
            )
        }
        for var, values in variables_vis.items():
            self.vis[var].mask(
                (self.vis[var] > values[1]) | (self.vis[var] < values[0]), inplace=True
            )
        for var, values in variables_pats.items():
            self.pats[var].mask(
                (self.pats[var] > values[1]) | (self.pats[var] < values[0]),
                inplace=True,
            )

    def _drop_zero_one_visit_patients(self, ns_visits_drop=[1]):
        """
        Remove all patients with 0 and ns_visits_drop.
        """

        # compute number of visits for each patient
        tmp = self.vis.groupby("Id Patient V2018").apply(lambda x: len(x))

        # patients with 0 visit
        p_0 = self.pats["Id Patient V2018"][
            ~self.pats["Id Patient V2018"].isin(tmp.index)
        ]
        p_drop = p_0

        # print
        print("total number of patients " + str(self.pats.shape[0]))
        print("number of patients with 0 visit " + str(len(p_drop)))

        # loop over all numbers of visits which should be dropped
        for n_vis_drop in ns_visits_drop:
            p_i = tmp[tmp == n_vis_drop].index

            # print
            print(
                "number of patients with " + str(n_vis_drop) + " visit " + str(len(p_i))
            )

            # patients which should be dropped
            p_drop = np.hstack([p_drop, p_i])

        # drop patients with 0 or 1,... visits
        self.vis = self.vis[~self.vis["Id Patient V2018"].isin(p_drop)]
        self.pats = self.pats[~self.pats["Id Patient V2018"].isin(p_drop)]
        self.meds = self.meds[~self.meds["Id Patient 2018"].isin(p_drop)]

        # print
        print(
            "number of patients with 0 and "
            + str(ns_visits_drop)
            + " visits "
            + str(len(p_drop))
        )
        print("new number of patients " + str(self.pats.shape[0]))

        assert len(self.vis.groupby("Id Patient V2018")) == self.pats.shape[0]

    def create_patients(self, Nsubset=None):
        """
        Create array of patients by extracting the raw data from vis, pats, meds
        """

        # group by patient ID
        vis_grouped = self.vis.groupby("Id Patient V2018")
        vis_keys = list(vis_grouped.groups.keys())

        pats_grouped = self.pats.groupby("Id Patient V2018")
        pats_keys = list(pats_grouped.groups.keys())

        meds_grouped = self.meds.groupby("Id Patient 2018")
        meds_keys = list(meds_grouped.groups.keys())

        if Nsubset is None:
            N = len(vis_keys)
        else:
            N = Nsubset

        #### create array of Patient
        self.Patients = np.zeros(N, dtype=Patient)

        for i in tqdm(range(N)):
            key_i = vis_keys[i]

            # get vis of patient
            vis_df = vis_grouped.get_group(key_i).copy()

            # get pats of patient
            pats_df = pats_grouped.get_group(key_i).copy()

            # there is not always a medication for each patient
            if key_i in meds_keys:
                meds_df = meds_grouped.get_group(key_i).copy()
            else:
                meds_df = pd.DataFrame()

            # create Patient object
            self.Patients[i] = Patient(vis_df, pats_df, meds_df)

        return self.Patients

    def extract_data_frame(self, borgan):
        # for pat in self.Patients:
        for i in tqdm(range(len(self.Patients))):
            pat = self.Patients[i]

            df_x = pat.df_vis[
                borgan.variable_names_xyt["x"] + borgan.variable_names_xyt["t"]
            ]

            df_s = pat.df_pat[borgan.variable_names_xyt["s"]]

            # drop rows where all columns (except time) is nan
            df_x = df_x.dropna(
                how="all",
                subset=[
                    var
                    for var in borgan.variable_names_xyt["x"]
                    if var != "time [years]"
                ],
            )
            df_s = df_s.dropna(
                how="all",
                subset=[
                    var
                    for var in borgan.variable_names_xyt["s"]
                    if var != "time [years]"
                ],
            )

            # df_t = df_t.dropna(how="all", subset = [var for var in borgan.variable_names_xyt["t"] if var != "time [years]"])

            setattr(pat, "df_vis_" + borgan.name, df_x)
            setattr(pat, "df_pat_" + borgan.name, df_s)

            df_all = df_x.copy()
            if len(df_x) == 0:
                df_all[df_s.columns] = None
            else:
                df_all[df_s.columns] = pd.concat([df_s] * len(df_x)).values
                for col in [
                    "Sex",
                    "Height",
                    "Race white",
                    "Hispanic",
                    "Any other white",
                    "Race asian",
                    "Race black",
                    "Subsets of SSc according to LeRoy (1988)",
                ]:
                    df_all[col] = pd.to_numeric(df_all[col])
            setattr(pat, "df_all_" + borgan.name, df_all)

    def encode_data(self, borgan):
        # df_name = "df_vis_" + borgan.name
        df_name = "df_all_" + borgan.name

        # fit the encoder for all training data
        DF_vis_train = pd.concat([getattr(pat, df_name) for pat in self.Patients_train])

        borgan.fit_encode_variables(DF_vis_train)

        # encode the data for each patient
        # for pat in self.Patients:
        for i in tqdm(range(len(self.Patients))):
            pat = self.Patients[i]

            # encode the df into array
            df = getattr(pat, df_name)
            if df.shape[0] > 0:
                array, missing, array_filled = borgan.encode_variables(df)
            else:
                ncols = len(borgan.encoding_names)
                array, missing, array_filled = (
                    np.zeros((0, ncols)),
                    np.zeros((0, ncols)),
                    np.zeros((0, ncols)),
                )
                print("patient df for ", borgan.name, "is empty")

            setattr(pat, "array_vis_" + borgan.name, array)
            setattr(pat, "missing_vis_" + borgan.name, missing)
            setattr(pat, "array_filled_vis_" + borgan.name, array_filled)
            pat.encoding_names = borgan.encoding_names

    def decode_data_for_organ(self, organ, attr_name="array_vis_"):
        # does it make sense? include argmax?

        decoded_list = []
        # for pat in self.Patients:
        for i in tqdm(range(len(self.Patients))):
            pat = self.Patients[i]

            dec = organ.decode_variables(getattr(pat, attr_name + organ.name))
            setattr(pat, attr_name + "decoded_" + organ.name, dec)

    def create_labels_for_organ(
        self, organ, use_body_df=False, attr_names=["df_vis_", "df_all_"]
    ):
        """
        Create additional labels for patients.
        """
        df_names = [
            attr_name + "body" if use_body_df else attr_name + organ.name
            for attr_name in attr_names
        ]
        for df_name in df_names:
            for i in tqdm(range(len(self.Patients))):
                pat = self.Patients[i]

                organ.create_labels(getattr(pat, df_name))

    def split_train_test(self, frac_train=0.75, seed=0):
        # set seed
        np.random.seed(seed)

        # consturct random permutation
        perm = np.random.permutation(len(self.Patients))

        # take fraction for training
        Ntrain = int(len(self.Patients) * frac_train)

        # select indeces for training and testing
        ind_train = perm[:Ntrain]
        ind_test = perm[Ntrain:]

        self.Patients_train = self.Patients[ind_train]
        self.Patients_test = self.Patients[ind_test]

        return self.Patients_train, self.Patients_test

    def data_train_test(self, borgan, path="", name="", PICKLE=True):
        indeces_list = [
            [i for i, item in enumerate(borgan.encoding_xyt1) if item == xyt]
            for xyt in ["x", "y", "t", "s"]
        ]

        dats = []
        for Patients in [self.Patients_train, self.Patients_test]:
            list_dat = [
                [
                    getattr(pat, "array_filled_vis_" + borgan.name)[:, il]
                    for pat in Patients
                ]
                for il in indeces_list
            ]
            list_miss = [
                [getattr(pat, "missing_vis_" + borgan.name)[:, il] for pat in Patients]
                for il in indeces_list
            ]

            dats.append(MissingDataset(*list_dat, *list_miss))

        if PICKLE:
            with open(path + "data_train" + name + ".pkl", "wb") as file:
                pickle.dump(dats[0], file)

            with open(path + "data_test" + name + ".pkl", "wb") as file:
                pickle.dump(dats[1], file)

            with open(path + "names_splits" + name + ".pkl", "wb") as file:
                pickle.dump(
                    (
                        borgan.encoding_names,
                        borgan.splits,
                        borgan.encoding_xyt0,
                        borgan.encoding_xyt1,
                    ),
                    file,
                )

        return (
            dats[0],
            dats[1],
            borgan.encoding_names,
            borgan.splits,
            borgan.encoding_xyt0,
            borgan.encoding_xyt1,
        )
