import sys
import pickle

sys.path.append("/home/cctrotte/krauthammer/eustar_clean/benchmark_VAE/src/")
sys.path.append("/cluster/work/medinfmk/EUSTAR2/code_ms/benchmark_VAE/src/")
sys.path.append("/cluster/work/medinfmk/EUSTAR2/code_ct/benchmark_VAE/src/")
import warnings

warnings.filterwarnings("ignore")
from pythae.ssc.organs.lung import LUNG_ILD
from pythae.ssc.organs.heart import HEART
from pythae.ssc.organs.lung_ph import LUNG_PH
from pythae.ssc.organs.kidney import KIDNEY
from pythae.ssc.organs.skin import SKIN
from pythae.ssc.organs.arthritis import ARTHRITIS
from pythae.ssc.organs.muscle import MUSCLE
from pythae.ssc.organs.gastro import GASTRO
from pythae.ssc.organs.general import GENERAL
from pythae.ssc.organs.static import STATIC

from pythae.ssc.cohort import Cohort
import numpy as np
import torch


class Body:
    def __init__(self, organs):
        self._organs = {organ.name: organ for organ in organs}
        self.init_variables()
        self.init_labels()
        self.name = "body"

    @property
    def organs(self):
        return self._organs.values()

    def get_organ(self, name):
        return self._organs.get(name)

    def init_variables(self):
        variable_dict = {}
        # to avoid having same variable twice
        for organ in self.organs:
            variable_dict.update({var.name: var for var in organ.variables})
        self.variable_names = list(variable_dict.keys())
        self.variables = list(variable_dict.values())
        self.variable_names_xyt = {
            "x": [
                name
                for name, val in zip(self.variable_names, self.variables)
                if val.xyt == "x"
            ],
            "y": [
                name
                for name, val in zip(self.variable_names, self.variables)
                if val.xyt == "y"
            ],
            "t": [
                name
                for name, val in zip(self.variable_names, self.variables)
                if val.xyt == "t"
            ],
            "s": [
                name
                for name, val in zip(self.variable_names, self.variables)
                if val.xyt == "s"
            ],
        }
        self.variables_xyt = {
            "x": [val for val in self.variables if val.xyt == "x"],
            "y": [val for val in self.variables if val.xyt == "y"],
            "t": [val for val in self.variables if val.xyt == "t"],
            "s": [val for val in self.variables if val.xyt == "s"],
        }
        return

    def init_labels(self):
        self.labels = []
        for organ in self.organs:
            self.labels += organ.labels
        self.labels_name = [var.name for var in self.labels]

    def get_var_by_name(self, var_name):
        variables = [
            var for var in self.variables + self.labels if var.name == var_name
        ]
        return variables[0]

    def encode_variables(self, DF):
        """
        encode all variables together
        """

        # encode each variable including the nans
        ENC = np.hstack([var.encode(DF) for var in self.variables + self.labels])

        # store the nans
        MISS = np.isnan(ENC)
        if MISS[:, 0].any():
            print("time is missing")

        # fill and encode
        ENC_FILLED = np.hstack(
            [var.encode(DF, fill_nan=True) for var in self.variables + self.labels]
        )

        return ENC, MISS, ENC_FILLED

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

    def decode_preds(self, out_array, splits, var_names):
        # split the output into each variable
        list_of_arrays = np.split(out_array, np.cumsum(splits[:-1]), axis=1)

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

    def decode(self, out_array, splits, var_names):
        # split the output into each variable
        list_of_arrays = np.split(out_array, np.cumsum(splits[:-1]), axis=1)

        res_list = []
        for i, arr in enumerate(list_of_arrays):
            Var = [
                var
                for var in (self.variables + self.labels)
                if var.name == var_names[i]
            ][0]
            res_list.append(Var.decode(arr))

        res_matrix = torch.tensor(
            np.concatenate([r_ for r_ in res_list], 1).astype(np.float32)
        )
        # prob_matrix = torch.cat([r_[1] for r_ in res_list], 1)

        return res_matrix, res_list


if __name__ == "__main__":
    local = True
    name = "_medium"
    # name = "_allcont"
    if local:
        data_path = "/home/cctrotte/krauthammer/eustar/fake_data/raw/"
        save_path = "/home/cctrotte/krauthammer/eustar/fake_data/processed/"
    else:
        data_path = "/cluster/work/medinfmk/EUSTAR2/data/raw/"
        save_path = "/cluster/work/medinfmk/EUSTAR2/data/processed/ct/"
    # full dataset
    # organs = [
    #     LUNG_ILD(),
    #     HEART(),
    #     LUNG_PH(),
    #     GASTRO(),
    #     GENERAL(),
    #     KIDNEY(),
    #     SKIN(),
    #     ARTHRITIS(),
    #     MUSCLE(),
    #     STATIC(),
    # ]
    # medium dataset
    organs = [
        LUNG_ILD(),
        HEART(),
        LUNG_PH(),
        STATIC(),
        ARTHRITIS(),
    ]
    # reduced dataset
    # organs = [LUNG_ILD(), STATIC()]
    body = Body(organs)
    cohort = Cohort(data_path)
    cohort.preprocess(ns_visits_drop=[1, 2, 3, 4] + [i for i in range(15, 35)])
    cut_time = padd_time = 15
    Patients = cohort.create_patients()
    Patients_train, Patients_test = cohort.split_train_test()
    cohort.extract_data_frame(body)
    for organ in body.organs:
        # cohort.extract_data_frame(organ)
        cohort.create_labels_for_organ(organ, use_body_df=True)
        # cohort.encode_data_for_organ(organ)
    cohort.encode_data(body)
    cohort.data_train_test(body, path=save_path, name=name, PICKLE=True)

    with open(save_path + "body_" + name + ".pkl", "wb") as file:
        pickle.dump(body, file)
    # save also the cohort
    with open(save_path + "cohort_" + name + ".pkl", "wb") as file:
        pickle.dump(cohort, file)

    print("end")
