import numpy as np
import os, pickle
import sys
import copy

from pythae.data.datasets import MissingDataset

# from benchmark_VAE.src.pythae.models.beta_vae_gp.classifier_config import ClassifierConfig
from pythae.models.beta_vae_gp.classifier_config import ClassifierConfig


def compute_folds(cohort, Patients, n_folds=5, frac_train=0.85, seed=0):
    cohorts = [copy.deepcopy(cohort) for i in range(n_folds)]
    np.random.seed(seed)
    perm = np.random.permutation(len(Patients))
    # take fraction for training
    Ntrain = int(len(Patients) * frac_train)
    # select indeces for training and testing
    ind_train = perm[:Ntrain]
    ind_test = perm[Ntrain:]

    # compute folds
    fold_size = int(Ntrain / n_folds)
    fold_ranges = [(i * fold_size, (i + 1) * fold_size) for i in range(n_folds)]

    for i, (start, end) in enumerate(fold_ranges):
        valid_indices = ind_train[start:end]
        train_indices = np.concatenate([ind_train[:start], ind_train[end:]])
        cohorts[i].Patients_train = cohorts[i].Patients[train_indices]
        cohorts[i].Patients_valid = cohorts[i].Patients[valid_indices]
        cohorts[i].Patients_test = cohorts[i].Patients[ind_test]

    return cohorts


def remove_short_samples(data_train, data_valid, data_test):
    data_train.list_x = [elem for elem in data_train.list_x if len(elem) > 1]
    data_train.list_y = [elem for elem in data_train.list_y if len(elem) > 1]
    data_train.list_t = [elem for elem in data_train.list_t if len(elem) > 1]
    data_train.list_s = [elem for elem in data_train.list_s if len(elem) > 1]

    data_train.missing_x = [elem for elem in data_train.missing_x if len(elem) > 1]
    data_train.missing_y = [elem for elem in data_train.missing_y if len(elem) > 1]
    data_train.missing_t = [elem for elem in data_train.missing_t if len(elem) > 1]
    data_train.missing_s = [elem for elem in data_train.missing_s if len(elem) > 1]

    data_valid.list_x = [elem for elem in data_valid.list_x if len(elem) > 1]
    data_valid.list_y = [elem for elem in data_valid.list_y if len(elem) > 1]
    data_valid.list_t = [elem for elem in data_valid.list_t if len(elem) > 1]
    data_valid.list_s = [elem for elem in data_valid.list_s if len(elem) > 1]
    data_valid.missing_x = [elem for elem in data_valid.missing_x if len(elem) > 1]
    data_valid.missing_y = [elem for elem in data_valid.missing_y if len(elem) > 1]
    data_valid.missing_t = [elem for elem in data_valid.missing_t if len(elem) > 1]
    data_valid.missing_s = [elem for elem in data_valid.missing_s if len(elem) > 1]

    data_test.list_x = [elem for elem in data_test.list_x if len(elem) > 1]
    data_test.list_y = [elem for elem in data_test.list_y if len(elem) > 1]
    data_test.list_t = [elem for elem in data_test.list_t if len(elem) > 1]
    data_test.list_s = [elem for elem in data_test.list_s if len(elem) > 1]
    data_test.missing_x = [elem for elem in data_test.missing_x if len(elem) > 1]
    data_test.missing_y = [elem for elem in data_test.missing_y if len(elem) > 1]
    data_test.missing_t = [elem for elem in data_test.missing_t if len(elem) > 1]
    data_test.missing_s = [elem for elem in data_test.missing_s if len(elem) > 1]
    return data_train, data_valid, data_test


def save_cv(cohorts, borgan, path="", name="", PICKLE=True):
    indeces_list = [
        [i for i, item in enumerate(borgan.encoding_xyt1) if item == xyt]
        for xyt in ["x", "y", "t", "s"]
    ]
    for i, cohort in enumerate(cohorts):
        dats = []
        for Patients in [
            cohort.Patients_train,
            cohort.Patients_valid,
            cohort.Patients_test,
        ]:
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
            with open(
                path + "data_train" + "fold_" + str(i) + "_" + name + ".pkl", "wb"
            ) as file:
                pickle.dump(dats[0], file)
            with open(
                path + "data_valid" + "fold_" + str(i) + "_" + name + ".pkl", "wb"
            ) as file:
                pickle.dump(dats[1], file)
            with open(
                path + "data_test" + "fold_" + str(i) + "_" + name + ".pkl", "wb"
            ) as file:
                pickle.dump(dats[2], file)

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


def load_missing_data_train_test(path, name=""):
    # load data
    # path = '/cluster/work/medinfmk/EUSTAR2/data/processed/'
    mod_name_train = "data_train" + name + ".pkl"
    mod_name_test = "data_test" + name + ".pkl"
    names_splits = "names_splits" + name + ".pkl"

    with open(path + mod_name_train, "rb") as file:
        data_train = pickle.load(file)

    with open(path + mod_name_test, "rb") as file:
        data_test = pickle.load(file)

    with open(path + names_splits, "rb") as file:
        names, splits, xyt0, xyt1 = pickle.load(file)

    return data_train, data_test, names, splits, xyt0, xyt1


def load_cv(path, n_folds=5, name=""):
    # load data
    # path = '/cluster/work/medinfmk/EUSTAR2/data/processed/'
    mod_name_train = "data_train"
    mod_name_test = "data_test"
    mod_name_valid = "data_valid"
    names_splits = "names_splits"
    data_train_folds = []
    data_valid_folds = []
    data_test_folds = []

    for i in range(n_folds):
        with open(
            path + mod_name_train + "fold_" + str(i) + "_" + name + ".pkl", "rb"
        ) as file:
            data_train_folds.append(pickle.load(file))
        with open(
            path + mod_name_valid + "fold_" + str(i) + "_" + name + ".pkl", "rb"
        ) as file:
            data_valid_folds.append(pickle.load(file))
        with open(
            path + mod_name_test + "fold_" + str(i) + "_" + name + ".pkl", "rb"
        ) as file:
            data_test_folds.append(pickle.load(file))

        with open(path + names_splits + name + ".pkl", "rb") as file:
            names, splits, xyt0, xyt1 = pickle.load(file)

    return (
        data_train_folds,
        data_valid_folds,
        data_test_folds,
        names,
        splits,
        xyt0,
        xyt1,
    )


# search for substrings in names
def find_substrings(names, sub):
    res = []
    for vc in names:
        if vc.find(sub) != -1:
            res.append(vc)
    return res


# defined here because pickle doesn't like lambda functions
def id_(x):
    return x


def arr_(x):
    return np.array(x)


# locifgal functions for dealing with nans
# TODO: make it efficient
def or_nan(x, y):
    """
    compute the "or" of 2 numerical (not boolean!) numpy arrays taking into account the nan's

    or | 0 | 1 | n |
     - - - - - - -
     0 | 0 | 1 | n |
     1 | 1 | 1 | 1 |
     n | n | 1 | n |

    """

    or_ = np.logical_or(x, y) * 1.0  # default np or

    x_nan = np.isnan(x)
    y_nan = np.isnan(y)

    or_[x_nan & y_nan] = np.nan
    or_[x_nan & (y == False)] = np.nan
    or_[y_nan & (x == False)] = np.nan

    return or_


def and_nan(x, y):
    """
    compute the "and" of 2 numerical (not boolean!) numpy arrays taking into account the nan's

    or | 0 | 1 | n |
     - - - - - - -
     0 | 0 | 1 | 0 |
     1 | 1 | 1 | n |
     n | 0 | n | n |

    """

    and_ = np.logical_and(x, y) * 1.0  # default np or

    x_nan = np.isnan(x)
    y_nan = np.isnan(y)

    and_[x_nan & y_nan] = np.nan
    and_[x_nan & (y == True)] = np.nan
    and_[y_nan & (x == True)] = np.nan

    return and_


def OR_nan(args):
    res = args[0]
    for arg in args[1:]:
        res = or_nan(res, arg)

    return res


def AND_nan(args):
    res = args[0]
    for arg in args[1:]:
        res = and_nan(res, arg)

    return res


def fun_nan(x, a, fun):
    """
    compute fun(x,a) and taking into account the nan's

         fun     | x!=n | x==n
     - - - -  - -  - -
     fun(x,a)==1 |   1  |  n  |
     fun(x,a)==0 |   0  |  n  |

    """

    fun_ = fun(x, a) * 1.0
    fun_[np.isnan(x)] = np.nan

    return fun_


def less_nan(x, a):
    return fun_nan(x, a, np.less)


def less_equal_nan(x, a):
    return fun_nan(x, a, np.less_equal)


def greater_nan(x, a):
    return fun_nan(x, a, np.greater)


def greater_equal_nan(x, a):
    return fun_nan(x, a, np.greater_equal)


def equal_nan(x, a):
    return fun_nan(x, a, np.equal)


def less_(x, a):
    return np.less(x, a) * 1.0


def less_equal_(x, a):
    return np.less_equal(x, a) * 1.0


def greater_(x, a):
    return np.greater(x, a) * 1.0


def greater_equal_(x, a):
    return np.greater_equal(x, a) * 1.0


def equal_(x, a):
    return np.equal(x, a) * 1.0


def logical_nan(x, a, logical="<"):
    if logical == "<":
        return less_nan(x, a)
    elif logical == "<=":
        return less_equal_nan(x, a)
    elif logical == "=":
        return equal_nan(x, a)
    elif logical == ">=":
        return greater_equal_nan(x, a)
    elif logical == ">":
        return greater_nan(x, a)
    else:
        raise ValueError("Logical is not defined")


def logical_nan_not_satisfied(x, a, logical="<"):
    if logical == "<":
        return np.less(x, a) * 1.0
    elif logical == "<=":
        return np.less_equal(x, a) * 1.0
    elif logical == "=":
        return np.equal(x, a) * 1.0
    elif logical == ">=":
        return np.greater_equal(x, a) * 1.0
    elif logical == ">":
        return np.greater(x, a) * 1.0
    else:
        raise ValueError("Logical is not defined")


def not_nan(x):
    """
    compute the "not" of a numerical (not boolean!) numpy array taking into account the nan's

    not|
     - - - -
     0 | 1 |
     1 | 0 |
     n | n |

    """

    not_ = 1 - x
    # not_[np.isnan(x)] = np.nan

    return not_


def where_nan(x):
    wh = np.where(x)[0]

    if len(wh) == 0:
        res = np.nan
    else:
        res = wh[0]

    return res


def get_classifier_config(names_y0, splits_y0, config_dict):
    # create config objects for classifiers
    # config dict must be of the form: {'name_1': {'y_names': [], 'z_dims': [], 'layers': []}, 'name_2': {}, ...}
    # It creates one classifier per key, y_names are the names of the corresponding variables, z dims are the dimensions
    # to use in z and layers are the hidden layer dims for the classifier.
    # E.g. config_dict= {'lung': {'y_names': ['LUNG_ILD_involvement_or', 'LUNG_ILD_stage_or'], 'z_dims': np.arange(0,2) , 'layers': [100]},
    # 'heart': {'y_names': ['HEART_involvement_or'], 'z_dims' : np.arange(1,2), 'layers': [100]}}
    classifier_configs = []
    # to select the corresponding dimensions in y
    Y_DIMS = np.cumsum(splits_y0)
    for key, config in config_dict.items():
        y_indices = [names_y0.index(name) for name in config["y_names"]]
        config["z_dims"] = config["z_dims"].tolist()
        config["y_dims"] = np.concatenate(
            [
                np.arange(Y_DIMS[index - 1], Y_DIMS[index])
                if index > 0
                else np.arange(0, Y_DIMS[index])
                for index in y_indices
            ]
        ).tolist()
        config["y_indices"] = y_indices
        config["output_dim"] = len(config["y_dims"])
        # config["input_dim"] = len(config["z_dims"])
        config["input_dim"] = (
            config["input_dim"] if "input_dim" in config else len(config["z_dims"])
        )
        config["organ_name"] = key
        classifier_configs.append(ClassifierConfig.from_dict(config))

    return classifier_configs
