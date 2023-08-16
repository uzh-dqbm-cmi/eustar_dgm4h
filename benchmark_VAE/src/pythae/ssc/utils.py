import numpy as np
import os, pickle
import sys

# sys.path.append(
#    "/home/cctrotte/krauthammer/eustar/benchmark_VAE/src/pythae/models/beta_vae_gp"
# )

# sys.path.append("/home/cctrotte/krauthammer/eustar/benchmark_VAE/src/")
sys.path.append("/cluster/work/medinfmk/EUSTAR2/code_ms/benchmark_VAE/src/pythae/")
# sys.path.append("/cluster/work/medinfmk/EUSTAR2/code_ct/benchmark_VAE/src/pythae/")

# current_dir = os.getcwd()
# os.chdir('/cluster/work/medinfmk/EUSTAR2/code_ms/benchmark_VAE/src/pythae/')
from pythae.data.datasets import MissingDataset

# from benchmark_VAE.src.pythae.models.beta_vae_gp.classifier_config import ClassifierConfig
from pythae.models.beta_vae_gp.classifier_config import ClassifierConfig


# os.chdir( current_dir )


# def N_log_prob(mu, log_var, y):
#     """
#     Evaluate the negative log probability -log p( y | z ) = -log N( y | mu(z), sigma(z) )
#     of the independent Guassian likelihood for observed data y.
#     mu: N x T x P
#     log_var: N x T x P
#     y: N x T x P
#     log_prob: 1
#     """

#     # norm = np.log( 2 * np.pi ) + log_var # ignore normalozation factor
#     # norm =  log_var
#     # square = (y - mu).pow(2) / log_var.exp()

#     # neg_log_prob = 0.5*(norm + square)#.sum()
#     neg_log_prob = log_var + (y - mu).pow(2) / log_var.exp()

#     return neg_log_prob


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

