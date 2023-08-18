import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

from sklearn.preprocessing import (
    StandardScaler,
    OrdinalEncoder,
    OneHotEncoder,
    MinMaxScaler,
)

from pythae.ssc.utils import id_, arr_

from torch.nn.functional import softmax
import torch


class Variable:
    def __init__(
        self,
        name,
        kind="continuous",
        loss="mse",
        nan_value=0.0,
        encoding="mean_var",
        xyt="x",
    ):
        # TODO: make it consitent and include restriction, loss, nan_value

        self.name = name

        assert kind in ["continuous", "binary", "categorical", "ordinal"]
        self.kind = kind

        assert loss in ["mse", "binary_CE", "multi_CE"]  # just define likelihoods!
        self.loss = loss

        # assert restriction in ['none', 'positive', 'percentage'] #range?
        # self.restriction = restriction

        # list of discrete levels [0,1,2]
        # self.levels = levels

        # default nan value AFTER transformation!
        self.nan_value = nan_value

        assert encoding in ["none", "mean_var", "zero_one", "one_hot", "ordinal"]
        self.encoding = encoding

        assert xyt in ["x", "y", "t", "s"]
        self.xyt = xyt

    def fill_nan(self, DF):
        return DF.fillna(self.nan_value)

    def fit_encode(self, Df_train):
        """
        Df_train is df with which should contain a column with self.name
        """

        # no encoding
        if self.encoding == "none":
            self.encoder = arr_
            self.decoder = id_
            self.enc = None
            self.encoding_names = [self.name]
        else:
            if self.encoding == "mean_var":
                self.enc = StandardScaler()

            elif self.encoding == "ordinal":
                self.enc = OrdinalEncoder()

            elif self.encoding == "one_hot":
                # ignore the nan category!
                labels = np.sort(Df_train[self.name].unique())
                labels = [labels[~np.isnan(labels)]]

                self.enc = OneHotEncoder(
                    sparse=False,
                    drop="if_binary",
                    categories=labels,
                    handle_unknown="ignore",
                )

            elif self.encoding == "zero_one":
                self.enc = MinMaxScaler()

            # fit to training data
            self.enc.fit(Df_train[[self.name]])
            self.encoder = self.enc.transform
            self.decoder = self.enc.inverse_transform

            self.encoding_names = self.enc.get_feature_names_out()
            if self.encoding == "one_hot":
                cats = self.enc.transform(Df_train[[self.name]].dropna())
                # multi class
                if cats.shape[1] > 1:
                    class_freq = np.sum(cats, axis=0)
                    class_weight = sum(class_freq) / class_freq
                    class_weight_norm = class_weight / np.sum(class_weight)
                else:
                    # binary: class_weight = #neg/#pos
                    class_weight_norm = (len(cats) - sum(cats)) / sum(cats)
                self.class_weight_norm = list(class_weight_norm)
            else:
                self.class_weight_norm = None

    def encode(self, Df, fill_nan=False):
        """
        apply the encoder to (new) data Df, which should contain a column with self.name
        """
        array = self.encoder(Df[[self.name]])

        if self.encoding == "one_hot":
            array[Df[self.name].isnull(), :] = np.nan
        # also need this beause otherwise NaT are not detected
        if self.encoding == "mean_var":
            array[Df[self.name].isnull(), :] = np.nan

        if fill_nan:
            array[np.isnan(array)] = self.nan_value

        return array

    def decode(self, Z):
        """
        apply the decoding to Z
        """

        return self.decoder(Z)

    def decode_(self, z):
        if self.kind in ["continuous", "ordinal"]:
            res = (z, z)

        elif self.kind == "binary":
            probs = torch.sigmoid(z)
            classes = (probs > 0.5) * 1

            res = (classes, probs)

        elif self.kind == "categorical":
            probs = softmax(z, dim=1)
            classes = torch.argmax(probs, dim=1)

            res = (classes.unsqueeze(1), probs)

        return res

    def get_categories(self, array):
        return torch.tensor([self.enc.categories[0][index] for index in array])
