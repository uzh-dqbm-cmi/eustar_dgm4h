"""The pythae's Datasets inherit from
:class:`torch.utils.data.Dataset` and must be used to convert the data before
training. As of today, it only contains the :class:`pythae.data.BaseDatset` useful to train a
VAE model but other Datatsets will be added as models are added.
"""
from collections import OrderedDict
from typing import Any, Tuple

import torch
from torch.utils.data import Dataset
import numpy as np

import itertools
from torch.utils.data import DataLoader


class DatasetOutput(OrderedDict):
    """Base DatasetOutput class fixing the output type from the dataset. This class is inspired from
    the ``ModelOutput`` class from hugginface transformers library"""

    def __getitem__(self, k):
        if isinstance(k, str):
            self_dict = {k: v for (k, v) in self.items()}
            return self_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())


class BaseDataset(Dataset):
    """This class is the Base class for pythae's dataset

    A ``__getitem__`` is redefined and outputs a python dictionnary
    with the keys corresponding to `data` and `labels`.
    This Class should be used for any new data sets.
    """

    def __init__(self, data, labels):
        self.labels = labels.type(torch.float)
        self.data = data.type(torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Generates one sample of data

        Args:
            index (int): The index of the data in the Dataset

        Returns:
            (dict): A dictionnary with the keys 'data' and 'labels' and corresponding
            torch.Tensor
        """
        # Select sample
        X = self.data[index]
        y = self.labels[index]

        return DatasetOutput(data=X, labels=y)


class MissingDataset(BaseDataset):
    """This class is a missing class for pythae's dataset

    A ``__getitem__`` is redefined and outputs a tuple
    """

    def __init__(
        self, list_x, list_y, list_t, list_s, missing_x, missing_y, missing_t, missing_s
    ):
        self.list_x = [torch.tensor(dx, dtype=torch.float) for dx in list_x]
        self.list_y = [torch.tensor(dy, dtype=torch.float) for dy in list_y]
        self.list_t = [torch.tensor(dt, dtype=torch.float) for dt in list_t]
        self.list_s = [torch.tensor(ds, dtype=torch.float) for ds in list_s]

        self.missing_x = [torch.tensor(dx, dtype=torch.bool) for dx in missing_x]
        self.missing_y = [torch.tensor(dy, dtype=torch.bool) for dy in missing_y]
        self.missing_t = [torch.tensor(dt, dtype=torch.bool) for dt in missing_t]
        self.missing_s = [torch.tensor(ds, dtype=torch.bool) for ds in missing_s]

    def __len__(self):
        return len(self.list_x)

    def __getitem__(self, index):
        """Generates one sample of data

        Args:
            index (int): The index of the data in the Dataset

        Returns:
            tuple
        """

        # Select sample
        dx = self.list_x[index]
        dy = self.list_y[index]
        dt = self.list_t[index]
        ds = self.list_s[index]
        mx = self.missing_x[index]
        my = self.missing_y[index]
        mt = self.missing_t[index]
        ms = self.missing_s[index]

        # return DatasetOutput(data_x=data_x)
        return dx, dy, dt, ds, mx, my, mt, ms

    def get_ith_sample_batch_with_customDataLoader(self, i, batch_size=1):
        DatLoader = DataLoader(
            self, batch_size=batch_size, shuffle=False, collate_fn=custom_collate
        )

        ith_batch = next(itertools.islice(DatLoader, i, None))

        return ith_batch

    def subsample(self, N, seed=0, permute=True):
        assert N <= self.__len__(), "N too large"

        torch.manual_seed(seed)

        if permute:
            perm = torch.randperm(self.__len__())[:N]
        else:
            perm = torch.arange(N)

        return MissingDataset(
            [self.list_x[p] for p in perm],
            [self.list_y[p] for p in perm],
            [self.list_t[p] for p in perm],
            [self.list_s[p] for p in perm],
            [self.missing_x[p] for p in perm],
            [self.missing_y[p] for p in perm],
            [self.missing_t[p] for p in perm],
            [self.missing_s[p] for p in perm],
        )

    def reshape_for_prediction(self, batch, strategy, max_num_to_pred):
        splits = batch["splits"]
        missing_y = batch["missing_y"]
        data_y = batch["data_y"]
        data_x = batch["data_x"]
        missing_x = batch["missing_x"]
        data_t = batch["data_t"]
        missing_t = batch["missing_t"]
        missing_y_splitted = missing_y.split(splits)
        missing_x_splitted = missing_x.split(splits)
        missing_t_splitted = missing_t.split(splits)
        data_y_splitted = data_y.split(splits)
        data_x_splitted = data_x.split(splits)
        data_t_splitted = data_t.split(splits)

        if strategy == "last":

            num_for_rec = [
                [i for i in range(max(1, elem - max_num_to_pred), elem)]
                for elem in splits
            ]
            num_to_pred = [
                [
                    min(max_num_to_pred, elem - i)
                    for i in range(max(1, elem - max_num_to_pred), elem)
                ]
                for elem in splits
            ]
            # num_for_rec = [[elem - 1] for elem in splits]
            # missing_y_recon = torch.cat(
            #     [elem[:-1, :] for elem in missing_y.split(splits)]
            # )
            # data_y_recon = torch.cat([elem[:-1, :] for elem in data_y.split(splits)])
            # missing_x_recon = torch.cat(
            #     [elem[:-1, :] for elem in missing_x.split(splits)]
            # )
            # data_x_recon = torch.cat([elem[:-1, :] for elem in data_x.split(splits)])
            # data_t_recon = torch.cat([elem[:-1, :] for elem in data_t.split(splits)])
            # missing_t_recon = torch.cat(
            #     [elem[:-1, :] for elem in missing_t.split(splits)]
            # )
            # data_y_pred = torch.cat([elem[-1, :] for elem in data_y.split(splits)])
            # missing_y_pred = torch.cat(
            #     [elem[-1, :] for elem in missing_y.split(splits)]
            # )
            # data_x_pred = torch.cat([elem[-1, :] for elem in data_x.split(splits)])
            # missing_x_pred = torch.cat(
            #     [elem[-1, :] for elem in missing_x.split(splits)]
            # )
            # data_t_pred = torch.cat([elem[-1, :] for elem in data_t.split(splits)])
            # missing_t_pred = torch.cat(
            #     [elem[-1, :] for elem in missing_t.split(splits)]
            # )

        elif strategy == "all":
            num_for_rec = [list(range(1, elem)) if elem > 1 else [0] for elem in splits]
            splits_flattened = [item + 1 for sublist in num_for_rec for item in sublist]
            missing_y_recon = torch.cat(
                [
                    missing_y_splitted[pat][: num_for_rec[pat][vis], :]
                    for pat, num_v in enumerate(splits)
                    for vis in range(len(num_for_rec[pat]))
                ]
            )
            data_y_recon = torch.cat(
                [
                    data_y_splitted[pat][: num_for_rec[pat][vis], :]
                    for pat, num_v in enumerate(splits)
                    for vis in range(len(num_for_rec[pat]))
                ]
            )
            missing_x_recon = torch.cat(
                [
                    missing_x_splitted[pat][: num_for_rec[pat][vis], :]
                    for pat, num_v in enumerate(splits)
                    for vis in range(len(num_for_rec[pat]))
                ]
            )
            data_x_recon = torch.cat(
                [
                    data_x_splitted[pat][: num_for_rec[pat][vis], :]
                    for pat, num_v in enumerate(splits)
                    for vis in range(len(num_for_rec[pat]))
                ]
            )
            data_t_recon = torch.cat(
                [
                    data_t_splitted[pat][: num_for_rec[pat][vis], :]
                    for pat, num_v in enumerate(splits)
                    for vis in range(len(num_for_rec[pat]))
                ]
            )
            missing_t_recon = torch.cat(
                [
                    missing_t_splitted[pat][: num_for_rec[pat][vis], :]
                    for pat, num_v in enumerate(splits)
                    for vis in range(len(num_for_rec[pat]))
                ]
            )
            data_y_pred = torch.cat(
                [
                    data_y_splitted[pat][num_for_rec[pat][vis], :]
                    for pat, num_v in enumerate(splits)
                    for vis in range(len(num_for_rec[pat]))
                ]
            )
            missing_y_pred = torch.cat(
                [
                    missing_y_splitted[pat][num_for_rec[pat][vis], :]
                    for pat, num_v in enumerate(splits)
                    for vis in range(len(num_for_rec[pat]))
                ]
            )
            data_x_pred = torch.cat(
                [
                    data_x_splitted[pat][num_for_rec[pat][vis], :]
                    for pat, num_v in enumerate(splits)
                    for vis in range(len(num_for_rec[pat]))
                ]
            )
            missing_x_pred = torch.cat(
                [
                    missing_x_splitted[pat][num_for_rec[pat][vis], :]
                    for pat, num_v in enumerate(splits)
                    for vis in range(len(num_for_rec[pat]))
                ]
            )
            data_t_pred = torch.cat(
                [
                    data_t_splitted[pat][num_for_rec[pat][vis], :]
                    for pat, num_v in enumerate(splits)
                    for vis in range(len(num_for_rec[pat]))
                ]
            )
            missing_t_pred = torch.cat(
                [
                    missing_t_splitted[pat][num_for_rec[pat][vis], :]
                    for pat, num_v in enumerate(splits)
                    for vis in range(len(num_for_rec[pat]))
                ]
            )

        out = DatasetOutput(
            {
                "splits": splits,
                "num_for_rec": num_for_rec,
                "data_x": data_x,
                "data_y": data_y,
                "data_t": data_t,
                "missing_x": missing_x,
                "missing_y": missing_y,
                "missing_t": missing_t,
                "missing_y_recon": missing_y_recon,
                "missing_x_recon": missing_x_recon,
                "missing_t_recon": missing_t_recon,
                "data_y_recon": data_y_recon,
                "data_x_recon": data_x_recon,
                "data_t_recon": data_t_recon,
                "data_y_pred": data_y_pred,
                "data_x_pred": data_x_pred,
                "data_t_pred": data_t_pred,
                "missing_y_pred": missing_y_pred,
                "missing_x_pred": missing_x_pred,
                "missing_t_pred": missing_t_pred,
            }
        )

        return out


def custom_collate(batch_list):
    # hard coded names!
    names = [
        "data_x",
        "data_y",
        "data_t",
        "data_s",
        "missing_x",
        "missing_y",
        "missing_t",
        "missing_s",
    ]

    # Separate the items in the batch_list
    items_tuple = zip(*batch_list)

    # iterate over all items and safe it in output
    DataOut = DatasetOutput()
    for i, items in enumerate(items_tuple):
        DataOut[names[i]] = torch.cat([item for item in items])

    # compute split lengths for going back to patients
    # DataOut['cum_splits'] = torch.cumsum( torch.tensor([item[0].shape[0] for item in batch_list]), 0)[:-1]
    DataOut["splits"] = [item[0].shape[0] for item in batch_list]

    return DataOut
