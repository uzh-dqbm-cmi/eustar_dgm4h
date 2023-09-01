import pandas as pd
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import random
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from pythae.ssc.plots import *
import imageio as iio
import os


class EvalPatient:
    def __init__(
        self,
        data_test,
        model,
        body,
        splits_x0,
        names_x0,
        kinds_x0,
        splits_y0,
        names_y0,
        kinds_y0,
        names_s0,
        kinds_x1,
        names_x1,
        size,
        batch_num=0,
    ):
        seed = 0
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = model
        self.body = body
        self.splits_x0 = splits_x0
        self.names_x0 = names_x0
        self.names_x1 = names_x1
        self.kinds_x0 = kinds_x0
        self.splits_y0 = splits_y0
        self.names_y0 = names_y0
        self.kinds_y0 = kinds_y0
        self.names_s0 = names_s0
        self.kinds_x1 = kinds_x1
        self.batch = data_test.get_ith_sample_batch_with_customDataLoader(
            batch_num, size
        )
        self.batch_num = batch_num
        self.data_x = self.batch["data_x"]
        self.data_s = self.batch["data_s"]
        self.data_t = self.batch["data_t"]
        self.missing_x = self.batch["missing_x"]
        self.non_missing_x = 1 - self.batch["missing_x"] * 1.0
        self.splits = self.batch["splits"]  # N_patients
        self.times = self.data_t[:, 0].reshape(-1, 1)  # N_patients x 1
        self.non_missing_y = 1 - self.batch["missing_y"] * 1.0  # N_patients x n_class
        self.non_missing_s = 1 - self.batch["missing_s"] * 1.0
        self.non_missing_s_splitted = torch.split(
            self.non_missing_s, self.splits, dim=0
        )
        self.data_y = self.batch["data_y"]  # N_patients x n_class
        self.data_x_splitted = torch.split(self.data_x, self.splits, dim=0)
        self.data_y_splitted = torch.split(self.data_y, self.splits, dim=0)
        self.data_s_splitted = torch.split(self.data_s, self.splits, dim=0)
        self.non_missing_x_splitted = torch.split(
            self.non_missing_x, self.splits, dim=0
        )
        self.non_missing_y_splitted = torch.split(
            self.non_missing_y, self.splits, dim=0
        )
        self.times_splitted = torch.split(self.times, self.splits, dim=0)
        self.data_x_recon = torch.cat(
            [
                self.data_x_splitted[pat].repeat(self.splits[pat] + 1, 1)
                for pat in range(len(self.splits))
            ]
        )
        self.data_s_recon = torch.cat(
            [
                self.data_s_splitted[pat].repeat(self.splits[pat] + 1, 1)
                for pat in range(len(self.splits))
            ]
        )
        self.data_y_recon = torch.cat(
            [
                self.data_y_splitted[pat].repeat(self.splits[pat] + 1, 1)
                for pat in range(len(self.splits))
            ]
        )
        self.non_missing_x_recon = torch.cat(
            [
                self.non_missing_x_splitted[pat].repeat(self.splits[pat] + 1, 1)
                for pat in range(len(self.splits))
            ]
        )
        self.non_missing_y_recon = torch.cat(
            [
                self.non_missing_y_splitted[pat].repeat(self.splits[pat] + 1, 1)
                for pat in range(len(self.splits))
            ]
        )
        self.times_recon = torch.cat(
            [
                self.times_splitted[pat].repeat(self.splits[pat] + 1, 1)
                for pat in range(len(self.splits))
            ]
        )
        self.indices_recon = torch.cat(
            [
                torch.cat(
                    [
                        torch.cat(
                            [
                                torch.full((index, 1), True),
                                torch.full((self.splits[pat] - index, 1), False),
                            ],
                            dim=0,
                        )
                        for index in range(0, self.splits[pat] + 1)
                    ]
                )
                for pat in range(len(self.splits))
            ]
        ).flatten()
        self.num_rec_for_pred = np.array(
            torch.cat(
                [
                    torch.cat(
                        [
                            torch.full((self.splits[pat], 1), index)
                            for index in range(0, self.splits[pat] + 1)
                        ]
                    )
                    for pat in range(len(self.splits))
                ]
            )
        )
        if not self.model.model_config.retrodiction:
            self.absolute_times = torch.cat(
                [
                    torch.cat(
                        [
                            self.times_splitted[pat][0].repeat(
                                len(self.times_splitted[pat]), 1
                            ),
                            torch.cat(
                                [
                                    torch.cat(
                                        [
                                            self.times_splitted[pat][:index, :],
                                            self.times_splitted[pat][index, :].repeat(
                                                len(self.times_splitted[pat]) - index, 1
                                            ),
                                        ],
                                        dim=0,
                                    )
                                    for index in range(len(self.times_splitted[pat]))
                                ]
                            ),
                        ]
                    )
                    for pat in range(len(self.times_splitted))
                ]
            )
        else:
            self.absolute_times = torch.cat(
                [
                    torch.cat(
                        [
                            self.times_splitted[pat][0].repeat(
                                len(self.times_splitted[pat]), 1
                            ),
                            torch.cat(
                                [
                                    elem.repeat(len(self.times_splitted[pat]), 1)
                                    for elem in self.times_splitted[pat]
                                ]
                            ),
                        ]
                    )
                    for pat in range(len(self.times_splitted))
                ]
            )
        self.absolute_times_resc = self.body.get_var_by_name("time [years]").decode(
            self.absolute_times
        )
        self.times_recon_resc = self.body.get_var_by_name("time [years]").decode(
            self.times_recon
        )
        self.delta_t_resc = self.times_recon_resc - self.absolute_times_resc

    def evaluate(self, num_samples=10, evaluate_y=True):
        self.predictions = self.model(self.batch)
        self.samples = [self.model(self.batch) for index in range(num_samples)]
        self.ground_truth_x = self.body.decode(
            self.data_x_recon, self.splits_x0, self.names_x0
        )
        if self.model.sample_z:
            self.res_matrix, self.probs_matrix, self.res_list = self.body.decode_preds(
                self.predictions.recon_x, self.splits_x0, self.names_x0
            )
            self.res_list_samples = [
                self.body.decode_preds(sample.recon_x, self.splits_x0, self.names_x0)[2]
                for sample in self.samples
            ]
        else:
            self.res_matrix, self.probs_matrix, self.res_list = self.body.decode_preds(
                self.predictions.recon_m, self.splits_x0, self.names_x0
            )
            self.res_list_samples = [
                self.body.decode_preds(sample.recon_m, self.splits_x0, self.names_x0)[2]
                for sample in self.samples
            ]
        if evaluate_y:
            self.ground_truth_y = self.body.decode(
                self.data_y_recon, self.splits_y0, self.names_y0
            )
            if self.model.sample_z:
                (
                    self.res_matrix_y,
                    self.probs_matrix_y,
                    self.res_list_y,
                ) = self.body.decode_preds(
                    self.predictions.y_out_rec, self.splits_y0, self.names_y0
                )
            else:
                (
                    self.res_matrix_y,
                    self.probs_matrix_y,
                    self.res_list_y,
                ) = self.body.decode_preds(
                    self.predictions.y_out_m_rec, self.splits_y0, self.names_y0
                )

            self.predicted_cats_y = torch.empty_like(self.res_matrix_y)
            for index, var in enumerate(self.names_y0):
                self.predicted_cats_y[:, index] = self.body.get_var_by_name(
                    var
                ).get_categories(self.res_matrix_y[:, index])

    def plot_continuous_x(self, figure_path, plot_missing=False):
        names_cont = [
            elem
            for index, elem in enumerate(self.names_x1)
            if self.kinds_x1[index] in ["continuous", "ordinal"]
        ]
        i = self.batch_num
        for num_rec in range(self.splits[0] + 1):
            # predicted samples
            x_recon = torch.stack(
                [
                    torch.split(elem.recon_x, self.splits * (self.splits[0] + 1))[
                        num_rec
                    ]
                    for elem in self.samples
                ]
            )
            x_recon_log_var = torch.stack(
                [
                    torch.split(
                        elem.recon_x_log_var, self.splits * (self.splits[0] + 1)
                    )[num_rec]
                    for elem in self.samples
                ]
            )
            tmp = torch.normal(x_recon, torch.exp(x_recon_log_var))
            x_recon_means = torch.mean(tmp, dim=0)
            x_recon_std = torch.std(tmp, dim=0)
            f = plot_x_overlaid(
                self.body,
                self.data_x,
                x_recon[0].reshape(1, len(self.times), self.data_x.shape[1]),
                torch.sqrt(torch.exp(x_recon_log_var[0])).reshape(
                    1, len(self.times), self.data_x.shape[1]
                ),
                self.body.get_var_by_name("time [years]")
                .decode(self.data_t[:, 0].reshape(-1, 1))
                .flatten(),
                self.missing_x,
                num_rec,
                self.names_x1,
                self.kinds_x1,
                names_cont,
                plot_missing,
            )
            # uncomment to plot mean and var of samples
            # f = plot_x_overlaid(self.data_x, x_recon_means.reshape(1, len(self.times),self.data_x.shape[1]), x_recon_std.reshape(1, len(self.times), self.data_x.shape[1]),self.data_t[:, 0], self.missing_x, num_rec,  self.names_x1, self.kinds_x1, names_cont)

            if not os.path.exists(figure_path + "/" + str(i)):
                os.makedirs(figure_path + "/" + str(i))
            f.savefig(figure_path + "/" + str(i) + "/" + "_" + str(num_rec))
        filenames = [
            figure_path + "/" + str(i) + "/" + "_" + str(num_rec) + ".png"
            for num_rec in range(self.splits[0])
        ]
        images = []
        for filename in filenames:
            images.append(iio.imread(filename))
        iio.mimsave(
            figure_path + "/" + str(i) + "/" + "all" + ".gif", images, duration=0.6
        )
        return

    def plot_categorical_x(self, figure_path):
        # sample only one point and plot the predicted probas
        probas = [
            torch.split(elem[1], self.splits * (self.splits[0] + 1))
            for elem in self.res_list
        ]
        i = self.batch_num
        for var_index, k in enumerate(self.kinds_x0):
            if k != "continuous":
                name = self.names_x0[var_index]
                categories = self.body.get_var_by_name(name).enc.categories_[0]
                ground_truth = self.body.decode(
                    self.data_x, self.splits_x0, self.names_x0
                )[1][var_index]
                non_miss = torch.split(self.non_missing_x, self.splits_x0, dim=1)[
                    var_index
                ][:, 0]
                for num_rec in range(self.splits[0]):
                    probs = probas[var_index][num_rec]
                    if probs.shape[1] == 1:
                        probs = torch.cat([1 - probs, probs], axis=1)
                    f = plot_categorical_preds(
                        self.body.get_var_by_name("time [years]")
                        .decode(self.data_t[:, 0].reshape(-1, 1))
                        .flatten(),
                        categories,
                        probs.detach().T,
                        ground_truth.flatten(),
                        non_miss,
                        self.splits[0] - num_rec,
                        name,
                    )
                    if not os.path.exists(figure_path + "/" + str(i)):
                        os.makedirs(figure_path + "/" + str(i))
                    f.savefig(
                        figure_path
                        + "/"
                        + str(i)
                        + "/"
                        + name.split("/")[0]
                        + "_"
                        + str(num_rec)
                    )
                filenames = [
                    figure_path
                    + "/"
                    + str(i)
                    + "/"
                    + name.split("/")[0]
                    + "_"
                    + str(num_rec)
                    + ".png"
                    for num_rec in range(self.splits[0])
                ]
                images = []
                for filename in filenames:
                    images.append(iio.imread(filename))
                iio.mimsave(
                    figure_path + "/" + str(i) + "/" + name.split("/")[0] + ".gif",
                    images,
                    duration=0.6,
                )
        return

    def plot_continuous_feature(self, name, figure_path, plot_missing=False):

        i = self.batch_num
        figsize = (6, 2)

        f, axs = plt.subplots(
            self.splits[0] + 1,
            1,
            sharex=False,
            sharey=False,
            figsize=(1 * figsize[0], (self.splits[0] + 1) * (figsize[1] + 1)),
        )
        f.subplots_adjust(hspace=0.5)  # , wspace=0.2)
        index_name = self.names_x1.index(name)
        non_miss = ~self.missing_x[:, index_name]

        for num_rec in range(self.splits[0] + 1):
            ax = axs[num_rec]
            # predicted samples
            x_recon = torch.stack(
                [
                    torch.split(elem.recon_x, self.splits * (self.splits[0] + 1))[
                        num_rec
                    ]
                    for elem in self.samples
                ]
            )
            x_recon_log_var = torch.stack(
                [
                    torch.split(
                        elem.recon_x_log_var, self.splits * (self.splits[0] + 1)
                    )[num_rec]
                    for elem in self.samples
                ]
            )

            recon_x_means = x_recon[0].reshape(1, len(self.times), self.data_x.shape[1])
            recon_x_stds = torch.sqrt(torch.exp(x_recon_log_var[0])).reshape(
                1, len(self.times), self.data_x.shape[1]
            )
            time = (
                self.body.get_var_by_name("time [years]")
                .decode(self.data_t[:, 0].reshape(-1, 1))
                .flatten()
            )
            recon_x_means = [elem.detach().numpy() for elem in recon_x_means]
            recon_x_stds = [elem.detach().numpy() for elem in recon_x_stds]

            colors = plt.cm.Blues(np.linspace(0.3, 1, 1))

            mean_rescaled = (
                self.body.get_var_by_name(name)
                .decode(recon_x_means[0][:, index_name].reshape(-1, 1))
                .flatten()
            )
            stds_rescaled = (
                self.body.get_var_by_name(name).enc.scale_
                * recon_x_stds[0][:, index_name]
            )

            ax.fill_between(
                time,
                mean_rescaled - 2 * stds_rescaled,
                mean_rescaled + 2 * stds_rescaled,
                alpha=0.5,
                color=colors[0],
            )

            if plot_missing:
                ax.plot(
                    time,
                    self.body.get_var_by_name(name)
                    .decode(self.data_x[:, index_name].reshape(-1, 1))
                    .flatten(),
                    ".-",
                    color="C2",
                    label="ground truth",
                )
                ax.plot(
                    time,
                    mean_rescaled,
                    ".-",
                    color="black",
                    label="predictions",
                )

            else:
                if self.data_x[non_miss, index_name].shape[0] > 0:
                    ax.plot(
                        time[non_miss],
                        self.body.get_var_by_name(name)
                        .decode(self.data_x[non_miss, index_name].reshape(-1, 1))
                        .flatten(),
                        ".-",
                        color="C2",
                        label="ground truth",
                    )
                ax.plot(
                    time[non_miss],
                    mean_rescaled[non_miss],
                    ".-",
                    color="black",
                    label="predictions",
                )

            if self.data_x[non_miss, index_name].shape[0] > 0:
                ax.plot(
                    time[non_miss],
                    self.body.get_var_by_name(name)
                    .decode(self.data_x[non_miss, index_name].reshape(-1, 1))
                    .flatten(),
                    "o",
                    color="C3",
                    label="available",
                )

            ax.set_title(name)
            if num_rec < len(time):
                ax.axvline(time[num_rec] - 0.05, ls="--")

            if self.data_x[non_miss, index_name].shape[0] > 0:
                y_min = (
                    min(
                        self.body.get_var_by_name(name).enc.mean_,
                        min(
                            self.body.get_var_by_name(name)
                            .decode(self.data_x[non_miss, index_name].reshape(-1, 1))
                            .flatten()
                        ),
                    )
                    - 2 * self.body.get_var_by_name(name).enc.scale_
                )
                y_max = (
                    max(
                        self.body.get_var_by_name(name).enc.mean_,
                        max(
                            self.body.get_var_by_name(name)
                            .decode(self.data_x[non_miss, index_name].reshape(-1, 1))
                            .flatten()
                        ),
                    )
                    + 2 * self.body.get_var_by_name(name).enc.scale_
                )
            else:
                y_min = (
                    self.body.get_var_by_name(name).enc.mean_
                    - 2 * self.body.get_var_by_name(name).enc.scale_
                )
                y_max = (
                    self.body.get_var_by_name(name).enc.mean_
                    + 2 * self.body.get_var_by_name(name).enc.scale_
                )
            ax.set_ylim(
                y_min,
                y_max,
            )
            ax.set_xlabel("time [years]")
            ax.set_ylabel("Value")
            ax.legend(loc="lower left", fontsize="x-small")
            ax.grid(linestyle="--")

            # uncomment to plot mean and var of samples
            # f = plot_x_overlaid(self.data_x, x_recon_means.reshape(1, len(self.times),self.data_x.shape[1]), x_recon_std.reshape(1, len(self.times), self.data_x.shape[1]),self.data_t[:, 0], self.missing_x, num_rec,  self.names_x1, self.kinds_x1, names_cont)
            if not os.path.exists(figure_path + "/" + str(i)):
                os.makedirs(figure_path + "/" + str(i))
            f.savefig(
                figure_path
                + "/"
                + str(i)
                + "/"
                + name.split("/")[0].split("-")[0].split(".")[0]
            )

        return f

    def plot_y(self, figure_path):
        # res_matrix_y, probs_matrix_y, res_list_y = body.decode_preds(samples[0].y_out_rec, splits_y0, names_y0)
        probas_y = [
            torch.split(elem[1], self.splits * (self.splits[0] + 1))
            for elem in self.res_list_y
        ]
        i = self.batch_num
        for var_index, k in enumerate(self.kinds_y0):
            if k != "continuous":
                name = self.names_y0[var_index]
                categories = self.body.get_var_by_name(name).enc.categories_[0]
                ground_truth = self.body.decode(
                    self.data_y, self.splits_y0, self.names_y0
                )[1][var_index]
                non_miss = torch.split(self.non_missing_y, self.splits_y0, dim=1)[
                    var_index
                ][:, 0]
                for num_rec in range(self.splits[0] + 1):
                    probs = probas_y[var_index][num_rec]
                    if probs.shape[1] == 1:
                        probs = torch.cat([1 - probs, probs], axis=1)
                    f = plot_categorical_preds(
                        self.body.get_var_by_name("time [years]")
                        .decode(self.data_t[:, 0].reshape(-1, 1))
                        .flatten(),
                        categories,
                        probs.detach().T,
                        ground_truth.flatten(),
                        non_miss,
                        self.splits[0] - num_rec,
                        name,
                    )
                    f.savefig(
                        figure_path
                        + "/"
                        + str(i)
                        + "/"
                        + name.split("/")[0]
                        + "_"
                        + str(num_rec)
                    )
                filenames = [
                    figure_path
                    + "/"
                    + str(i)
                    + "/"
                    + name.split("/")[0]
                    + "_"
                    + str(num_rec)
                    + ".png"
                    for num_rec in range(self.splits[0])
                ]
                images = []
                for filename in filenames:
                    images.append(iio.imread(filename))
                iio.mimsave(
                    figure_path + "/" + str(i) + "/" + name.split("/")[0] + ".gif",
                    images,
                    duration=1.5,
                )
        return


class EvaluationDataset(EvalPatient):
    def __init__(
        self,
        data_test,
        model,
        body,
        splits_x0,
        names_x0,
        kinds_x0,
        splits_y0,
        names_y0,
        kinds_y0,
        names_s0,
        kinds_x1,
        names_x1,
        size,
        batch_num=0,
    ):
        super().__init__(
            data_test,
            model,
            body,
            splits_x0,
            names_x0,
            kinds_x0,
            splits_y0,
            names_y0,
            kinds_y0,
            names_s0,
            kinds_x1,
            names_x1,
            size,
            batch_num,
        )

    def get_patient_specific_baseline_x(self):

        list_ = np.concatenate(([0], np.cumsum(self.splits_x0)))
        patient_specific_baseline = []
        # iterate over x variables
        for index, elem in enumerate(list_[:-1]):
            naive_all = []
            if self.kinds_x0[index] == "continuous":
                for pat in range(len(self.data_x_splitted)):
                    # copy patient data
                    new_naive = self.data_x_splitted[pat][
                        :, elem : list_[index + 1]
                    ].clone()
                    # store mean value of cohort for imputation of missing values
                    mean_cohort = torch.mean(
                        self.data_x[self.non_missing_x[:, index] > 0, index]
                    )
                    # available values
                    mask = (
                        self.non_missing_x_splitted[pat][:, elem : list_[index + 1]] > 0
                    )
                    # fill missing values with previous value of patient
                    new_naive = self.fill_tensor(new_naive, mask)
                    # shift predictions so that we always predict the last available value.
                    # first concatenate two tensors (ie predictions before any info is available) filled with the mean value of the cohort
                    # then for m=1, ... T -1, recursively fill tensor so that if m values are available for prediction, we predict [mean_cohort, data[1], .., data[m], data[m], ..., data[m]] for ground truth [data[1], ..., data[T]]
                    new_naive = torch.cat(
                        [
                            torch.full(new_naive.shape, mean_cohort),
                            torch.full(new_naive.shape, mean_cohort),
                            torch.cat(
                                [
                                    torch.cat(
                                        [
                                            torch.tensor([[mean_cohort]]),
                                            new_naive[:index],
                                            new_naive[index].repeat(
                                                len(new_naive) - index - 1, 1
                                            ),
                                        ],
                                        dim=0,
                                    )
                                    for index in range(len(new_naive) - 1)
                                ]
                            ),
                        ]
                    )
                    naive_all.append(new_naive)
            else:
                for pat in range(len(self.data_x_splitted)):
                    new_naive = self.data_x_splitted[pat][
                        :, elem : list_[index + 1]
                    ].clone()
                    # mean_cohort = torch.mean(self.data_x[non_missing_x[:, index]>0, index])
                    mask = (
                        (
                            self.non_missing_x_splitted[pat][:, elem : list_[index + 1]]
                            > 0
                        )
                        .any(dim=1)
                        .reshape(-1, 1)
                        .repeat(new_naive.shape)
                    )
                    new_naive = self.fill_tensor(new_naive, mask, cat=True)
                    # same as for continuous, but instead of filling with mean of the cohort we fill with first value
                    new_naive = torch.cat(
                        [
                            self.create_one_tensor(new_naive.shape[1]).repeat(
                                len(new_naive), 1
                            ),
                            self.create_one_tensor(new_naive.shape[1]).repeat(
                                len(new_naive), 1
                            ),
                            torch.cat(
                                [
                                    torch.cat(
                                        [
                                            self.create_one_tensor(
                                                new_naive.shape[1]
                                            ).reshape(1, -1),
                                            new_naive[:index],
                                            new_naive[index].repeat(
                                                len(new_naive) - index - 1, 1
                                            ),
                                        ],
                                        dim=0,
                                    )
                                    for index in range(len(new_naive) - 1)
                                ]
                            ),
                        ]
                    )
                    naive_all.append(new_naive)

            patient_specific_baseline.append(torch.cat(naive_all))
        self.patient_specific_baseline_x = torch.cat(patient_specific_baseline, dim=1)
        self.cat_baseline_x = self.body.decode(
            self.patient_specific_baseline_x, self.splits_x0, self.names_x0
        )

    def get_patient_specific_baseline_y(self):
        patient_specific_baseline_y = []
        list_y = np.concatenate(([0], np.cumsum(self.splits_y0)))

        for index, elem in enumerate(list_y[:-1]):
            naive_all = []
            if self.kinds_y0[index] == "continuous":
                print(index)
                for pat in range(len(self.data_y_splitted)):
                    new_naive = self.data_y_splitted[pat][
                        :, elem : list_y[index + 1]
                    ].clone()
                    mean_cohort = torch.mean(
                        self.data_y[self.non_missing_y[:, index] > 0, index]
                    )
                    mask = (
                        self.non_missing_y_splitted[pat][:, elem : list_y[index + 1]]
                        > 0
                    )
                    new_naive = self.fill_tensor(new_naive, mask)
                    new_naive = torch.cat(
                        [
                            torch.full(new_naive.shape, mean_cohort),
                            torch.full(new_naive.shape, mean_cohort),
                            torch.cat(
                                [
                                    torch.cat(
                                        [
                                            torch.tensor([[mean_cohort]]),
                                            new_naive[:index],
                                            new_naive[index].repeat(
                                                len(new_naive) - index - 1, 1
                                            ),
                                        ],
                                        dim=0,
                                    )
                                    for index in range(len(new_naive) - 1)
                                ]
                            ),
                        ]
                    )
                    naive_all.append(new_naive)
            else:
                for pat in range(len(self.data_y_splitted)):
                    new_naive = self.data_y_splitted[pat][
                        :, elem : list_y[index + 1]
                    ].clone()
                    mean_cohort = torch.mean(
                        self.data_y[self.non_missing_y[:, index] > 0, index]
                    )
                    mask = (
                        (
                            self.non_missing_y_splitted[pat][
                                :, elem : list_y[index + 1]
                            ]
                            > 0
                        )
                        .any(dim=1)
                        .reshape(-1, 1)
                        .repeat(new_naive.shape)
                    )
                    new_naive = self.fill_tensor(new_naive, mask, cat=True)
                    new_naive = torch.cat(
                        [
                            self.create_random_tensor(new_naive.shape[1]).repeat(
                                len(new_naive), 1
                            ),
                            self.create_random_tensor(new_naive.shape[1]).repeat(
                                len(new_naive), 1
                            ),
                            torch.cat(
                                [
                                    torch.cat(
                                        [
                                            self.create_random_tensor(
                                                new_naive.shape[1]
                                            ).reshape(1, -1),
                                            new_naive[:index],
                                            new_naive[index].repeat(
                                                len(new_naive) - index - 1, 1
                                            ),
                                        ],
                                        dim=0,
                                    )
                                    for index in range(len(new_naive) - 1)
                                ]
                            ),
                        ]
                    )
                    naive_all.append(new_naive)

            patient_specific_baseline_y.append(torch.cat(naive_all))
        self.patient_specific_baseline_y = torch.cat(patient_specific_baseline_y, dim=1)
        self.cat_baseline_y, _ = self.body.decode(
            self.patient_specific_baseline_y, self.splits_y0, self.names_y0
        )

    def get_result_df_x(
        self,
        time_flags=[
            (0, 0),
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 12),
        ],
        verbose=True,
    ):
        dfs_cont = {
            name: pd.DataFrame(
                columns=["count", "mae", "mae_naive", "mae_pat_spec", "cov"],
                index=time_flags,
            )
            for i, name in enumerate(self.names_x0)
            if self.kinds_x0[i] == "continuous"
        }
        dfs_cat = {
            name: pd.DataFrame(
                columns=["acc", "naive acc", "pat_spec"], index=time_flags
            )
            for i, name in enumerate(self.names_x0)
            if self.kinds_x0[i] != "continuous"
        }
        dfs_cat_f1 = {
            name: pd.DataFrame(columns=["f1", "naive f1", "pat_spec"], index=time_flags)
            for i, name in enumerate(self.names_x0)
            if self.kinds_x0[i] != "continuous"
        }
        dfs_cont_scaled = {
            name: pd.DataFrame(
                columns=["count", "mae", "mae_naive", "mae_pat_spec", "cov"],
                index=time_flags,
            )
            for i, name in enumerate(self.names_x0)
            if self.kinds_x0[i] == "continuous"
        }
        for j, interv in enumerate(time_flags):
            if interv[0] == interv[1]:
                to_keep = torch.tensor(
                    (self.delta_t_resc == interv[0]) & (self.num_rec_for_pred > 0)
                ).flatten()
            else:
                to_keep = torch.tensor(
                    (self.delta_t_resc > interv[0])
                    & (self.delta_t_resc <= interv[1])
                    & (self.num_rec_for_pred > 0)
                ).flatten()
            list_ = np.concatenate(([0], np.cumsum(self.splits_x0)))
            for index, elem in enumerate(list_[:-1]):
                name = self.names_x0[index]
                if verbose:
                    print(name)
                if self.kinds_x0[index] == "continuous":
                    data_sc = self.data_x_recon[
                        (
                            self.non_missing_x_recon[:, elem : list_[index + 1]] > 0
                        ).flatten(),
                        elem : list_[index + 1],
                    ]
                    all_targets = self.body.get_var_by_name(name).decode(data_sc)
                    mask_ = (
                        self.non_missing_x_recon[to_keep, elem : list_[index + 1]] > 0
                    )
                    recon = self.res_list[index][0][to_keep][mask_].detach()
                    true = self.data_x_recon[to_keep, elem : list_[index + 1]][
                        mask_
                    ].detach()
                    log_var = self.predictions.recon_x_log_var[
                        to_keep, elem : list_[index + 1]
                    ][mask_]
                    var = torch.exp(log_var)

                    pat_baseline_ff = self.patient_specific_baseline_x[
                        to_keep, elem : list_[index + 1]
                    ][mask_].detach()

                    if len(recon) > 0:
                        recon_resc = self.body.get_var_by_name(name).decode(
                            recon.reshape(-1, 1)
                        )
                        true_resc = self.body.get_var_by_name(name).decode(
                            true.reshape(-1, 1)
                        )
                        patient_specific_baseline_ff_resc = self.body.get_var_by_name(
                            name
                        ).decode(pat_baseline_ff.reshape(-1, 1))
                        cov = self.count_within_range(true, recon, var) / len(log_var)

                        mse = sum((recon_resc - true_resc) ** 2) / len(recon)
                        mae = sum(abs(recon_resc - true_resc)) / len(recon)
                        mae_naive = sum(abs(np.mean(all_targets) - true_resc)) / len(
                            true_resc
                        )
                        mae_pat_spec_ff = sum(
                            abs(patient_specific_baseline_ff_resc - true_resc)
                        ) / len(true_resc)
                        mse_sc = sum((recon - true) ** 2) / len(recon)
                        mae_sc = sum(abs(recon - true)) / len(recon)
                        mae_naive_sc = sum(
                            abs(np.mean(np.array(data_sc)) - true)
                        ) / len(true)
                        mae_pat_spec_ff_sc = sum(abs(pat_baseline_ff - true)) / len(
                            true
                        )

                        dfs_cont[self.names_x0[index]].iloc[j]["count"] = len(recon)
                        dfs_cont[self.names_x0[index]].iloc[j]["mae"] = mae.item()
                        dfs_cont[self.names_x0[index]].iloc[j]["cov"] = cov

                        dfs_cont[self.names_x0[index]].iloc[j][
                            "mae_naive"
                        ] = mae_naive.item()
                        dfs_cont[self.names_x0[index]].iloc[j][
                            "mae_pat_spec"
                        ] = mae_pat_spec_ff.item()
                        dfs_cont_scaled[self.names_x0[index]].iloc[j]["count"] = len(
                            recon
                        )
                        dfs_cont_scaled[self.names_x0[index]].iloc[j][
                            "mae"
                        ] = mae_sc.item()
                        dfs_cont_scaled[self.names_x0[index]].iloc[j][
                            "mae_naive"
                        ] = mae_naive_sc.item()
                        dfs_cont_scaled[self.names_x0[index]].iloc[j][
                            "mae_pat_spec"
                        ] = mae_pat_spec_ff_sc.item()
                        dfs_cont_scaled[self.names_x0[index]].iloc[j]["cov"] = cov

                    else:
                        dfs_cont[self.names_x0[index]].iloc[j]["count"] = 0
                        dfs_cont[self.names_x0[index]].iloc[j]["mae"] = np.nan
                        dfs_cont[self.names_x0[index]].iloc[j]["cov"] = np.nan

                        dfs_cont[self.names_x0[index]].iloc[j]["mae_naive"] = np.nan
                        dfs_cont[self.names_x0[index]].iloc[j]["mae_pat_spec"] = np.nan

                elif self.kinds_x0[index] != "continuous":
                    all_targets = self.ground_truth_x[1][index][
                        (self.non_missing_x_recon[:, elem : list_[index + 1]] > 0).any(
                            dim=1
                        )
                    ]
                    mask_ = (
                        self.non_missing_x_recon[to_keep, elem : list_[index + 1]] > 0
                    ).any(dim=1)
                    recon = self.res_list[index][0][to_keep]
                    true = self.ground_truth_x[1][index][to_keep][mask_]
                    pat_spec_ff = self.cat_baseline_x[1][index][to_keep][mask_]

                    recon = self.body.get_var_by_name(
                        self.names_x0[index]
                    ).get_categories(recon)[mask_]
                    if len(recon) > 0:
                        acc = accuracy_score(true.flatten().astype(float), recon)
                        value, counts = np.unique(
                            all_targets.flatten(), return_counts=True
                        )
                        naive = np.random.choice(
                            value, size=len(true.flatten()), p=counts / sum(counts)
                        )
                        naive_acc = accuracy_score(
                            naive.astype(float), true.flatten().astype(float)
                        )
                        pat_spec_acc_ff = accuracy_score(
                            pat_spec_ff, true.flatten().astype(float)
                        )
                        f1 = f1_score(
                            true.flatten().astype(float), recon, average="macro"
                        )
                        naive_f1 = f1_score(
                            naive.astype(float),
                            true.flatten().astype(float),
                            average="macro",
                        )
                        pat_spec_f1_ff = f1_score(
                            pat_spec_ff, true.flatten().astype(float), average="macro"
                        )
                        if verbose:
                            print(f"acc {acc}")
                            print(f"acc naive {naive_acc}")
                        # print(f'{classification_report(true.flatten().astype(float), recon)}')
                        dfs_cat[self.names_x0[index]].iloc[j]["acc"] = acc
                        dfs_cat[self.names_x0[index]].iloc[j]["naive acc"] = naive_acc
                        dfs_cat[self.names_x0[index]].iloc[j][
                            "pat_spec"
                        ] = pat_spec_acc_ff
                        dfs_cat_f1[self.names_x0[index]].iloc[j]["f1"] = f1
                        dfs_cat_f1[self.names_x0[index]].iloc[j]["naive f1"] = naive_f1
                        dfs_cat_f1[self.names_x0[index]].iloc[j][
                            "pat_spec"
                        ] = pat_spec_f1_ff
                    else:
                        dfs_cat[self.names_x0[index]].iloc[j]["acc"] = np.nan
                        dfs_cat[self.names_x0[index]].iloc[j]["naive acc"] = np.nan
                        dfs_cat[self.names_x0[index]].iloc[j]["pat_spec"] = np.nan
                        dfs_cat_f1[self.names_x0[index]].iloc[j]["f1"] = np.nan
                        dfs_cat_f1[self.names_x0[index]].iloc[j]["naive f1"] = np.nan
                        dfs_cat_f1[self.names_x0[index]].iloc[j]["pat_spec"] = np.nan

        self.df_res_cont = pd.DataFrame(
            np.nanmean(
                np.array([elem for elem in dfs_cont_scaled.values()], dtype=np.float64),
                axis=0,
            ),
            index=dfs_cont_scaled["Forced Vital Capacity (FVC - % predicted)"].index,
            columns=dfs_cont_scaled[
                "Forced Vital Capacity (FVC - % predicted)"
            ].columns,
        )
        self.df_res_cat_acc = pd.DataFrame(
            np.nanmean(
                np.array([elem for elem in dfs_cat.values()], dtype=np.float64), axis=0
            ),
            index=time_flags,
            columns=["acc", "naive acc", "pat_spec"],
        )
        self.df_res_cat_f1 = pd.DataFrame(
            np.nanmean(
                np.array([elem for elem in dfs_cat_f1.values()], dtype=np.float64),
                axis=0,
            ),
            index=time_flags,
            columns=["f1", "naive f1", "pat_spec"],
        )
        # plot
        fig, ax = plt.subplots()
        x_axis = range(len(self.df_res_cont))
        ax.plot(x_axis, self.df_res_cont.mae, label="Model")

        ax.plot(
            x_axis,
            self.df_res_cont.mae_pat_spec,
            "-.",
            label="previous value for patient",
        )
        ax.plot(x_axis, self.df_res_cont.mae_naive, "-.", label="cohort mean")
        ax.set_title("Average accross all (scaled) MAE")
        ax.set_ylabel("MAE")
        ax.set_xlabel("years")
        ax.legend()

        fig, ax = plt.subplots()
        # x_axis = range(-len(tmp)+1, 1)
        x_axis = range(len(self.df_res_cat_acc))
        ax.plot(x_axis, self.df_res_cat_acc.acc, label="Model")

        ax.plot(
            x_axis,
            self.df_res_cat_acc.pat_spec,
            "-.",
            label="previous value for patient",
        )
        ax.plot(x_axis, self.df_res_cat_acc["naive acc"], "-.", label="cohort baseline")
        ax.set_title("Average accross all (scaled) accuracy")
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("years")
        ax.legend()

        fig, ax = plt.subplots()
        # x_axis = range(-len(tmp)+1, 1)
        x_axis = range(len(self.df_res_cat_f1))
        ax.plot(x_axis, self.df_res_cat_f1.f1, label="Model")

        ax.plot(
            x_axis,
            self.df_res_cat_f1.pat_spec,
            "-.",
            label="previous value for patient",
        )
        ax.plot(x_axis, self.df_res_cat_f1["naive f1"], "-.", label="cohort baseline")
        ax.set_title("Average accross all (scaled) f1")
        ax.set_ylabel("f1")
        ax.set_xlabel("years")
        ax.legend()

    def get_result_df_y(
        self,
        time_flags=[
            (0, 0),
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 12),
        ],
        verbose=True,
    ):
        list_ = np.cumsum([0] + self.splits_y0)
        df = {
            name: pd.DataFrame(
                index=time_flags,
                columns=[
                    "acc",
                    "f1 macro",
                    "f1 weighted",
                    "acc_base",
                    "f1_macro_base",
                    "f1_weighted_base",
                    "acc_naive",
                    "f1_macro_naive",
                    "f1_weighted_naive",
                ],
            )
            for name in self.names_y0
        }
        for j, interv in enumerate(time_flags):
            if interv[0] == interv[1]:
                to_keep = torch.tensor(
                    (self.delta_t_resc == interv[0]) & (self.num_rec_for_pred >= 0)
                ).flatten()
            else:
                to_keep = torch.tensor(
                    (self.delta_t_resc > interv[0])
                    & (self.delta_t_resc <= interv[1])
                    & (self.num_rec_for_pred >= 0)
                ).flatten()
            for i, elem in enumerate(list_[:-1]):
                mask_ = (
                    self.non_missing_y_recon[to_keep, list_[i] : list_[i + 1]] > 0
                ).any(dim=1)
                true_recon = self.ground_truth_y[0][to_keep, i][mask_]
                baseline_ = self.cat_baseline_y[to_keep, i][mask_]
                model_recon = self.predicted_cats_y[to_keep, i][mask_]
                value, counts = np.unique(true_recon.flatten(), return_counts=True)
                naive = np.random.choice(
                    value, size=len(true_recon.flatten()), p=counts / sum(counts)
                )
                name = self.names_y0[i]
                if verbose:
                    print(name)
                    print(classification_report(true_recon, model_recon))
                    print(classification_report(true_recon, baseline_))
                    print(confusion_matrix(true_recon, model_recon))
                df[name].iloc[j]["acc"] = accuracy_score(true_recon, model_recon)
                df[name].iloc[j]["f1 macro"] = f1_score(
                    true_recon, model_recon, average="macro"
                )
                df[name].iloc[j]["f1 weighted"] = f1_score(
                    true_recon, model_recon, average="weighted"
                )
                df[name].iloc[j]["acc_base"] = accuracy_score(true_recon, baseline_)
                df[name].iloc[j]["f1_macro_base"] = f1_score(
                    true_recon, baseline_, average="macro"
                )
                df[name].iloc[j]["f1_weighted_base"] = f1_score(
                    true_recon, baseline_, average="weighted"
                )
                df[name].iloc[j]["acc_naive"] = accuracy_score(true_recon, naive)
                df[name].iloc[j]["f1_macro_naive"] = f1_score(
                    true_recon, naive, average="macro"
                )
                df[name].iloc[j]["f1_weighted_naive"] = f1_score(
                    true_recon, naive, average="weighted"
                )

        self.df_res_y = df
        # plot
        for index in range(len(self.names_y0)):
            name = self.names_y0[index]
            tmp = self.df_res_y[name]
            fig, ax = plt.subplots()
            ax.plot(range(len(tmp)), tmp["f1 macro"], label="ours")
            ax.plot(
                range(len(tmp)),
                tmp["f1_macro_base"],
                "-.",
                label="previous value for patient",
            )
            ax.plot(
                range(len(tmp)),
                tmp["f1_macro_naive"],
                "-.",
                label="cohort baseline",
            )
            ax.set_title(name)
            ax.set_ylabel("Macro F1")
            ax.set_xlabel("years")
            ax.legend()
        for index in range(len(self.names_y0)):
            name = self.names_y0[index]
            tmp = self.df_res_y[name]
            fig, ax = plt.subplots()
            ax.plot(range(len(tmp)), tmp["f1 weighted"], label="ours")
            ax.plot(
                range(len(tmp)),
                tmp["f1_weighted_base"],
                "-.",
                label="previous value for patient",
            )
            ax.plot(
                range(len(tmp)),
                tmp["f1_weighted_naive"],
                "-.",
                label="cohort baseline",
            )
            ax.set_title(name)
            ax.set_ylabel("Weighted F1")
            ax.set_xlabel("years")
            ax.legend()
        for index in range(len(self.names_y0)):
            name = self.names_y0[index]
            tmp = self.df_res_y[name]
            fig, ax = plt.subplots()
            ax.plot(range(len(tmp)), tmp.acc, label="ours")
            ax.plot(
                range(len(tmp)), tmp.acc_base, "-.", label="previous value for patient"
            )
            ax.plot(range(len(tmp)), tmp.acc_naive, "-.", label="cohort baseline")
            ax.set_title(name)
            ax.set_ylabel("Accuracy")
            ax.set_xlabel("years")
            ax.legend()
        for index in range(len(self.names_y0)):
            name = self.names_y0[index]
            tmp = self.df_res_y[name]
            fig, ax = plt.subplots()
            ax.plot(range(len(tmp)), tmp.acc, label="ours")
            ax.plot(
                range(len(tmp)), tmp.acc_base, "-.", label="previous value for patient"
            )
            ax.plot(range(len(tmp)), tmp.acc_naive, "-.", label="cohort baseline")
            ax.set_title(name)
            ax.set_ylabel("Accuracy")
            ax.set_xlabel("years")
            ax.legend()

    def fill_tensor(self, data, mask, cat=False):
        # fill data
        filled_data = torch.zeros_like(data)  # Initialize the filled tensor with zeros
        if cat:
            # intialize first row with some category
            filled_data[0] = self.create_one_tensor(data.shape[1])
        else:
            filled_data[0] = data[0]  # Copy the first row of data as it is

        # Iterate over columns and rows starting from the second row
        for i in range(1, data.size(0)):
            # fill with previous value if value is not available
            filled_data[i] = torch.where(mask[i] == 1, data[i], filled_data[i - 1])

        return filled_data

    def create_one_tensor(self, n):
        tensor = torch.zeros(n)  # Initialize tensor with zeros
        tensor[0] = 1  # Set the value index to 1
        return tensor

    def create_random_tensor(self, n):
        tensor = torch.zeros(n)  # Initialize tensor with zeros
        random_index = random.randint(
            0, n - 1
        )  # Generate a random index within the range of n
        tensor[random_index] = 1  # Set the value at the random index to 1
        return tensor

    def count_within_range(self, x_true, x_pred, x_var_pred):
        lower_bound = x_pred - 2 * torch.sqrt(x_var_pred)
        upper_bound = x_pred + 2 * torch.sqrt(x_var_pred)
        within_range = (x_true >= lower_bound) & (x_true <= upper_bound)
        count = torch.sum(within_range).item()

        return count

    def plot_prior_preds(self):
        recon_x_grouped = torch.split(
            self.predictions.recon_x, [elem * (elem + 1) for elem in self.splits]
        )
        recon_x_grouped = [
            torch.split(elem, [self.splits[index]] * (self.splits[index] + 1))
            for index, elem in enumerate(recon_x_grouped)
        ]
        self.prior_preds = torch.split(
            torch.cat([recon_x_grouped[i][0] for i in range(len(recon_x_grouped))]),
            self.splits,
        )
        #         non_missing_s = 1- inputs["missing_s"] * 1.0
        #         non_missing_s_splitted = torch.split(non_missing_s, splits, dim=0)
        list_x = np.concatenate(([0], np.cumsum(self.splits_x0)))
        # names_s0 = [vN for i,vN in enumerate(var_names0) if xyt0[i]=='s']
        # plot
        combinations = list(set(map(tuple, np.array(self.data_s[:, 2:7]))))
        num_combinations = len(combinations)
        colors = cm.tab20(np.linspace(0, 1, num_combinations))
        races = [
            "Race white",
            "Hispanic",
            "Any other white",
            "Race asian",
            "Race black",
        ]
        category_names = [
            [
                elem + " "
                for index, elem in enumerate(races)
                if combinations[j][index] == 1
            ]
            for j in range(num_combinations)
        ]
        category_names = {
            combinations[i]: category_names[i] for i in range(num_combinations)
        }
        array_ = np.array(self.data_s[:, 2:7])
        colors = {combinations[i]: colors[i] for i in range(num_combinations)}
        # random subset
        indices_to_plot = random.sample(range(len(self.splits)), 100)
        # plots for race
        for i_x, index_x in enumerate(list_x[:-1]):
            name_x = self.names_x0[i_x]
            if self.kinds_x0[i_x] == "continuous":
                fig, ax = plt.subplots()
                for pat in indices_to_plot:
                    ax.plot(
                        self.body.get_var_by_name("time [years]").decode(
                            self.times_splitted[pat]
                        ),
                        self.body.get_var_by_name(name_x).decode(
                            self.prior_preds[pat][:, index_x].detach().reshape(-1, 1)
                        ),
                        color=colors[tuple(array_[pat])],
                        label=category_names[tuple(array_[pat])],
                    )

                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys())
                plt.title(name_x)
                ax.set_xlabel("time [years]")
                ax.set_ylabel("Value")
                ax.grid(linestyle="--")

        # plots for other static
        for i_x, index_x in enumerate(list_x[:-1]):
            name_x = self.names_x0[i_x]
            if self.kinds_x0[i_x] == "continuous":
                for index_s, name_s in enumerate(self.names_s0):
                    if name_s not in races:

                        fig, ax = plt.subplots()
                        if name_s in [
                            "Date of birth",
                            "Onset of first non-Raynaud?s of the disease",
                        ]:

                            c_array = np.array(
                                self.body.get_var_by_name(name_s)
                                .decode(self.data_s[:, index_s].reshape(-1, 1))
                                .flatten()
                                .astype("datetime64[ns]")
                                .astype("datetime64[Y]")
                                .astype(int)
                                + 1970,
                                dtype=np.float32,
                            )
                        else:
                            c_array = np.array(
                                self.body.get_var_by_name(name_s)
                                .decode(self.data_s[:, index_s].reshape(-1, 1))
                                .flatten(),
                                dtype=np.float32,
                            )

                        cmap = plt.cm.get_cmap("viridis")
                        norm = plt.Normalize(
                            vmin=np.nanmin(c_array), vmax=np.nanmax(c_array)
                        )

                        for pat in indices_to_plot:
                            if name_s in [
                                "Date of birth",
                                "Onset of first non-Raynaud?s of the disease",
                            ]:
                                c_array_pat = np.array(
                                    self.body.get_var_by_name(name_s)
                                    .decode(
                                        self.data_s_splitted[pat][0, index_s].reshape(
                                            -1, 1
                                        )
                                    )
                                    .flatten()
                                    .astype("datetime64[ns]")
                                    .astype("datetime64[Y]")
                                    .astype(int)
                                    + 1970,
                                    dtype=np.float32,
                                )
                            else:
                                c_array_pat = np.array(
                                    self.body.get_var_by_name(name_s)
                                    .decode(
                                        self.data_s_splitted[pat][0, index_s].reshape(
                                            -1, 1
                                        )
                                    )
                                    .flatten(),
                                    dtype=np.float32,
                                )

                            if self.non_missing_s_splitted[pat][0][index_s] > 0:
                                line = ax.plot(
                                    self.body.get_var_by_name("time [years]").decode(
                                        self.times_splitted[pat]
                                    ),
                                    self.body.get_var_by_name(name_x).decode(
                                        self.prior_preds[pat][:, index_x]
                                        .detach()
                                        .reshape(-1, 1)
                                    ),
                                    c=cmap(norm(c_array_pat)),
                                )
                        plt.title(name_x)
                        ax.set_xlabel("time [years]")
                        ax.set_ylabel("Value")
                        ax.grid(linestyle="--")

                        # Create a ScalarMappable using the colormap and norm
                        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                        sm.set_array([])  # Set an empty array

                        # Create a colorbar
                        cbar = plt.colorbar(
                            sm,
                            ax=ax,
                        )
                        cbar.ax.set_ylabel(name_s)
        return

    def plot_prior_preds_y(self):
        preds_y_grouped = torch.split(
            self.probs_matrix_y, [elem * (elem + 1) for elem in self.splits]
        )
        preds_y_grouped = [
            torch.split(elem, [self.splits[index]] * (self.splits[index] + 1))
            for index, elem in enumerate(preds_y_grouped)
        ]
        self.prior_preds_y = torch.split(
            torch.cat([preds_y_grouped[i][0] for i in range(len(preds_y_grouped))]),
            self.splits,
        )
        list_y = np.concatenate(([0], np.cumsum(self.splits_y0)))
        combinations = list(set(map(tuple, np.array(self.data_s[:, 2:7]))))
        num_combinations = len(combinations)
        colors = cm.tab20(np.linspace(0, 1, num_combinations))
        races = [
            "Race white",
            "Hispanic",
            "Any other white",
            "Race asian",
            "Race black",
        ]
        category_names = [
            [
                elem + " "
                for index, elem in enumerate(races)
                if combinations[j][index] == 1
            ]
            for j in range(num_combinations)
        ]
        category_names = {
            combinations[i]: category_names[i] for i in range(num_combinations)
        }
        array_ = np.array(self.data_s[:, 2:7])
        colors = {combinations[i]: colors[i] for i in range(num_combinations)}
        # random subset
        indices_to_plot = random.sample(range(len(self.splits)), 100)
        # plots for race
        for i_y, index_y in enumerate(list_y[:-1]):
            name_y = self.names_y0[i_y]
            if name_y in [
                "LUNG_ILD_involvement_or",
                "HEART_involvement_or",
                "ARTHRITIS_involvement_or",
            ]:
                fig, ax = plt.subplots()
                for pat in indices_to_plot:
                    ax.plot(
                        self.body.get_var_by_name("time [years]").decode(
                            self.times_splitted[pat]
                        ),
                        self.prior_preds_y[pat][:, index_y].detach(),
                        color=colors[tuple(array_[pat])],
                        label=category_names[tuple(array_[pat])],
                    )
                ax.set(ylim=(0, 1))
                ax.set_ylabel("Probability")
                ax.set_xlabel("time [years]")
                ax.grid(linestyle="--")

                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys())
                plt.title(name_y)

        for i_y, index_y in enumerate(list_y[:-1]):
            name_y = self.names_y0[i_y]
            if name_y in [
                "LUNG_ILD_involvement_or",
                "HEART_involvement_or",
                "ARTHRITIS_involvement_or",
            ]:
                for index_s, name_s in enumerate(self.names_s0):
                    if name_s not in races:

                        fig, ax = plt.subplots()
                        if name_s in [
                            "Date of birth",
                            "Onset of first non-Raynaud?s of the disease",
                        ]:

                            c_array = np.array(
                                self.body.get_var_by_name(name_s)
                                .decode(self.data_s[:, index_s].reshape(-1, 1))
                                .flatten()
                                .astype("datetime64[ns]")
                                .astype("datetime64[Y]")
                                .astype(int)
                                + 1970,
                                dtype=np.float32,
                            )
                        else:
                            c_array = np.array(
                                self.body.get_var_by_name(name_s)
                                .decode(self.data_s[:, index_s].reshape(-1, 1))
                                .flatten(),
                                dtype=np.float32,
                            )

                        cmap = plt.cm.get_cmap("viridis")
                        norm = plt.Normalize(
                            vmin=np.nanmin(c_array), vmax=np.nanmax(c_array)
                        )

                        for pat in indices_to_plot:
                            if name_s in [
                                "Date of birth",
                                "Onset of first non-Raynaud?s of the disease",
                            ]:
                                c_array_pat = np.array(
                                    self.body.get_var_by_name(name_s)
                                    .decode(
                                        self.data_s_splitted[pat][0, index_s].reshape(
                                            -1, 1
                                        )
                                    )
                                    .flatten()
                                    .astype("datetime64[ns]")
                                    .astype("datetime64[Y]")
                                    .astype(int)
                                    + 1970,
                                    dtype=np.float32,
                                )
                            else:
                                c_array_pat = np.array(
                                    self.body.get_var_by_name(name_s)
                                    .decode(
                                        self.data_s_splitted[pat][0, index_s].reshape(
                                            -1, 1
                                        )
                                    )
                                    .flatten(),
                                    dtype=np.float32,
                                )

                            if self.non_missing_s_splitted[pat][0][index_s] > 0:
                                line = ax.plot(
                                    self.body.get_var_by_name("time [years]").decode(
                                        self.times_splitted[pat]
                                    ),
                                    self.prior_preds_y[pat][:, index_y].detach(),
                                    c=cmap(norm(c_array_pat)),
                                )
                        ax.set(ylim=(0, 1))
                        ax.set_ylabel("Probability")
                        ax.set_xlabel("time [years]")
                        ax.grid(linestyle="--")

                        plt.title(name_y)
                        # Create a ScalarMappable using the colormap and norm
                        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                        sm.set_array([])  # Set an empty array

                        # Create a colorbar
                        cbar = plt.colorbar(
                            sm,
                            ax=ax,
                        )
                        cbar.ax.set_ylabel(name_s)
        return
