import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from torch.distributions.categorical import Categorical


def plot_z_space(z, y, I=0, J=1, figsize=(6, 6)):
    # z = z.detach().numpy()
    # y = y.detach().numpy()

    plt.figure(figsize=figsize)

    zI = z[:, I]
    zJ = z[:, J]

    plt.scatter(zI, zJ, c=y, s=15)

    plt.colorbar()

    return plt


def plot_losses(pipeline):
    for suff in ["", "_pred"]:
        fig, axs = plt.subplots(
            nrows=3 + len(pipeline.model.classifiers), ncols=1, figsize=(8, 16)
        )
        if suff == "":
            names = ["loss", "loss_recon", "loss_kld"] + [
                classif.name for classif in pipeline.model.classifiers
            ]
        else:
            names = ["loss", "loss_recon"] + [
                classif.name for classif in pipeline.model.classifiers
            ]
        for i, name in enumerate(names):
            # axs[i].plot(pipeline.trainer.all_losses_train[:, i], label="train " + name)
            # axs[i].plot(pipeline.trainer.all_losses_valid[:, i], label="valid " + name)
            axs[i].plot(
                getattr(pipeline.trainer, "all_losses_train" + suff)[:, i],
                label="train " + name + suff,
            )
            axs[i].plot(
                getattr(pipeline.trainer, "all_losses_valid" + suff)[:, i],
                label="valid " + name + suff,
            )
            axs[i].set_ylabel(name + suff)
            axs[i].legend()

        plt.tight_layout()

        fig, axs = plt.subplots(
            nrows=3 + len(pipeline.model.classifiers), ncols=1, figsize=(8, 16)
        )
        if suff == "":
            names = ["loss", "loss_recon", "loss_kld"] + [
                classif.name + "_unw" for classif in pipeline.model.classifiers
            ]
        else:
            names = ["loss", "loss_recon"] + [
                classif.name + "_unw" for classif in pipeline.model.classifiers
            ]
        for i, name in enumerate(names):
            # axs[i].plot(pipeline.trainer.all_losses_train_unw[:, i], label="train " + name)
            # axs[i].plot(pipeline.trainer.all_losses_valid_unw[:, i], label="valid " + name)
            axs[i].plot(
                getattr(pipeline.trainer, "all_losses_train_unw" + suff)[:, i],
                label="train " + name + suff,
            )
            axs[i].plot(
                getattr(pipeline.trainer, "all_losses_valid_unw" + suff)[:, i],
                label="valid " + name + suff,
            )
            axs[i].set_ylabel(name + suff)
            axs[i].legend()

        plt.tight_layout()


def plot_recon_losses(pipeline, data_train):
    for suff in ["", "_pred"]:
        fig, axs = plt.subplots(
            nrows=len(pipeline.model.splits_x0), ncols=1, figsize=(8, 60)
        )
        # compute percentage of missing values
        missing_ = (
            torch.cat(data_train.missing_x).sum(dim=0)
            / len(torch.cat(data_train.missing_x))
            * 100
        )
        index_tracker = 0
        for i, x in enumerate(pipeline.model.splits_x0):
            # axs[i].plot(pipeline.trainer.losses_recon_stack_train[:, i], label="train")
            # axs[i].plot(pipeline.trainer.losses_recon_stack_eval[:, i], label="eval")
            axs[i].plot(
                getattr(pipeline.trainer, "losses_recon_stack_train" + suff)[:, i],
                label="train",
            )
            axs[i].plot(
                getattr(pipeline.trainer, "losses_recon_stack_eval" + suff)[:, i],
                label="eval",
            )
            axs[i].set_title(
                pipeline.model.model_config.names_x0[i]
                + suff
                + ", percentage of missingness : "
                + str(np.round(missing_[index_tracker].item(), 2))
            )
            axs[i].legend()
            index_tracker += x


def plot_x(
    data_x,
    recon_x_mean,
    recon_x_std,
    time,
    missing_x,
    names_x=None,
    probs_matrix=None,
    probs_matrix_std=None,
    kinds_x1=None,
    figsize=(6, 3),
):
    recon_x = recon_x_mean.detach().numpy()
    recon_std = recon_x_std.detach().numpy()
    if not probs_matrix is None:
        probs_matrix = probs_matrix.detach().numpy()
    if not probs_matrix_std is None:
        probs_matrix_std = probs_matrix_std.detach().numpy()
    nP = data_x.shape[1]

    # create figure with subplots
    f, axs = plt.subplots(
        nP, 1, sharex=True, sharey=False, figsize=(1 * figsize[0], nP * figsize[1])
    )
    f.subplots_adjust(hspace=0.2)  # , wspace=0.2)

    for i in range(nP):
        if nP > 1:
            ax = axs[i]
        else:
            ax = axs
        ax.plot(time, data_x[:, i], ".-", color="C2", label="data_x")

        if kinds_x1 is None:
            ax.plot(time, recon_x[:, i], ".:", color="C0", label="recon_x")

            if not probs_matrix is None:
                ax.plot(time, probs_matrix[:, i], ".:", color="C1", label="probs")
        else:
            if kinds_x1[i] in ["continuous", "ordinal"]:
                ax.plot(time, recon_x[:, i], ".:", color="C0", label="recon_x")
                ax.fill_between(
                    time,
                    recon_x[:, i] - recon_std[:, i],
                    recon_x[:, i] + recon_std[:, i],
                    alpha=0.5,
                )
            else:
                if not probs_matrix is None:
                    ax.plot(time, probs_matrix[:, i], ".:", color="C1", label="probs")
                if not probs_matrix_std is None:
                    ax.fill_between(
                        time,
                        probs_matrix[:, i] - probs_matrix_std[:, i],
                        probs_matrix[:, i] + probs_matrix_std[:, i],
                        alpha=0.5,
                    )

        non_miss = ~missing_x[:, i]
        ax.plot(
            time[non_miss],
            data_x[non_miss, i],
            "o",
            color="C3",
            label="non-missing_x",
        )

        if not names_x is None:
            ax.set_title(names_x[i])

        ax.set_ylim(-3, 3)
        ax.legend()


def plot_x_overlaid(
    data_x,
    recon_x_mean,
    recon_x_std,
    time,
    missing_x,
    num_rec,
    names_x,
    kinds_x1,
    names,
    figsize=(6, 3),
):
    recon_x_means = [elem.detach().numpy() for elem in recon_x_mean]
    recon_x_stds = [elem.detach().numpy() for elem in recon_x_std]
    # nP = data_x.shape[1]
    cont = [elem for elem in kinds_x1 if elem in ["continuous", "ordinal"]]
    cont_indices = [
        i for i, elem in enumerate(kinds_x1) if elem in ["continuous", "ordinal"]
    ]
    # nP = len(cont)
    nP = len(names)
    # create figure with subplots
    f, axs = plt.subplots(
        nP, 1, sharex=True, sharey=False, figsize=(1 * figsize[0], nP * figsize[1])
    )
    f.subplots_adjust(hspace=0.2)  # , wspace=0.2)
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(recon_x_means)))

    for j, i in enumerate(cont_indices):

        if names_x[i] in names:
            if nP > 1:
                ax = axs[j]
            else:
                ax = axs
            ax.plot(time, data_x[:, i], ".-", color="C2", label="data_x")

            for index in range(len(recon_x_means)):
                ax.plot(
                    time,
                    recon_x_means[index][:, i],
                    ".:",
                    color="black",
                    label="predictions",
                )
                ax.fill_between(
                    time,
                    recon_x_means[index][:, i] - 2 * recon_x_stds[index][:, i],
                    recon_x_means[index][:, i] + 2 * recon_x_stds[index][:, i],
                    alpha=0.5,
                    color=colors[index],
                )

            non_miss = ~missing_x[:, i]
            ax.plot(
                time[non_miss],
                data_x[non_miss, i],
                "o",
                color="C3",
                label="non-missing_x",
            )

            if not names_x is None:
                ax.set_title(names_x[i])
            if num_rec < len(time):
                ax.axvline(time[num_rec] - 0.01, ls="--")
            ax.set_ylim(-3, 3)
            ax.legend(loc="upper right")

    return f


def plot_all_categorical_preds(
    name,
    body,
    data_x,
    non_missing_x,
    splits_x0,
    names_x0,
    num_for_rec_flat,
    res_list_all,
    times,
    samples,
):
    categories = body.get_var_by_name(name).enc.categories_[0]
    index = names_x0.index(name)
    cat_samples = []
    for num_pred in range(len(num_for_rec_flat)):
        cat_samples.append(
            torch.stack(
                [
                    Categorical(res_list_all[num_pred][samp][index][1]).sample()
                    if res_list_all[num_pred][samp][index][1].size()[1] > 1
                    else Categorical(
                        torch.cat(
                            [
                                res_list_all[num_pred][samp][index][1],
                                1 - res_list_all[num_pred][samp][index][1],
                            ],
                            dim=1,
                        )
                    ).sample()
                    for samp in range(len(res_list_all[num_pred]))
                ]
            )
        )
    for index_pred, num_rec in enumerate(num_for_rec_flat):
        predicted_probas = torch.empty((len(categories), len(times)))
        for i, cat in enumerate(categories):
            for t in range(len(times)):
                predicted_probas[i, t] = len(
                    [elem for elem in cat_samples[index_pred][:, t] if elem == i]
                ) / len(samples)
        ground_truth = body.decode(data_x, splits_x0, names_x0)[1][index]
        non_miss = torch.split(non_missing_x, splits_x0, dim=1)[index][:, 0]
        num_pred = len(num_for_rec_flat) - index_pred
        plot_categorical_preds(
            times, categories, predicted_probas, ground_truth, non_miss, num_pred, name
        )
    return


def plot_categorical_preds(
    times, categories, predicted_probas, ground_truth, non_miss, num_pred, name
):
    # Generate example data
    time_points = [i.item() for i in times.flatten()]
    categories = list(categories)
    ground_truth = ground_truth.flatten()
    predicted_probs = np.array(predicted_probas).T

    # Create DataFrame
    data = pd.DataFrame(
        {
            "time": time_points,
            "category": ground_truth,
            **{f"prob_{c}": predicted_probs[:, i] for i, c in enumerate(categories)},
        }
    )

    # Create line plot of ground truth category
    fig, ax = plt.subplots()
    ax.plot(data["time"], data["category"], "-o", color="black", label="ground truth")
    # non missing
    ax.plot(
        data["time"].values[non_miss.bool()],
        data["category"].values[non_miss.bool()],
        "o",
        color="red",
        label="non missing",
    )
    # predicted points
    if num_pred > 0:
        ax.axvline(time_points[-num_pred] - 0.04, ls="--")
    # Create heatmap of predicted probabilities
    heatmap_data = data.drop(columns=["time", "category"])
    heatmap_data = heatmap_data.set_index(pd.Index(time_points, name="time"))

    # Create meshgrid for heatmap
    X, Y = np.meshgrid(heatmap_data.index, np.array(categories))

    # Compute the width and height of each rectangle
    dx = np.diff(
        heatmap_data.index + np.concatenate(([0], np.diff(heatmap_data.index) / 2))
    )
    dy = np.ones(len(categories)) * 0.5

    # Create the heatmap
    cmap = plt.cm.get_cmap("Blues")
    im = ax.pcolormesh(X, Y, heatmap_data.values.T, cmap=cmap, vmin=0.0, vmax=1.0)

    # Add colorbar and labels
    cbar = fig.colorbar(im)
    cbar.set_label("Probability")
    ax.set_xlabel("Time")
    ax.set_ylabel("Category")
    plt.legend(loc="upper right")
    plt.yticks(categories)
    plt.title(f"{name}: Predicting last {num_pred} data points")

    # plt.show()
    return fig


def plot_x_overlaid_sep(
    data_x,
    recon_x_mean,
    recon_x_std,
    time,
    missing_x,
    num_to_pred,
    names_x,
    kinds_x1,
    num_pred=3,
    figsize=(6, 3),
):
    recon_x_means = [elem.detach().numpy() for elem in recon_x_mean]
    recon_x_stds = [elem.detach().numpy() for elem in recon_x_std]
    # nP = data_x.shape[1]
    cont = [elem for elem in kinds_x1 if elem in ["continuous", "ordinal"]]
    cont_indices = [
        i for i, elem in enumerate(kinds_x1) if elem in ["continuous", "ordinal"]
    ]
    nP = len(cont) * num_pred
    # create figure with subplots
    f, axs = plt.subplots(
        nP, 1, sharex=True, sharey=False, figsize=(1 * figsize[0], nP * figsize[1])
    )
    f.subplots_adjust(hspace=0.2)  # , wspace=0.2)
    colors = plt.cm.Blues(np.linspace(0.2, 1, len(recon_x_means)))

    index_count = 0
    for j, i in enumerate(cont_indices):

        for index in range(len(recon_x_means)):
            ax = axs[index_count]
            ax.plot(time, data_x[:, i], ".-", color="C2", label="data_x")

            ax.plot(
                time,
                recon_x_means[index][:, i],
                ".:",
                color=colors[index],
                label="prediction",
            )
            ax.fill_between(
                time,
                recon_x_means[index][:, i] - recon_x_stds[index][:, i],
                recon_x_means[index][:, i] + recon_x_stds[index][:, i],
                alpha=0.5,
                color=colors[index],
            )

            non_miss = ~missing_x[:, i]
            ax.plot(
                time[non_miss],
                data_x[non_miss, i],
                "o",
                color="C3",
                label="non-missing_x",
            )

            if not names_x is None:
                ax.set_title(names_x[i])
            ax.axvline(time[-num_pred + index] - 0.01, ls="--")
            ax.set_ylim(-3, 3)
            ax.legend()

            index_count += 1
