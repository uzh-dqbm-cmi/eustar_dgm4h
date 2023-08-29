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
    body,
    data_x,
    recon_x_mean,
    recon_x_std,
    time,
    missing_x,
    num_rec,
    names_x,
    kinds_x1,
    names,
    plot_missing=False,
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
        nP,
        1,
        sharex=False,
        sharey=False,
        figsize=(1 * figsize[0], nP * (figsize[1] + 4)),
    )
    f.subplots_adjust(hspace=0.2)  # , wspace=0.2)
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(recon_x_means)))

    for j, i in enumerate(cont_indices):
        non_miss = ~missing_x[:, i]

        if names_x[i] in names:
            if nP > 1:
                ax = axs[j]
            else:
                ax = axs
            if plot_missing:
                ax.plot(
                    time,
                    body.get_var_by_name(names_x[i])
                    .decode(data_x[:, i].reshape(-1, 1))
                    .flatten(),
                    ".-",
                    color="C2",
                    label="ground truth",
                )
            else:
                if data_x[non_miss, i].shape[0] > 0:
                    ax.plot(
                        time[non_miss],
                        body.get_var_by_name(names_x[i])
                        .decode(data_x[non_miss, i].reshape(-1, 1))
                        .flatten(),
                        ".-",
                        color="C2",
                        label="ground truth",
                    )

            for index in range(len(recon_x_means)):
                mean_rescaled = (
                    body.get_var_by_name(names_x[i])
                    .decode(recon_x_means[index][:, i].reshape(-1, 1))
                    .flatten()
                )
                ax.plot(
                    time,
                    mean_rescaled,
                    ".-",
                    color="black",
                    label="predictions",
                )
                stds_rescaled = (
                    body.get_var_by_name(names_x[i]).enc.scale_
                    * recon_x_stds[index][:, i]
                )
                #                 ax.fill_between(
                #                     time,
                #                     recon_x_means[index][:, i] - 2 * recon_x_stds[index][:, i],
                #                     recon_x_means[index][:, i] + 2 * recon_x_stds[index][:, i],
                #                     alpha=0.5,
                #                     color=colors[index],
                #                 )
                ax.fill_between(
                    time,
                    mean_rescaled - 2 * stds_rescaled,
                    mean_rescaled + 2 * stds_rescaled,
                    alpha=0.5,
                    color=colors[index],
                )

            if data_x[non_miss, i].shape[0] > 0:
                ax.plot(
                    time[non_miss],
                    body.get_var_by_name(names_x[i])
                    .decode(data_x[non_miss, i].reshape(-1, 1))
                    .flatten(),
                    "o",
                    color="C3",
                    label="available",
                )

            if not names_x is None:
                ax.set_title(names_x[i])
            if num_rec < len(time):
                ax.axvline(time[num_rec] - 0.05, ls="--")

            if data_x[non_miss, i].shape[0] > 0:
                y_min = (
                    min(
                        body.get_var_by_name(names_x[i]).enc.mean_,
                        min(
                            body.get_var_by_name(names_x[i])
                            .decode(data_x[non_miss, i].reshape(-1, 1))
                            .flatten()
                        ),
                    )
                    - 2 * body.get_var_by_name(names_x[i]).enc.scale_
                )
                y_max = (
                    max(
                        body.get_var_by_name(names_x[i]).enc.mean_,
                        max(
                            body.get_var_by_name(names_x[i])
                            .decode(data_x[non_miss, i].reshape(-1, 1))
                            .flatten()
                        ),
                    )
                    + 2 * body.get_var_by_name(names_x[i]).enc.scale_
                )
            else:
                y_min = (
                    body.get_var_by_name(names_x[i]).enc.mean_
                    - 2 * body.get_var_by_name(names_x[i]).enc.scale_
                )
                y_max = (
                    body.get_var_by_name(names_x[i]).enc.mean_
                    + 2 * body.get_var_by_name(names_x[i]).enc.scale_
                )
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("time [years]")
            ax.set_ylabel("Value")
            ax.legend(loc="lower left", fontsize="x-small")
            ax.grid(linestyle="--")

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
    times,
    categories,
    predicted_probas,
    ground_truth,
    non_miss,
    num_pred,
    name,
    plot_missing=False,
):
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
    if plot_missing:
        ax.plot(
            data["time"], data["category"], "-o", color="black", label="ground truth"
        )
    else:
        ax.plot(
            data["time"].values[non_miss.bool()],
            data["category"].values[non_miss.bool()],
            "-o",
            color="black",
            label="ground truth",
        )
    # non missing
    ax.plot(
        data["time"].values[non_miss.bool()],
        data["category"].values[non_miss.bool()],
        "o",
        color="red",
        label="available",
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
    plt.title(f"{name}")

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


def plot_label_counterfacts(eval_p, eval_p_cf):
    probas_y = [
        torch.split(elem[1], eval_p.splits * (eval_p.splits[0] + 1))
        for elem in eval_p.res_list_y
    ]
    probas_y_cf = [
        torch.split(elem[1], eval_p_cf.splits * (eval_p_cf.splits[0] + 1))
        for elem in eval_p_cf.res_list_y
    ]
    time = (
        eval_p.body.get_var_by_name("time [years]")
        .decode(eval_p.data_t[:, 0].reshape(-1, 1))
        .flatten()
    )

    for var_index, k in enumerate(eval_p.kinds_y0):
        if k != "continuous":
            name = eval_p.names_y0[var_index]
            if name in [
                "LUNG_ILD_involvement_or",
                "HEART_involvement_or",
                "ARTHRITIS_involvement_or",
            ]:
                categories = eval_p.body.get_var_by_name(name).enc.categories_[0]
                ground_truth = eval_p.body.decode(
                    eval_p.data_y, eval_p.splits_y0, eval_p.names_y0
                )[1][var_index]
                non_miss = torch.split(eval_p.non_missing_y, eval_p.splits_y0, dim=1)[
                    var_index
                ][:, 0]
                figsize = (6, 2)

                f, axs = plt.subplots(
                    eval_p.splits[0] + 1,
                    1,
                    sharex=False,
                    sharey=False,
                    figsize=(1 * figsize[0], (eval_p.splits[0] + 1) * (figsize[1] + 1)),
                )
                f.subplots_adjust(hspace=0.5)  # , wspace=0.2)
                for num_rec in range(eval_p.splits[0] + 1):
                    ax = axs[num_rec]
                    probs = probas_y[var_index][num_rec]
                    probs_cf = probas_y_cf[var_index][num_rec]
                    ax.plot(
                        time,
                        probs.detach(),
                        ".-",
                        color="black",
                        label="predictions patient ",
                    )
                    ax.plot(
                        time,
                        probs_cf.detach(),
                        ".-",
                        color="blue",
                        label="predictions counterfactual patient",
                    )
                    ax.set(ylim=(0, 1))
                    # predicted points
                    num_pred = eval_p.splits[0] - num_rec
                    if num_pred > 0:
                        ax.axvline(time[num_rec] - 0.01, ls="--")
                    ax.legend()
                    ax.set_title(name)
                    ax.set_ylabel("Probability")
                    ax.set_xlabel("time [years]")
                    ax.grid(linestyle="--")
                    ax.legend(bbox_to_anchor=(1, 1))
    return


def plot_continuous_counterfacts(eval_p, eval_p_cf, name, figure_path):

    figsize = (6, 2)

    f, axs = plt.subplots(
        eval_p.splits[0] + 1,
        1,
        sharex=False,
        sharey=False,
        figsize=(1 * figsize[0], (eval_p.splits[0] + 1) * (figsize[1] + 1)),
    )
    f.subplots_adjust(hspace=0.5)  # , wspace=0.2)
    index_name = eval_p.names_x1.index(name)
    for num_rec in range(eval_p.splits[0] + 1):
        ax = axs[num_rec]
        # predicted samples
        x_recon = torch.stack(
            [
                torch.split(elem.recon_x, eval_p.splits * (eval_p.splits[0] + 1))[
                    num_rec
                ]
                for elem in eval_p.samples
            ]
        )
        x_recon_log_var = torch.stack(
            [
                torch.split(
                    elem.recon_x_log_var, eval_p.splits * (eval_p.splits[0] + 1)
                )[num_rec]
                for elem in eval_p.samples
            ]
        )

        recon_x_means = x_recon[0].reshape(1, len(eval_p.times), eval_p.data_x.shape[1])
        recon_x_stds = torch.sqrt(torch.exp(x_recon_log_var[0])).reshape(
            1, len(eval_p.times), eval_p.data_x.shape[1]
        )
        time = (
            eval_p.body.get_var_by_name("time [years]")
            .decode(eval_p.data_t[:, 0].reshape(-1, 1))
            .flatten()
        )
        recon_x_means = [elem.detach().numpy() for elem in recon_x_means]
        recon_x_stds = [elem.detach().numpy() for elem in recon_x_stds]

        # cf quantities
        x_recon_cf = torch.stack(
            [
                torch.split(elem.recon_x, eval_p_cf.splits * (eval_p_cf.splits[0] + 1))[
                    num_rec
                ]
                for elem in eval_p_cf.samples
            ]
        )
        x_recon_log_var_cf = torch.stack(
            [
                torch.split(
                    elem.recon_x_log_var, eval_p_cf.splits * (eval_p_cf.splits[0] + 1)
                )[num_rec]
                for elem in eval_p_cf.samples
            ]
        )

        recon_x_means_cf = x_recon_cf[0].reshape(
            1, len(eval_p_cf.times), eval_p_cf.data_x.shape[1]
        )
        recon_x_stds_cf = torch.sqrt(torch.exp(x_recon_log_var_cf[0])).reshape(
            1, len(eval_p_cf.times), eval_p_cf.data_x.shape[1]
        )
        # time_cf = eval_p_cf.body.get_var_by_name("time [years]").decode(eval_p_cf.data_t[:, 0].reshape(-1,1)).flatten()
        recon_x_means_cf = [elem.detach().numpy() for elem in recon_x_means_cf]
        recon_x_stds_cf = [elem.detach().numpy() for elem in recon_x_stds_cf]
        colors = plt.cm.Blues(np.linspace(0.3, 1, 1))
        colors_cf = plt.cm.Greys(np.linspace(0.3, 1, 1))
        #         ax.plot(
        #             time,
        #             eval_p.body.get_var_by_name(name)
        #             .decode(eval_p.data_x[:, index_name].reshape(-1, 1))
        #             .flatten(),
        #             ".-",
        #             color="C2",
        #             label="ground truth",
        #         )

        mean_rescaled = (
            eval_p.body.get_var_by_name(name)
            .decode(recon_x_means[0][:, index_name].reshape(-1, 1))
            .flatten()
        )
        ax.plot(
            time,
            mean_rescaled,
            ".-",
            color="black",
            label="predictions patient",
        )
        stds_rescaled = (
            eval_p.body.get_var_by_name(name).enc.scale_
            * recon_x_stds[0][:, index_name]
        )

        ax.fill_between(
            time,
            mean_rescaled - 2 * stds_rescaled,
            mean_rescaled + 2 * stds_rescaled,
            alpha=0.5,
            color=colors[0],
        )
        # cf
        mean_rescaled_cf = (
            eval_p_cf.body.get_var_by_name(name)
            .decode(recon_x_means_cf[0][:, index_name].reshape(-1, 1))
            .flatten()
        )
        ax.plot(
            time,
            mean_rescaled_cf,
            ".-",
            color="blue",
            label="predictions counterfactual patient",
        )
        stds_rescaled_cf = (
            eval_p_cf.body.get_var_by_name(name).enc.scale_
            * recon_x_stds_cf[0][:, index_name]
        )

        ax.fill_between(
            time,
            mean_rescaled_cf - 2 * stds_rescaled_cf,
            mean_rescaled_cf + 2 * stds_rescaled_cf,
            alpha=0.5,
            color=colors_cf[0],
        )
        #         non_miss = ~eval_p.missing_x[:, index_name]
        #         if eval_p.data_x[non_miss, index_name].shape[0] > 0:
        #             ax.plot(
        #                 time[non_miss],
        #                 eval_p.body.get_var_by_name(name)
        #                 .decode(eval_p.data_x[non_miss, index_name].reshape(-1, 1))
        #                 .flatten(),
        #                 "o",
        #                 color="C3",
        #                 label="available",
        #             )

        ax.set_title(name)
        if num_rec < len(time):
            ax.axvline(time[num_rec] - 0.01, ls="--")
        ax.set_ylim(
            eval_p.body.get_var_by_name(name).enc.mean_
            - 2 * eval_p.body.get_var_by_name(name).enc.scale_,
            eval_p.body.get_var_by_name(name).enc.mean_
            + 2 * eval_p.body.get_var_by_name(name).enc.scale_,
        )
        ax.set_xlabel("time [years]")
        ax.set_ylabel("Value")
        ax.legend(loc="upper right")
        ax.grid(linestyle="--")
        ax.legend(bbox_to_anchor=(1, 1))
        # uncomment to plot mean and var of samples
        # f = plot_x_overlaid(self.data_x, x_recon_means.reshape(1, len(self.times),self.data_x.shape[1]), x_recon_std.reshape(1, len(self.times), self.data_x.shape[1]),self.data_t[:, 0], self.missing_x, num_rec,  self.names_x1, self.kinds_x1, names_cont)
        # if not os.path.exists(figure_path + "/" + str(i)):
        #     os.makedirs(figure_path + "/" + str(i))
        # f.savefig(figure_path + "/" + str(i) + "/" + name.split('/')[0].split('-')[0].split('.')[0] + "counterfactuals")

    return
