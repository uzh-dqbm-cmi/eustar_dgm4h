import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors as col

# progress visualization
from tqdm.notebook import tqdm


class Patient:
    """
    Class for storing and processing a patient.
    """

    def __init__(self, df_vis=None, df_pat=None, df_med=None):
        """
        Constructing a patient with the visits, patients and medication data
        """

        # raw data
        self.df_vis = df_vis
        self.df_pat = df_pat
        self.df_med = df_med

        if not self.df_vis is None:
            # number of visits and meds
            self.n_visits = self.df_vis.shape[0]
            self.n_meds = self.df_med.shape[0]

            # Preprocess the patient data
            self.init_preprocess()

            # create labels
            # self.create_labels()

    def init_preprocess(self):
        """
        Prerocess the patient data
        """

        # sort and preprocess visits (on patient level)
        self.sort_preprocess_vis()

        # preprocess pat
        # TODO

        # sort and preprocess meds (on patient level)
        # TODO

    def sort_preprocess_vis(self):
        """
        Sort and preprocess visits (on patient level).
        Inparticular, creating variables time, aligned from the first visit.
        """

        # sort visits
        self.df_vis = self.df_vis.sort_values(["Visit Date"])

        # compute time in days, weeks and years
        self.df_vis["time [days]"] = (
            self.df_vis["Visit Date"] - self.df_vis["Visit Date"].iloc[0]
        ).apply(lambda x: x.total_seconds() / (3600 * 24))

        self.df_vis["time [weeks]"] = self.df_vis["time [days]"] / (7)
        self.df_vis["time [months]"] = self.df_vis["time [days]"] / (365.25 / 12)
        self.df_vis["time [quarters]"] = self.df_vis["time [days]"] / (365.25 / 4)
        self.df_vis["time [half-years]"] = self.df_vis["time [days]"] / (365.25 / 2)
        self.df_vis["time [years]"] = self.df_vis["time [days]"] / (365.25)

        return self.df_vis

    def plot_df(self, attr_name="df_vis_LUNG_ILD", figsize=(5, 2)):
        df = getattr(self, attr_name)

        nP = df.shape[1]

        # create figure with subplots
        f, axs = plt.subplots(
            nP, 1, sharex=True, sharey=False, figsize=(1 * figsize[0], nP * figsize[1])
        )
        f.subplots_adjust(hspace=0.2)  # , wspace=0.2)

        time = df.iloc[:, 0]

        for i, col in enumerate(df):
            ax = axs[i]
            ax.plot(time, df[col], ".-")
            ax.set_title(col)

    def plot_array(self, attr_name="array_vis_LUNG_ILD", figsize=(5, 2)):
        array = getattr(self, attr_name)

        nP = array.shape[1]

        # create figure with subplots
        f, axs = plt.subplots(
            nP, 1, sharex=True, sharey=False, figsize=(1 * figsize[0], nP * figsize[1])
        )
        f.subplots_adjust(hspace=0.2)  # , wspace=0.2)

        time = array[:, 0]

        for i in range(nP):
            ax = axs[i]
            ax.plot(time, array[:, i], ".-")
            ax.set_title(self.encoding_names[i])
