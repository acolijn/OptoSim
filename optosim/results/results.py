import glob
import os
import pickle
import sys
import json
import h5py
import argparse

import numpy as np
from sklearn.metrics import mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker


import optosim

from optosim.settings import DATA_DIR, MODEL_DIR
from optosim.settings import PROJECT_DIR

from optosim.simulation.event_reader import EventReader, show_data
from optosim.super_resolution.model import SuperResolutionModel
from optosim.super_resolution.model import create_datasets
import optosim.super_resolution.model_utils as model_utils
from optosim.model_train import read_events, get_model_parameters

from optosim.super_resolution.model_utils import (
    reshape_data,
    weighted_average_estimator,
    downsample_heatmaps_to_dimensions,
    mse,
    r_squared,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Plotting the results of the super resolution model")

    parser.add_argument(
        "--test_run_id",
        type=str,
        required=True,
        help="The id of the data to test the model on",
    )

    parser.add_argument(
        "--train_run_id",
        type=str,
        required=True,
        help="The id of the data the model was trained on",
    )

    parser.add_argument("nmax", type=int, help="The number of events to plot", default=20_000)

    parser.add_argument(
        "--normalise",
        action="store_true",
        help="Normalise the data",
    )

    return parser.parse_args()


class Results:
    def __init__(
        self,
        test_run_id,
        train_run_id,
        nmax,
        normalise=False,
        stupid_pmts=2,
        great_pmts=5,
        finest_pmts=20,
        save_figures=False,
    ):
        self.test_run_id = test_run_id
        self.train_run_id = train_run_id
        self.nmax = nmax
        self.normalise = normalise
        self.save_figures = save_figures

        self._stupid_pmts = stupid_pmts
        self._great_pmts = great_pmts
        self._finest_pmts = finest_pmts

        print(f"Initialising results for test run {self.test_run_id} and train run {self.train_run_id}")
        print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
        self.setup()

    def setup(self):
        self.test_data_path = os.path.join(DATA_DIR, self.test_run_id)
        self.model_path = os.path.join(MODEL_DIR, self.train_run_id)

        self.suffix = "_normalise" if self.normalise else ""
        self.id = f"-train_{self.train_run_id}{self.suffix}-test_{self.test_run_id}"

        self.files = glob.glob(self.test_data_path + "/*.hd*")[::-1]

        with h5py.File(self.files[0], "r") as f:
            self.config = json.loads(f.attrs.get("config"))

        self.nevents = self.config["nevents"]  # per file
        self.radius = self.config["pmt"]["size"]
        self.cylinder_radius = self.config["geometry"]["radius"]
        self.npmts_low = self.config["npmt_xy"]
        self.npmts_high = self.config["pmt"]["ndivs"] * self.npmts_low

        print(f"Number of events per file: {self.nevents}")
        print(f"Radius: {self.radius} cm")
        print(f"Cylinder radius: {self.cylinder_radius} cm")
        print(f"Number of low resolution PMTs: {self.npmts_low}")
        print(f"Number of high resolution PMTs: {self.npmts_high}")

        self.nfiles = int(np.ceil(self.nmax / self.nevents))
        print(f"Total number of files is {len(self.files)}")
        print(f"Total number of events is {len(self.files)*self.nevents}")
        print(f"---------------------------------------")
        print(f"Number of files to read: {self.nfiles}")
        print(f"Number of events to read: {self.nfiles*self.nevents} ( {self.nfiles/len(self.files)*100:.2f}% )")
        print(f"---------------------------------------")

        set_matplotlib_params()

        return

    def get_data(self):
        e = EventReader(self.files[: self.nfiles])

        true_pos = e.data_dict["true_position"]
        fine_top = e.data_dict["fine_top"]
        top = e.data_dict["pmt_top"]
        n_true_photon = e.data_dict["nphoton"]
        n_detected_photon = np.sum(top, axis=(1, 2))
        # transpose every element of top. So not top itself but every element of top
        top = np.array([np.transpose(t) for t in top])
        fine_top = np.array([np.transpose(t) for t in fine_top])

        self.true_pos = true_pos
        self.fine_top = fine_top
        self.top = top
        self.n_true_photon = n_true_photon
        self.n_detected_photon = n_detected_photon

        y = np.asarray(self.fine_top)
        X = np.asarray(self.top)
        pos = [pos[:2] for pos in self.true_pos]

        if self.normalise:
            # Normalise X and y such that sum is 1
            print("Normalising X and y such that sum is 1")
            X = [x / np.sum(x) for x in X]
            y = [y / np.sum(y) for y in y]

        X_train, y_train, pos_train, X_test, y_test, pos_test = create_datasets(X, y, pos, train_fraction=0)

        self.X_test = X_test
        self.y_test = y_test
        self.pos_test = pos_test

        print(f"---------------------------------------")
        print(f"Created test data with {len(self.X_test)} events")
        print(f"Made true_pos, fine_top, top, n_true_photon, n_detected_photon")
        print("Made X_test, y_test, pos_test")

        return X_test, y_test, pos_test

    def get_wa_model(self):
        wa_pred = weighted_average_estimator(self.X_test, r=self.radius)
        wa_mse = mse(self.pos_test, wa_pred)
        wa_r2 = r_squared(self.pos_test, wa_pred)

        print(f"Weighted average MSE: {wa_mse:.2f}")
        print(f"Weighted average R2: {wa_r2:.2f}")

        self.wa_pred = wa_pred
        self.wa_mse = wa_mse
        self.wa_r2 = wa_r2

        return wa_pred, wa_mse, wa_r2

    def _get_models(self):
        models_to_read = [
            os.path.basename(x)
            for x in glob.glob(os.path.join(MODEL_DIR, self.train_run_id, f"*{self.train_run_id}*.pkl"))
        ]

        # I want to select only the models relative to the suffix
        models_to_read = [x for x in models_to_read if self.suffix in x]
        if self.suffix == "":
            # I want to read only the models without suffix
            models_to_read = [x for x in models_to_read if "norm" not in x]
            models_to_read = [x for x in models_to_read if x.endswith(f"{self.train_run_id}.pkl")]

        models_to_read = sorted(models_to_read, key=lambda x: int(get_pmts_from_filename(x)))

        self.models = {}

        for model_name in models_to_read:
            self.models[model_name] = read_model(model_name, self.train_run_id)

    def get_model_predictions(self):
        self._get_models()

        self.results = {}

        for model_name, model in self.models.items():
            print(f"Predicting with model {model_name}", end=": ")

            _pmts = get_pmts_from_filename(model_name)
            res = model.evaluate(self.X_test, self.pos_test, normalise=self.normalise)

            self.results[model_name] = {}
            self.results[model_name]["PMTs"] = _pmts
            self.results[model_name]["pred"] = res["pred"]
            self.results[model_name]["pred_heatmap"] = res["pred_heatmap"]
            self.results[model_name]["MSE"] = res["MSE"]
            self.results[model_name]["R^2"] = res["R^2"]

            print(f"MSE: {res['MSE']:.2f} - PMTs {_pmts}")

        return self.results

    def _define_comparison_models(self):
        self.stupid_model_name = get_filename_from_pmts(self._stupid_pmts, self.train_run_id, self.suffix)

        self.great_model_name = get_filename_from_pmts(self._great_pmts, self.train_run_id, self.suffix)

        self.finest_model_name = get_filename_from_pmts(self._finest_pmts, self.train_run_id, self.suffix)

        self.stupid_model = self.models[self.stupid_model_name]
        self.great_model = self.models[self.great_model_name]
        self.finest_model = self.models[self.finest_model_name]

        print(f"Stupid model: {self.stupid_model_name}")
        print(f"Great model: {self.great_model_name}")
        print(f"Finest model: {self.finest_model_name}")

        return

    def get_comparison_model_predictions(self):
        self._define_comparison_models()

        self.r = np.linalg.norm(self.pos_test, axis=1)

        self.stupid_pred_pos = self.results[self.stupid_model_name]["pred"]
        self.great_pred_pos = self.results[self.great_model_name]["pred"]
        self.finest_pred_pos = self.results[self.finest_model_name]["pred"]

        self.wa_r_pred = np.linalg.norm(self.wa_pred, axis=1)
        self.stupid_r_pred = np.linalg.norm(self.stupid_pred_pos, axis=1)
        self.great_r_pred = np.linalg.norm(self.great_pred_pos, axis=1)
        self.finest_r_pred = np.linalg.norm(self.finest_pred_pos, axis=1)

        self.wa_dr = self.wa_r_pred - self.r
        self.stupid_dr = self.stupid_r_pred - self.r
        self.great_dr = self.great_r_pred - self.r
        self.finest_dr = self.finest_r_pred - self.r

        self.wa_dist = np.linalg.norm(self.wa_pred - self.pos_test, axis=1)
        self.stupid_dist = np.linalg.norm(self.stupid_pred_pos - self.pos_test, axis=1)
        self.great_dist = np.linalg.norm(self.great_pred_pos - self.pos_test, axis=1)
        self.finest_dist = np.linalg.norm(self.finest_pred_pos - self.pos_test, axis=1)

        return

    #################
    # Plotting
    #################

    def do_plot_mse_scatter(self, figsize=(6, 4)):
        results = self.results
        wa_mse = self.wa_mse

        fig, ax = plot_mse_scatter(results, wa_mse, figsize)

        save_figure(fig, f"rmse_per_model{self.id}", save=self.save_figures)

        return fig, ax

    def do_plot_dx_dy_histogram(self):
        # Usage
        for model in [self.stupid_model_name, self.great_model_name, self.finest_model_name]:
            _pmts = self.results[model]["PMTs"]
            fig, ax = plot_dx_dy_histogram(self.pos_test, self.results[model]["pred"], model, f"{_pmts}x{_pmts} PMTs")

            save_figure(fig, f"xy_2dhist{self.id}-{_pmts}x{_pmts}pmts", save=self.save_figures)

            plt.show()

    def do_plot_1d_histogram(self, which="dist"):
        if which == "dr":
            values = [self.wa_dr, self.stupid_dr, self.great_dr, self.finest_dr]
            labels = [
                "Weighted Avg.",
                f"{self._stupid_pmts} PMTs",
                f"{self._great_pmts} PMTs",
                f"{self._finest_pmts} PMTs",
            ]

            fig, ax = plot_1d_histogram(
                values,
                labels,
                "Deviation from true radius $\Delta r$ (cm)",
                "Number of events",
                range=(-5, 5),
                log_scale=True,
                n_bins=50,
            )

            save_figure(fig, f"dr_histogram{self.id}", save=self.save_figures)

        elif which == "dist":
            values = [self.wa_dist, self.stupid_dist, self.great_dist, self.finest_dist]
            labels = [
                "Weighted Avg.",
                f"{self._stupid_pmts} PMTs",
                f"{self._great_pmts} PMTs",
                f"{self._finest_pmts} PMTs",
            ]

            fig, ax = plot_1d_histogram(
                values,
                labels,
                "Distance between reconstructed and true position (cm)",
                "Number of events",
                range=(0, 5),
                log_scale=True,
                n_bins=50,
            )

            save_figure(fig, f"dist_histogram{self.id}", save=self.save_figures)

        else:
            print("Please choose between dr and dist")
            return

        return fig, ax

    def do_plot_statistics(self, which="dr", nbins=30):
        labels = [
            "Weighted Avg.",
            f"{self._stupid_pmts} PMTs",
            f"{self._great_pmts} PMTs",
            f"{self._finest_pmts} PMTs",
        ]

        if which == "r":
            bins = np.linspace(0, 3.2, nbins)

            statistics = compute_statistics(
                self.r,
                self.pos_test,
                [self.wa_pred, self.stupid_pred_pos, self.great_pred_pos, self.finest_pred_pos],
                bins,
                func_xy_dist,
            )

            fig, ax = plot_statistics(self.r, bins, statistics, labels)

            plt.xlabel("True $r$ (cm)")
            plt.ylabel("Average $\Delta r$ (cm)")
            save_figure(fig, f"dxy_vs_r{self.id}", save=self.save_figures)

        elif which == "ph":
            bins = np.logspace(1, 6, nbins)

            statistics = compute_statistics(
                self.n_detected_photon,
                self.pos_test,
                [self.wa_pred, self.stupid_pred_pos, self.great_pred_pos, self.finest_pred_pos],
                bins,
                func_xy_dist,
            )

            fig, ax = plot_statistics(self.n_detected_photon, bins, statistics, labels)

            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Number of photons")
            plt.ylabel("Average $\Delta$XY (cm)")
            save_figure(fig, f"dxy_vs_ph{self.id}", save=self.save_figures)

        else:
            print("Please choose between r and ph")
            return

        return fig, ax

    def do_plot_2d_hist_with_weights(self):
        for model in [self.stupid_model_name, self.great_model_name, self.finest_model_name]:
            _pmts = self.results[model]["PMTs"]
            fig, ax = plot_2d_histogram_with_weights(
                self.pos_test,
                self.results[model]["pred"],
                model,
                f"{_pmts}x{_pmts} PMTs",
                radius=self.radius,
                cylinder_radius=self.cylinder_radius,
            )
            save_figure(fig, f"xy_2dhist_with_weights{self.id}-{_pmts}x{_pmts}pmts", save=self.save_figures)

            plt.show()


#################
# Plotting
#################


def plot_test_data(top, fine_top, true_pos, num, r=2.54, cylinder_radius=3.2):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax = ax.ravel()

    im = ax[0].imshow(top[num], interpolation="nearest", origin="lower", extent=[-r, r, -r, r])

    plt.colorbar(im)

    im = ax[1].imshow(
        fine_top[num],
        interpolation="nearest",
        origin="lower",
        extent=[-r, r, -r, r],
    )

    plt.colorbar(im)

    for i in range(2):
        ax[i].plot(
            true_pos[num][0],
            true_pos[num][1],
            marker="x",
            markersize=10,
            color="red",
            label="true position",
        )

        plot_pmt_and_tpc(ax[i], r, cylinder_radius)

        ax[i].set_xlabel("x (cm)")

        # Colorbar

    ax[0].set_ylabel("y (cm)")

    return fig, ax


def plot_dx_dy_histogram(true_pos, predicted_pos, model_name, model_details, bins=50, range=((-5, 5), (-5, 5))):
    """
    Plot a 2D histogram for the deviation in x and y positions.

    Parameters:
    - true_pos: True positions.
    - predicted_pos: Predicted positions from the model.
    - model_name: Name of the model.
    - model_details: Additional details about the model (e.g., number of PMTs).
    - bins: Number of bins for the histogram.
    - range: Range for the histogram.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    plt.hist2d(
        predicted_pos[:, 0] - true_pos[:, 0],
        predicted_pos[:, 1] - true_pos[:, 1],
        bins=bins,
        range=range,
        norm=LogNorm(),
    )

    plt.xlabel("$\Delta x$ (cm)")
    plt.ylabel("$\Delta y$ (cm)")
    plt.colorbar()
    ax.set_aspect(1)
    plt.text(
        range[0][0] + 0.1 * (range[0][1] - range[0][0]),
        range[1][1] - 0.1 * (range[1][1] - range[1][0]),
        f"({model_details})",
        fontsize=16,
    )

    return fig, ax


def plot_mse_scatter(results, wa_mse, figsize=(6, 4)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Get the first color of the color cycle from the style
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]

    for i, model_name in enumerate(results.keys()):
        res = results[model_name]
        if i == 0:
            ax.scatter(res["PMTs"], res["MSE"], label="n PMTs models", color=color)
        else:
            ax.scatter(res["PMTs"], res["MSE"], c=color)

    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][1]
    ax.scatter(1, wa_mse, marker="^", color=color, label="Weighted average")

    ax.set_ylabel("Root mean squared error (cm)")
    ax.set_xlabel("Number of PMTs per dimension")
    plt.legend()

    return fig, ax


def plot_1d_histogram(data_list, labels, xlabel, ylabel, range=None, n_bins=30, log_scale=False):
    """
    Plot histograms for a list of data.

    Parameters:
    - data_list: List of data arrays to plot.
    - labels: Labels for each data array.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - range: Range for the histogram.
    - n_bins: Number of bins.
    - log_scale: Whether to use a logarithmic scale on the y-axis.
    - save_path: Path to save the figure (optional).
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    for data, label in zip(data_list, labels):
        plt.hist(
            data,
            bins=n_bins,
            range=range,
            label=label,
            histtype="step",
            linewidth=2,
        )

    plt.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    if log_scale:
        plt.yscale("log")

    return fig, ax


def plot_statistics(x_data, bins, binned_statistic_list, labels):
    """
    Plot the computed statistics.

    Parameters:
    - x_data: Data for the x-axis.
    - bins: Binning for the x-axis data.
    - binned_statistic_list: Output from compute_statistics.
    - labels: Labels for each model.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    for (mean, std), label in zip(binned_statistic_list, labels):
        plt.plot(bins[:-1], mean, label=label, linewidth=2)
        try:
            plt.fill_between(bins[:-1], mean - std, mean + std, alpha=0.2)
        except:
            pass
    plt.legend()

    return fig, ax


def plot_2d_histogram_with_weights(
    true_pos,
    predicted_pos,
    model_name,
    model_details,
    bins=50,
    range=((-5, 5), (-5, 5)),
    radius=2.54,
    cylinder_radius=3.2,
):
    """
    Plot a 2D histogram for the deviation in x and y positions.

    Parameters:
    - true_pos: True positions.
    - predicted_pos: Predicted positions from the model.
    - model_name: Name of the model.
    - model_details: Additional details about the model (e.g., number of PMTs).
    - bins: Number of bins for the histogram.
    - range: Range for the histogram.
    """
    # Calculate the average xy distance for each bin
    xy_dist = np.linalg.norm(predicted_pos - true_pos, axis=1)
    H, xedges, yedges = np.histogram2d(
        true_pos[:, 0],
        true_pos[:, 1],
        bins=bins,
        range=range,
    )
    H_dist, xedges, yedges = np.histogram2d(
        true_pos[:, 0],
        true_pos[:, 1],
        bins=bins,
        range=range,
        weights=xy_dist,
    )
    H_norm = np.divide(H_dist, H, out=np.zeros_like(H_dist), where=H != 0)

    # Plot the histogram
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    plt.imshow(
        H_norm.T,
        interpolation="nearest",
        origin="lower",
        extent=[range[0][0], range[0][1], range[1][0], range[1][1]],
        norm=LogNorm(),
    )

    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")
    plt.colorbar(label="Average $\Delta$XY (cm)")
    ax.set_aspect(1)
    plt.text(
        range[0][0] + 0.1 * (range[0][1] - range[0][0]),
        range[1][1] - 0.1 * (range[1][1] - range[1][0]),
        f"({model_details})",
        fontsize=16,
    )

    plot_pmt_and_tpc(ax, radius=radius, cylinder_radius=cylinder_radius, which="pmt")

    return fig, ax


#################
# Utils
#################


def plot_pmt_and_tpc(ax, radius, cylinder_radius, which="all"):
    if (which == "all") | (which == "tpc"):
        circle = plt.Circle((0, 0), cylinder_radius, color="r", linewidth=3, fill=False)
        ax.add_artist(circle)
        ax.set_xlim(-cylinder_radius, cylinder_radius)
        ax.set_ylim(-cylinder_radius, cylinder_radius)

    elif (which == "all") | (which == "pmt"):
        # plot the four suqares of the PMTs
        ax.plot([-radius, -radius], [radius, -radius], color="r", linewidth=3)
        ax.plot([radius, -radius], [radius, radius], color="r", linewidth=3)
        ax.plot([-radius, radius], [-radius, -radius], color="r", linewidth=3)
        ax.plot([radius, radius], [radius, -radius], color="r", linewidth=3)
        ax.plot([0, 0], [radius, -radius], color="r", linewidth=1)
        ax.plot([radius, -radius], [0, 0], color="r", linewidth=1)

    else:
        print("Please choose between tpc and pmt")

    return ax


# Define your statistic functions
def func_dr(true_pos, pred_pos):
    _r = np.linalg.norm(true_pos, axis=1)
    _pred_r = np.linalg.norm(pred_pos, axis=1)
    result = _pred_r - _r
    return result


def func_xy_dist(true_pos, pred_pos):
    return np.linalg.norm(pred_pos - true_pos, axis=1)


def compute_statistics(X, true_data, predicted_data_list, bins, statistic_func):
    """
    Compute statistics for a given property.

    Parameters:
    - true_data: The true values of the property.
    - predicted_data_list: A list of predicted values from different models.
    - bins: Binning for the x-axis data.
    - statistic_func: Function to compute the desired property (e.g., dr, xy_dist).

    Returns:
    - binned_statistic_list: A list of binned statistics for each model.
    """
    binned_statistic_list = []

    for predicted_data in predicted_data_list:
        diff = statistic_func(true_data, predicted_data)
        mean, _, _ = stats.binned_statistic(X, diff, statistic="mean", bins=bins)
        std, _, _ = stats.binned_statistic(X, diff, statistic="std", bins=bins)
        binned_statistic_list.append((mean, std))

    return binned_statistic_list


def set_matplotlib_params():
    # Set some matplotlib parameters to make it look very nice and LaTex like and professional and really good ready for publication

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 16
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["text.usetex"] = True
    plt.rcParams["axes.grid"] = False
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.linewidth"] = 1
    plt.rcParams["axes.edgecolor"] = "black"

    # Plot 10 random lines to test the colors
    # Set also the colors to be visually pleasing and colorblind friendly
    plt.style.use("seaborn-v0_8-colorblind")
    plt.rcParams["image.cmap"] = "viridis"


def read_model(model_name, train_run_id):
    model_file = os.path.join(MODEL_DIR, train_run_id, model_name)
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    return model


# Define a function to get the number of pmts from a filename like model_2x2_mc0006.pkl
def get_pmts_from_filename(filename):
    return int(filename.split("_")[1].split("x")[0])


# Define a function to get the run id from a filename like model_2x2_mc0006.pkl
def get_filename_from_pmts(pmts_per_dim, train_run_id, suffix):
    return f"model_{pmts_per_dim}x{pmts_per_dim}_{train_run_id}{suffix}.pkl"


def save_figure(fig, filename, notebook_name="results", save=False):
    """Save a matplotlib figure in the figures folder of the project.
    The prefix of the figure filename needs to be the same as the notebook filename
    The figures folder is located in PROJECT_DIR/notbooks/figures

    Args:
        fig (matplotlib.figure.Figure): Matplotlib figure to save
        filename (str): Filename of the notebook
        prefix (str, optional): Prefix of the figure filename. Defaults to None.
        folder (str, optional): Folder to save the figure in. Defaults to "figures".
    """

    prefix = notebook_name.lower()
    if save:
        folder = "figures"

        # Save one png and one pdf version of the figure
        fig.savefig(
            os.path.join(PROJECT_DIR, "notebook", folder, f"{prefix}-{filename}.png"), dpi=300, bbox_inches="tight"
        )
        fig.savefig(
            os.path.join(PROJECT_DIR, "notebook", folder, f"{prefix}-{filename}.pdf"), dpi=300, bbox_inches="tight"
        )

        print(f"Figure saved as {prefix}_{filename}")

    else:
        print(f"Figure NOT saved as {prefix}_{filename}")
