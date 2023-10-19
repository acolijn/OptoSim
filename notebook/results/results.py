import glob
import os
import pickle
import sys
import json

import numpy as np
from sklearn.metrics import mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm


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

# Set some matplotlib parameters to make it look very nice and LaTex like and professional and really good ready for publication

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 16
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["text.usetex"] = True
plt.rcParams["axes.grid"] = False
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["axes.edgecolor"] = "black"

# Plot 10 random lines to test the colors
# Set also the colors to be visually pleasing and colorblind friendly
plt.style.use("seaborn-v0_8-colorblind")
plt.rcParams["image.cmap"] = "viridis"


# check if data is ok
num = 0


def plot_test_data(top, fine_top, true_pos, num, r=2.5):
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

        # Colorbar

    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")
