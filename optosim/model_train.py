import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys

import optosim

from optosim.settings import DATA_DIR, MODEL_DIR
from optosim.simulation.event_reader import EventReader, show_data
from optosim.super_resolution.model import SuperResolutionModel
from optosim.super_resolution.model import create_datasets
import optosim.super_resolution.model_utils as model_utils


def argparser():
    """
    Parse command line arguments
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train a neural network to do super resolution")

    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="Run ID",
    )

    parser.add_argument(
        "--nmax",
        type=int,
        default=10_000_000,
        help="Maximum number of events to read",
    )

    parser.add_argument(
        "--pmts_per_dim",
        type=int,
        default=5,
        help="Number of PMTs per dimension",
    )

    return parser.parse_args()


def main():
    args = argparser()

    run_id = args.run_id
    nmax = args.nmax
    pmts_per_dim = args.pmts_per_dim

    # read data
    run_id_dir = os.path.join(DATA_DIR, run_id)
    files = glob.glob(run_id_dir + "/*.hdf5")

    print(f"Reading data from {run_id_dir}")
    print(f"Found {len(files)} files")

    true_pos, fine_top, top = read_events(files, nmax=nmax)

    # create train and test sets
    y = fine_top  # downsample if wanted
    X = top
    pos = [pos[:2] for pos in true_pos]  # depth is not used

    # Normalise X and y such that sum is 1
    X = [x / np.sum(x) for x in X]
    y = [y / np.sum(y) for y in y]

    X_train, y_train, pos_train, X_test, y_test, pos_test = create_datasets(X, y, pos, train_fraction=0.8)

    y_train_downsampled = model_utils.downsample_heatmaps_to_dimensions(y_train, pmts_per_dim, pmts_per_dim)
    y_test_downsampled = model_utils.downsample_heatmaps_to_dimensions(y_test, pmts_per_dim, pmts_per_dim)

    low_to_high_res_net_params, high_res_to_true_net_params = get_model_parameters()

    model = SuperResolutionModel(
        low_to_high_res_net_params=low_to_high_res_net_params, high_res_to_true_net_params=high_res_to_true_net_params
    )

    model.train(X_train, y_train_downsampled, pos_train)

    # Make model dir if it doesn't exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"Created {MODEL_DIR} because it didn't exist")

    # Create run_id dir in model dir if it doesn't exist
    run_id_dir = os.path.join(MODEL_DIR, run_id)
    if not os.path.exists(run_id_dir):
        os.makedirs(run_id_dir)
        print(f"Created {run_id_dir} because it didn't exist")

    # save
    outfile = os.path.join(run_id_dir, f"model_{pmts_per_dim}x{pmts_per_dim}_{run_id}.pkl")

    if os.path.exists(outfile):
        print(f"WARNING: {outfile} already exists. Overwriting.")

    with open(outfile, "wb") as f:
        pickle.dump(model, f)


def read_events(files, nmax=1_000_000):
    """
    Read the events from the files and return the data in the correct format

    Parameters
    ----------
    files : list
        list of files to read
    n : int
        number of firt event to read
    nmax : int
        maximum number of events to read
    """

    # This for now is still a bit silly
    # We read 1 million events and then take the first nmax
    e = EventReader(files)

    # show data in directory
    show_data(DATA_DIR)

    true_pos = e.data_dict["true_position"][:nmax]
    fine_top = e.data_dict["fine_top"][:nmax]
    top = e.data_dict["pmt_top"][:nmax]
    n_true_photon = e.data_dict["nphoton"][:nmax]

    # transpose every element of top. So not top itself but every element of top
    top = np.array([np.transpose(t) for t in top])
    fine_top = np.array([np.transpose(t) for t in fine_top])

    print(f"We have {len(top)} events")
    print(f"low res PMT has shape {top[0].shape}")
    print(f"high res truth has shape {fine_top[0].shape}")
    print(f"true positions have shape {true_pos[0].shape}")

    return true_pos, fine_top, top


def get_model_parameters():
    low_to_high_res_net_params = {
        "hidden_layer_sizes": (100, 100),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0001,
        "batch_size": "auto",
        "learning_rate": "constant",
        "learning_rate_init": 0.001,
        "power_t": 0.5,
        "max_iter": 200,
        "shuffle": True,
        "random_state": None,
        "tol": 1e-4,
        "verbose": False,
        "warm_start": False,
        "momentum": 0.9,
        "nesterovs_momentum": True,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-8,
        "n_iter_no_change": 10,
        "max_fun": 15000,
    }

    high_res_to_true_net_params = {
        "hidden_layer_sizes": (100,),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0001,
        "batch_size": "auto",
        "learning_rate": "constant",
        "learning_rate_init": 0.001,
        "power_t": 0.5,
        "max_iter": 200,
        "shuffle": True,
        "random_state": None,
        "tol": 1e-4,
        "verbose": False,
        "warm_start": False,
        "momentum": 0.9,
        "nesterovs_momentum": True,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-8,
        "n_iter_no_change": 10,
        "max_fun": 15000,
    }

    return low_to_high_res_net_params, high_res_to_true_net_params


if __name__ == "__main__":
    main()
