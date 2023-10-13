import os
import h5py
import json
import numpy as np
import pandas as pd

# from IPython.display import display, HTML

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

class EventReader:
    """A class for reading optical simulation data from a list of files

    Parameters
    ----------
    filenames : list
        A list of filenames to read from

    Attributes
    ----------
    filenames : list
        A list of filenames to read from
    file_index : int
        The index of the current file being read
    file : h5py.File
        The current file being read
    event_names : list
        A list of the names of the events in the current file
    event_index : int
        The index of the current event being read
    nfiles : int
        The number of files to read from
    config : dict
        The configuration of the data reader

    Methods
    -------
    read_config()
        Reads the configuration of the data reader
    print_config()
        Prints the configuration of the data reader
    print_event(event)
        Prints the event
    show_event(event)
        Shows the event

    A.P. Colijn
    """
    def __init__(self, filenames):
        """Initializes the data reader with a directory

        Parameters
        ----------
        filenames : list
            A list of filenames to read from

        Returns
        -------
        None

        A.P. Colijn
        """

        # Load config attribute from the first file
        with h5py.File(filenames[0], 'r') as f:
            self.config = json.loads(f.attrs.get("config"))

        print("huh config = ", self.config)        

        if self.config['data_type_version'] == 1.0:
            # throw an error that this version is obsolete
            raise ValueError("Data type version 1.0 is obsolete. Please use version 2.0")

        # Load data from all files
        self.data_dict = self._load_data(filenames)
        self.number_of_files = len(filenames)
        self.num_events = len(next(iter(self.data_dict.values())))  # assuming all datasets have the same length

        print("EventReader initialized with:")
        print("  number of files: ", self.number_of_files)
        print("  number of events: ", self.num_events)
        print("  configuration: ", self.config)
            
    def _load_data(self, filenames):
        """Loads data from a list of files

        Parameters
        ----------
        filenames : list
            A list of filenames to read from

        Returns
        -------
        data_dict : dict
            A dictionary containing the data from all files

        A.P. Colijn
        """
        data_dicts = []        
        for filename in filenames:
            with h5py.File(filename, 'r') as f:
                group = f['events']
                data_dict = {name: np.array(dataset) for name, dataset in group.items()}
                data_dicts.append(data_dict)
        
        # Now, concatenate arrays from all dictionaries along the first axis (events)
        # Start with the first dictionary and update it with concatenated arrays from other dictionaries
        combined_data = data_dicts[0]
        for key in combined_data.keys():
            combined_data[key] = np.concatenate([d[key] for d in data_dicts], axis=0)
        
        return combined_data

    def get_event(self, n):
        if n >= self.num_events:
            raise IndexError("Event index out of range")
        
        return {name: dataset[n] for name, dataset in self.data_dict.items()}
    
    def print_event(self, n):
        """Prints the event

        Parameters
        ----------
        event : dict
            The event to print

        Returns
        -------
        None

        A.P. Colijn
        """
        event = self.get_event(n)
        print(event)

    # def show_event(self, n):
    #     """Shows the event

    #     Parameters
    #     ----------
    #     event : dict
    #         The event to show

    #     Returns
    #     -------
    #     None

    #     A.P. Colijn
    #     """

    #     # get the event
    #     event = self.get_event(n)

    #     # Get the true position of the event
    #     truth = np.array(event["true_position"])
    #     # Get the size of the PMT array
    #     dx = self.config["pmt"]["size"] * self.config["npmt_xy"] / 2
    #     # Get the radius of the cylinder
    #     radius = self.config["geometry"]["radius"]

    #     # Create figure
    #     fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    #     # Plot fine bin signal
    #     fine = np.array(event["fine_top"])
    #     im = axs[0].imshow(fine.T, cmap="viridis", interpolation="nearest", origin="lower", extent=[-dx, dx, -dx, dx])
    #     plt.colorbar(im, ax=axs[0])
    #     axs[0].plot(truth[0], truth[1], marker="o", markersize=10, color="red", label="Marker")
    #     axs[0].set_xlabel("x (cm)")
    #     axs[0].set_ylabel("y (cm)")
    #     axs[0].set_xlim([-radius, radius])
    #     axs[0].set_ylim([-radius, radius])
    #     # Draw a circle with a specified radius
    #     circle = Circle((0, 0), radius, color="blue", fill=False)
    #     axs[0].add_patch(circle)
    #     # Draw the outline of the PMTs
    #     # Iterate through PMT positions and draw green boxes
    #     pmt_size = self.config["pmt"]["size"]
    #     for ix in range(self.config["npmt_xy"]):
    #         for iy in range(self.config["npmt_xy"]):
    #             x_pmt = ix * pmt_size - dx + pmt_size / 2
    #             y_pmt = iy * pmt_size - dx + pmt_size / 2
    #             pmt_box = Rectangle(
    #                 (x_pmt - pmt_size / 2, y_pmt - pmt_size / 2), pmt_size, pmt_size, color="white", fill=False
    #             )
    #             axs[0].add_patch(pmt_box)

    #     # Plot PMT signal
    #     pmt = np.array(event["pmt_top"])
    #     im = axs[1].imshow(pmt.T, cmap="viridis", interpolation="nearest", origin="lower", extent=[-dx, dx, -dx, dx])
    #     plt.colorbar(im, ax=axs[1])
    #     axs[1].plot(truth[0], truth[1], marker="o", markersize=10, color="red", label="Marker")
    #     axs[1].set_xlabel("x (cm)")
    #     axs[1].set_ylabel("y (cm)")
    #     axs[1].set_xlim([-radius, radius])
    #     axs[1].set_ylim([-radius, radius])
    #     circle = Circle((0, 0), radius, color="blue", fill=False)
    #     axs[1].add_patch(circle)
    #     for ix in range(self.config["npmt_xy"]):
    #         for iy in range(self.config["npmt_xy"]):
    #             x_pmt = ix * pmt_size - dx + pmt_size / 2
    #             y_pmt = iy * pmt_size - dx + pmt_size / 2
    #             pmt_box = Rectangle(
    #                 (x_pmt - pmt_size / 2, y_pmt - pmt_size / 2), pmt_size, pmt_size, color="white", fill=False
    #             )
    #             axs[1].add_patch(pmt_box)

    #     plt.show()

    def show_event(self, n):
        """Shows the event

        Parameters
        ----------
        event : dict
            The event to show

        Returns
        -------
        None

        A.P. Colijn
        """

        event = self.get_event(n)

        # Get the true position of the event
        truth = np.array(event["true_position"])
        # Get the size of the PMT array (in the end this is a bit hackky, since the array is twic this size)
        dx = self.config["pmt"]["size"] * self.config["npmt_xy"]
        # Get the radius of the cylinder
        radius = self.config["geometry"]["radius"]

        # Create figure and axes
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        # Plot fine bin signal and PMT signal
        fine = np.array(event["fine_top"])
        pmt = np.array(event["pmt_top"])

        # Plot fine bin signal
        self.plot_signal(axs[0], fine, truth, dx, radius, "Fine Bin Signal")
        # Plot PMT signal
        self.plot_signal(axs[1], pmt, truth, dx, radius, "PMT Signal")

        plt.show()

    def plot_signal(self, ax, data, truth, dx, radius, title):
        """Plots the signal

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to plot on
        data : numpy.ndarray
            The data to plot
        truth : numpy.ndarray
            The true position of the event
        dx : float
            The size of the PMT array
        radius : float
            The radius of the cylinder
        title : str
            The title of the plot

        Returns
        -------
        None

        A.P. Colijn
        """
        im = ax.imshow(
            data.T, cmap="viridis", interpolation="nearest", origin="lower", extent=[-dx / 2, dx / 2, -dx / 2, dx / 2]
        )
        plt.colorbar(im, ax=ax)
        ax.plot(truth[0], truth[1], marker="o", markersize=10, color="red", label="Marker")
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_xlim([-radius, radius])
        ax.set_ylim([-radius, radius])
        circle = Circle((0, 0), radius, color="blue", fill=False)
        ax.add_patch(circle)
        self.draw_pmt_outline(ax, dx / 2)

    def draw_pmt_outline(self, ax, dx):
        """Draws the outline of the PMTs

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to plot on
        dx : float
            The size of the PMT array

        Returns
        -------
        None

        A.P. Colijn
        """
        pmt_size = self.config["pmt"]["size"]
        for ix in range(self.config["npmt_xy"]):
            for iy in range(self.config["npmt_xy"]):
                x_pmt = ix * pmt_size - dx + pmt_size / 2
                y_pmt = iy * pmt_size - dx + pmt_size / 2
                pmt_box = Rectangle(
                    (x_pmt - pmt_size / 2, y_pmt - pmt_size / 2), pmt_size, pmt_size, color="white", fill=False
                )
                ax.add_patch(pmt_box)


def show_data(data_dir):
    """Shows the data

    Parameters
    ----------
    data_dir : str
        The directory containing the data

    Returns
    -------
    None

    A.P. Colijn
    """
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    subdirs = sorted(subdirs)

    print(f"Found {len(subdirs)} subdirectories")
    print("Subdirectories:")
    print(subdirs)

    attributes_list = []
    # Loop over subdirectories
    for subdir in subdirs:
        # Get list of files in subdirectory
        subdir_path = os.path.join(data_dir, subdir)
        hd5_files = sorted([f for f in os.listdir(subdir_path) if (f.endswith(".hdf5") or f.endswith(".hd5f"))])

        # Loop over files in subdirectory
        if hd5_files:
            first_hd5_file = os.path.join(subdir_path, hd5_files[0])
            try:
                with h5py.File(first_hd5_file, "r") as file:
                    attrs = dict(file.attrs)

                    config_str = attrs.get("config", None)
                    if config_str is not None:
                        config_dict = json.loads(config_str)
                        config_dict["subdir"] = subdir
                        if "geometry" in config_dict:
                            config_dict.update(config_dict["geometry"])
                        if "pmt" in config_dict:
                            config_dict.update(config_dict["pmt"])

                        attributes_list.append(config_dict)
            except OSError:
                print(f"File {first_hd5_file} is currently open by another process. Skipping...")
            except Exception as e:
                print(f"Error reading file {first_hd5_file}: {e}")

    df = pd.DataFrame(attributes_list)
    # display(HTML(df.to_html(index=False)))
    cols = [
        "subdir",
        "detector",
        "nevents",
        "nphoton_per_event",
        "scatter",
        "experimental_scatter_model",
        "radius",
    ]
    # Filter the list to include only columns that exist in the DataFrame
    cols = [col for col in cols if col in df.columns]

    return df[cols]
