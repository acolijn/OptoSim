import os
import h5py
import json
import numpy as np
import pandas as pd
#from IPython.display import display, HTML

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
    __iter__()
        Returns an iterator for the data reader
    __next__()
        Reads the next event from the file
    close()
        Closes the file
    reset()
        Resets the reader to start from the first event in the first file
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
        self.filenames = filenames
        self.file_index = 0
        self.file = None
        self.event_names = []
        self.event_index = -1

        self.nfiles = len(self.filenames)
        print("number of files: ", len(self.filenames))
        self.read_config()
    
    def read_config(self):
        """Reads the configuration of the data reader
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None

        A.P. Colijn
        """
        if self.nfiles > 0:
            with h5py.File(self.filenames[0], 'r') as hf:
                self.config = json.loads(hf.attrs['config'])
            
            hf.close()
            	
    def print_config(self):
        """Prints the configuration of the data reader
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        A.P. Colijn
        """
        print(self.config)
    
    def __iter__(self):
        """Returns an iterator for the data reader

        Parameters
        ----------
        None

        Returns
        -------
        self : EventReader
            The iterator

        A.P. Colijn
        """
        return self

    def __next__(self):
        """Reads the next event from the file
        
        Parameters
        ----------
        None
        
        Returns
        -------
        event_data : dict
            The event data
                
        A.P. Colijn
        """	
        while True:
            # Check if it's time to load or switch to a new file
            if self.file is None or self.event_index >= len(self.event_names) - 1:
                # Close current file if it exists
                if self.file:
                    self.file.close()

                # Check if we're out of files to read from
                if self.file_index >= len(self.filenames):
                    raise StopIteration

                # Open next file
                self.file = h5py.File(self.filenames[self.file_index], 'r')
                self.file_index += 1
                self.event_names = list(self.file['events'].keys())
                self.event_index = -1

            self.event_index += 1
            event_dataset = self.file['events'][self.event_names[self.event_index]]
            event_data = {key: event_dataset[key][()] for key in event_dataset.keys()}
            return event_data

    def close(self):
        """Closes the file
        
        Parameters
        ----------
        None

        Returns
        -------
        None

        A.P. Colijn
        """
        if self.file:
            self.file.close()

    def reset(self):
        """Resets the reader to start from the first event in the first file
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        A.P. Colijn
        """
        # If a file is currently open, close it
        if self.file:
            self.file.close()
            self.file = None

        # Reset indices and counters
        self.file_index = 0
        self.event_index = -1

    def print_event(self, event):
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
        print(event)

    def show_event(self, event):
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

        # Get the true position of the event
        truth = np.array(event['true_position'])
        # Get the size of the PMT array
        dx = self.config['pmt']['size']*self.config['npmt_xy']/2
        # Get the radius of the cylinder
        radius = self.config['geometry']['radius']

        # Create figure
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        # Plot fine bin signal
        fine = np.array(event['fine_top'])
        im = axs[0].imshow(fine.T, cmap='viridis', interpolation='nearest', origin='lower', extent=[-dx,dx,-dx,dx])
        plt.colorbar(im, ax=axs[0])
        axs[0].plot(truth[0], truth[1], marker='o', markersize=10, color='red', label='Marker')
        axs[0].set_xlabel('x (cm)')
        axs[0].set_ylabel('y (cm)')
        axs[0].set_xlim([-radius,radius])
        axs[0].set_ylim([-radius,radius])
        # Draw a circle with a specified radius
        circle = Circle((0, 0), radius, color='blue', fill=False)
        axs[0].add_patch(circle)
        # Draw the outline of the PMTs
        # Iterate through PMT positions and draw green boxes
        pmt_size = self.config['pmt']['size']
        for ix in range(self.config['npmt_xy']):
            for iy in range(self.config['npmt_xy']):

                x_pmt = ix*pmt_size - dx + pmt_size/2   
                y_pmt = iy*pmt_size - dx + pmt_size/2
                pmt_box = Rectangle((x_pmt - pmt_size/2, y_pmt - pmt_size/2), pmt_size, pmt_size, color='white', fill=False)
                axs[0].add_patch(pmt_box)


        # Plot PMT signal
        pmt = np.array(event['pmt_top'])
        im = axs[1].imshow(pmt.T, cmap='viridis', interpolation='nearest', origin='lower', extent=[-dx,dx,-dx,dx])
        plt.colorbar(im, ax=axs[1])
        axs[1].plot(truth[0], truth[1], marker='o', markersize=10, color='red', label='Marker')
        axs[1].set_xlabel('x (cm)')
        axs[1].set_ylabel('y (cm)')
        axs[1].set_xlim([-radius,radius])
        axs[1].set_ylim([-radius,radius])
        circle = Circle((0, 0), radius, color='blue', fill=False)
        axs[1].add_patch(circle)
        for ix in range(self.config['npmt_xy']):
            for iy in range(self.config['npmt_xy']):

                x_pmt = ix*pmt_size - dx + pmt_size/2   
                y_pmt = iy*pmt_size - dx + pmt_size/2
                pmt_box = Rectangle((x_pmt - pmt_size/2, y_pmt - pmt_size/2), pmt_size, pmt_size, color='white', fill=False)
                axs[1].add_patch(pmt_box)

        plt.show()


    def show_event(self, event):
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
        # Get the true position of the event
        truth = np.array(event['true_position'])
        # Get the size of the PMT array (in the end this is a bit hackky, since the array is twic this size)
        dx = self.config['pmt']['size'] * self.config['npmt_xy'] 
        # Get the radius of the cylinder
        radius = self.config['geometry']['radius']

        # Create figure and axes
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        # Plot fine bin signal and PMT signal
        fine = np.array(event['fine_top'])
        pmt = np.array(event['pmt_top'])

        # Plot fine bin signal
        self.plot_signal(axs[0], fine, truth, dx, radius, 'Fine Bin Signal')
        # Plot PMT signal
        self.plot_signal(axs[1], pmt, truth, dx, radius, 'PMT Signal')

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
        im = ax.imshow(data.T, cmap='viridis', interpolation='nearest', origin='lower', extent=[-dx/2, dx/2, -dx/2, dx/2])
        plt.colorbar(im, ax=ax)
        ax.plot(truth[0], truth[1], marker='o', markersize=10, color='red', label='Marker')
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_xlim([-radius, radius])
        ax.set_ylim([-radius, radius])
        circle = Circle((0, 0), radius, color='blue', fill=False)
        ax.add_patch(circle)
        self.draw_pmt_outline(ax, dx/2)

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
        pmt_size = self.config['pmt']['size']
        for ix in range(self.config['npmt_xy']):
            for iy in range(self.config['npmt_xy']):
                x_pmt = ix * pmt_size - dx + pmt_size / 2
                y_pmt = iy * pmt_size - dx + pmt_size / 2
                pmt_box = Rectangle((x_pmt - pmt_size/2, y_pmt - pmt_size/2), pmt_size, pmt_size, color='white', fill=False)
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
        hd5_files = sorted([f for f in os.listdir(subdir_path) if f.endswith('.hd5f')])

        # Loop over files in subdirectory      
        if hd5_files:
            first_hd5_file = os.path.join(subdir_path, hd5_files[0])
            try:
                with h5py.File(first_hd5_file, 'r') as file:
                    attrs = dict(file.attrs)
                    
                    config_str = attrs.get('config', None)
                    if config_str is not None:
                        config_dict = json.loads(config_str)
                        config_dict['subdir'] = subdir
                        if 'geometry' in config_dict:
                            config_dict.update(config_dict['geometry'])
                        if 'pmt' in config_dict:
                            config_dict.update(config_dict['pmt'])

                        attributes_list.append(config_dict)
            except OSError:
                print(f"File {first_hd5_file} is currently open by another process. Skipping...")
            except Exception as e:
                print(f"Error reading file {first_hd5_file}: {e}")


    df = pd.DataFrame(attributes_list)
    #display(HTML(df.to_html(index=False)))
    cols = ['subdir','detector','nevents','nphoton_per_event','set_no_scatter','set_experimental_scatter_model', 'radius']
    # Filter the list to include only columns that exist in the DataFrame
    cols = [col for col in cols if col in df.columns]

    return df[cols]

