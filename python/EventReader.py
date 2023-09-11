import os
import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display, HTML


class EventReader:
    """A class for reading optical simulation data from a list of files"""
    def __init__(self, filenames):
        """Initializes the data reader with a directory"""
        self.filenames = filenames
        self.file_index = 0
        self.file = None
        self.event_names = []
        self.event_index = -1

        self.nfiles = len(self.filenames)
        print("number of files: ", len(self.filenames))
        self.read_config()
    
    def read_config(self):
        """Reads the configuration of the data reader"""
        if self.nfiles > 0:
            with h5py.File(self.filenames[0], 'r') as hf:
                self.config = json.loads(hf.attrs['config'])
            
            hf.close()
            	
    def print_config(self):
        """Prints the configuration of the data reader"""
        print(self.config)
    
    def __iter__(self):
        return self

    def __next__(self):
        """Reads the next event from the file"""	
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
        """Closes the file"""
        if self.file:
            self.file.close()

    def reset(self):
        """Resets the reader to start from the first event in the first file"""
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
        truth = np.array(event['true_position'])
        
        r = self.config['geometry']['radius']

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        fine = np.array(event['fine_top'])
        im = axs[0].imshow(fine.T, cmap='viridis', interpolation='nearest', origin='lower', extent=[-r,r,-r,r])
        plt.colorbar(im, ax=axs[0])
        axs[0].plot(truth[0], truth[1], marker='o', markersize=10, color='red', label='Marker')
        axs[0].set_xlabel('x (cm)')
        axs[0].set_ylabel('y (cm)')

        pmt = np.array(event['pmt_top'])
        im = axs[1].imshow(pmt.T, cmap='viridis', interpolation='nearest', origin='lower', extent=[-r,r,-r,r])
        plt.colorbar(im, ax=axs[1])
        axs[1].plot(truth[0], truth[1], marker='o', markersize=10, color='red', label='Marker')
        axs[1].set_xlabel('x (cm)')
        axs[1].set_ylabel('y (cm)')

        plt.show()

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

    attributes_list = []

    for subdir in subdirs:
        subdir_path = os.path.join(data_dir, subdir)
        hd5_files = sorted([f for f in os.listdir(subdir_path) if f.endswith('.hd5f')])
        
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

