import h5py
import json
import matplotlib.pyplot as plt
import numpy as np

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

        pmt = np.array(event['pmt_top'])
        im = axs[1].imshow(pmt.T, cmap='viridis', interpolation='nearest', origin='lower', extent=[-r,r,-r,r])
        plt.colorbar(im, ax=axs[1])
        axs[1].plot(truth[0], truth[1], marker='o', markersize=10, color='red', label='Marker')

        plt.show()



