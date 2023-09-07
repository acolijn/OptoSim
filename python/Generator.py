import numpy as np
import json
import os
import h5py
import OpticalPhoton as op

class Generator:
    """ Class for generating S2-like events. Photons are generated from positions as 
        specified in the configuration of the class. The photons are propagated through 
        the detector and the PMT response is simulated. The class can be used to generate
        a single event or a batch of events.

        The data structure that is used to store the events is a dictionary with the following
        structure:
        1. The configuration is stored in a dictionary with the following structure:
            {   'filename': 'filename', output filename
                'radius': 0,
                'ztop', 0,
                'zbot': 0,
                'zliq': 0,
                'N_fine': 0,
                'detector_fine_x': [N_fine x N_fine] x-positions of the fine detector
                'detector_fine_y': [N_fine x N_fine] y-positions of the fine detector
                'N_det': 0,
                'detector_x': [N_det x N_det] x-positions of the real detector
                'detector_y': [N_det x N_det] y-positions of the real detector
            }             

        2. The per event data is stored in a dictionary with the following structure:
            {           
                'number': 0,
                'true_position': [0, 0, 0],
                'detected_photons_fine': [N_fine x N_fine] for each event. This is the number
                    of photons detected in the 'fine' detector.
                'detected_photons_det': [N_det x N_det] for each event. This is the number of 
                    photons deteccted in the 'real' detector.
            }

    """
    
    def __init__(self, **kwargs):
        """Initializes the generator class. The configuration is read from the config file.

        """
        # Read configuration file
        self.config_file = kwargs.get('config', 'config.json')
        if os.path.isfile(self.config_file):
            print("Generator::Reading configuration from file: {}".format(self.config_file))
            self.config = json.load(open(self.config_file, 'r'))
        else:
            raise ValueError("Config file does not exist.")
        
        # Initialize the detector
        self.radius = self.config['geometry']['radius']

        self.ievent = 0 # Event counter

        # define an optical photon
        self.aPhoton = op.OpticalPhoton(config=self.config_file)

            
    def generate_event(self):
        """Generates a single event. 

        The x,y position of the event is generated from a uniform distribution. The z-position
        is at a fixed value.

        """
        # Generate random position
        phi = np.random.uniform(0, 2*np.pi)
        r = self.radius * np.sqrt(np.random.uniform(0, 1))
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = self.config['photon_zgen']

        print(x,y,z)

        event_data = {}
        event_data['number'] = self.ievent
        event_data['true_position'] = [x, y, z]

        return event_data
    
    
    def generate_batch(self, batch_size):
        """Generates a batch of events. 
        """

        self.open_file(self.config['filename']) # Open file for writing events to       

        for i in range(batch_size):

            self.ievent = i
            event = self.generate_event()
            self.write_event(event) # Write event to file

        self.close_file() # Close file

        return 0

    def open_file(self, filename):
        """Opens a file for writing events to.
        """
        # Check if file exists
        #if os.path.isfile(filename):
        #    raise ValueError("File already exists.")
        
        # Open file
        self.file = h5py.File(filename, 'w')
        self.file.attrs['config'] = json.dumps(self.config)

        # Create group for configuration
        self.event_group = self.file.create_group('events')

        return 0
    
    def write_event(self, event):
        """Writes an event to the file.
        """

        # Create group for event
        ev = 'event_{}'.format(self.ievent)
        event_group = self.event_group.create_group(ev)

        # Write event data
        for key in event.keys():
            print(key)
            event_group[key] = event[key]

        return 0    
    
    def close_file(self):
        """Closes the file.
        """
        self.file.close()
        return 0

        