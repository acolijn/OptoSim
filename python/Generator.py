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
            {
                "filename": filename of output
                "nevents": number of events to generate,
                "nphoton_per_event":  number of photons to generate per event,
                "photon_zgen": z-position of photon generation,
                "geometry":{
                    "type": type of geometry (currently only 'cylinder' is supported),
                    "radius": radius of cylinder,
                    "ztop": top of cylinder,
                    "zliq": liquid level,
                    "zbot": bottom of cylinder
                }
                "npmt_xy": pmt grid size -> 1 = 1x1 grid, 2 = 2x2 grid, etc. (a single 2" hamamatsu 
                                        multianode PMT should be defined as 2x2 grid!)
                "pmt":{
                    "type": type of pmt (currently only 'square' is supported),
                    "size": size of pmt,
                    "ndivs": number of divisions in pmt for fine granularity (2 means 2x2 grid, etc.)
                }
            }  

        2. The per event data is stored in a dictionary with the following structure:
            {           
                'number': event number,
                'true_position': true position of event,
                'fine_top': [npmt_xy*ndivs x npmt_xy*ndivs] for each event. This is the number
                    of photons detected in the 'fine' detector top.
                'pmt_top': [npmt_xy x npmt_xy] for each event. This is the number of 
                    photons deteccted in the 'real' detector top.
                'fine_bot': [npmt_xy*ndivs x npmt_xy*ndivs] for each event. This is the number
                    of photons detected in the 'fine' detector bottom.
                'pmt_bot': [npmt_xy x npmt_xy] for each event. This is the number of 
                    photons deteccted in the 'real' detector bottom.
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
            raise ValueError("Config file does not exist. {}".format(self.config_file))
        
        # Initialize the detector
        self.radius = self.config['geometry']['radius']

        self.ievent = 0 # Event counter

        # define an optical photon
        self.aPhoton = op.OpticalPhoton(config=self.config_file)

    def generate_event(self):
        """Generates a single event. 

        The x,y position of the event is generated from a uniform distribution. The z-position
        is at a fixed value.

        The event data is stored in a dictionary with the following structure:
            {
                'number': event number,
                'true_position': true position of event,
                'fine_top': [npmt_xy*ndivs x npmt_xy*ndivs] for each event. This is the number
                    of photons detected in the 'fine' detector top.
                'pmt_top': [npmt_xy x npmt_xy] for each event. This is the number of
                    photons deteccted in the 'real' detector top.
                'fine_bot': [npmt_xy*ndivs x npmt_xy*ndivs] for each event. This is the number  
                    of photons detected in the 'fine' detector bottom.
                'pmt_bot': [npmt_xy x npmt_xy] for each event. This is the number of
                    photons deteccted in the 'real' detector bottom.
            }

        """
        # Generate random position
        x0 = self.generate_random_position()

        # Get detector parameters
        offset = self.config['npmt_xy']*self.config['pmt']['size']/2
        npmt = self.config['npmt_xy']
        nfine = self.config['npmt_xy']*self.config['pmt']['ndivs']
        dx = self.config['pmt']['size']
        dy = self.config['pmt']['size']

        ndivs = self.config['pmt']['ndivs']
        dx_fine = dx/ndivs
        dy_fine = dy/ndivs
        # Initialize pmt signal
        pmt_signal_top = np.zeros((npmt, npmt), dtype=np.int32)
        pmt_signal_bot = np.zeros((npmt, npmt), dtype=np.int32)

        # Initialize fine signal
        fine_signal_top = np.zeros((nfine, nfine), dtype=np.int32)
        fine_signal_bot = np.zeros((nfine, nfine), dtype=np.int32)

        # Generate photons
        for _ in range(self.config['nphoton_per_event']):
            # Generate photon at position x0
            self.aPhoton.generate_photon(x0)
            # Propagate photon.... this is where the magic happens
            self.aPhoton.propagate()
            # Check if photon is detected
            if self.aPhoton.is_detected():
                # get photon position
                x = self.aPhoton.get_photon_position()
                # get bin in x and y
                ix = int((x[0] + offset) / dx)
                iy = int((x[1] + offset) / dy)
                # if in range
                if ix < npmt and iy < npmt:
                    if x[2] > 0:
                        # add photon to pmt bin in top array
                        pmt_signal_top[ix, iy] += 1
                    else:
                        # add photon to pmt bin in bottom array
                        pmt_signal_bot[ix, iy] += 1

                # get bin in x and y
                ix_fine = int((x[0] + offset) / dx_fine)
                iy_fine = int((x[1] + offset) / dy_fine)
                # if in range
                if ix_fine < npmt*ndivs and iy_fine < npmt*ndivs:
                    if x[2] > 0:
                        # add photon to fine bin in top array
                        fine_signal_top[ix_fine, iy_fine] += 1
                    else:
                        # add photon to fine bin in bottom array
                        fine_signal_bot[ix_fine, iy_fine] += 1

        event_data = {}
        event_data['number'] = self.ievent
        event_data['true_position'] = x0
        event_data['pmt_top'] = pmt_signal_top
        event_data['fine_top'] = fine_signal_top
        event_data['pmt_bot'] = pmt_signal_bot
        event_data['fine_bot'] = fine_signal_bot

        return event_data
    

    def generate_random_position(self):
        """Generates a random position within the detector volume.
        """
        # Generate random position
        phi = np.random.uniform(0, 2*np.pi)
        r = self.radius * np.sqrt(np.random.uniform(0, 1))
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = self.config['photon_zgen']

        return [x, y, z]
    
    def generate_batch(self):
        """Generates a batch of events. 
        """

        batch_size = self.config['nevents']
        self.open_file(self.config['filename']) # Open file for writing events to       

        for i in range(batch_size):
            if i%10 == 0:
                print('Generating event {} of {}'.format(i, batch_size))
            self.ievent = i
            # Generate event
            event = self.generate_event()
            # Write event to file
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
        """Writes an even+t to the file.
        """

        # Create group for event
        ev = 'event_{}'.format(self.ievent) 
        event_group = self.event_group.create_group(ev)

        # Write event data
        for key in event.keys():
            event_group[key] = event[key]

        return 0    
    
    def close_file(self):
        """Closes the file.
        """
        self.file.close()
        return 0

        