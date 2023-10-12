import numpy as np
import json
import os
import h5py

import optosim

from optosim.settings import DATA_DIR, OPTOSIM_DIR, CONFIG_DIR, DATA_TYPE_VERSION
import optosim.simulation.optical_photon as op


class Generator:
    """Class for generating S2-like events. Photons are generated from positions as
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

    2. Data structure.
    
    If DATA_TYPE_VERSION = 1.0
    
    The per event data is stored in a dictionary with the following structure:
        {
            'number': event number,
            'nphoton': number of photons generated in event,
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

    If DATA_TYPE_VERSION = 2.0

    
    """

    def __init__(self, filename, config_file, **kwargs):
        """Initializes the generator class. The configuration is read from the config file.

        Parameters
        ----------
        filename : str
            Filename of output file. Needs to be an absolute path ending with .hdf5
        config_file : str
            Filename of configuration file. For example: 'config_example.json'.

        """

        # Read configuration file
        self.config_file = os.path.join(CONFIG_DIR, config_file)

        if os.path.isfile(self.config_file):
            print("Generator::Reading configuration from file: {}".format(self.config_file))
            self.config = json.load(open(self.config_file, "r"))
        else:
            raise ValueError("Config file does not exist. {}".format(self.config_file))

        # Check filename extension
        if not filename.endswith(".hd5"):
            raise ValueError("Filename needs to end with .hd5")

        # Check if the path is not a relative path
        if not os.path.isabs(filename):
            raise ValueError("Filename needs to be an absolute path.")

        self.filename = filename

        # Initialize the detector
        self.radius = self.config["geometry"]["radius"]

        # Get nphoton range
        self.nph_range = self.config["nphoton_per_event"]
        self.log0 = np.log10(self.nph_range[0])
        self.log1 = np.log10(self.nph_range[1])

        self.ievent = 0  # Event counter

        # define an optical photon
        self.aPhoton = op.OpticalPhoton(config_file=self.config_file)

        print("Generator::Initialized generator.")
        print("Generator::Filename: {}".format(self.filename))	
        print("Generator::Data type version: {}".format(DATA_TYPE_VERSION))


    def initialize_file(self):
        """Initializes the file for writing events to."""

        # Write configuration to file
        self.config["data_type_version"] = DATA_TYPE_VERSION
        self.file.attrs["config"] = json.dumps(self.config)

        if DATA_TYPE_VERSION == 1.0:
            # inefficient data structure that we started teh project with
            print("Using data type version 1.0")
        elif DATA_TYPE_VERSION == 2.0:
            # more efficient data structure
            print("Using data type version 2.0")
            total_events = self.config['nevents']
            self.events_dset = {}

             # Create 'events' group
            events_group = {}
            if "events" not in self.file:
                events_group = self.file.create_group("events")
            else:
                events_group = self.file["events"]

            self.events_dset["number"] = events_group.create_dataset("number", (total_events,), dtype=np.int32)
            self.events_dset["nphoton"] = events_group.create_dataset("nphoton", (total_events,), dtype=np.float32)
            self.events_dset["true_position"] = events_group.create_dataset("true_position", (total_events, 3), dtype=np.float32)
            self.events_dset["fine_top"] = events_group.create_dataset("fine_top", (total_events, self.config["npmt_xy"]*self.config["pmt"]["ndivs"], self.config["npmt_xy"]*self.config["pmt"]["ndivs"]), dtype=np.int32) 
            self.events_dset["pmt_top"] = events_group.create_dataset("pmt_top", (total_events, self.config["npmt_xy"], self.config["npmt_xy"]), dtype=np.int32)
            self.events_dset["fine_bot"] = events_group.create_dataset("fine_bot", (total_events, self.config["npmt_xy"]*self.config["pmt"]["ndivs"], self.config["npmt_xy"]*self.config["pmt"]["ndivs"]), dtype=np.int32) 
            self.events_dset["pmt_bot"] = events_group.create_dataset("pmt_bot", (total_events, self.config["npmt_xy"], self.config["npmt_xy"]), dtype=np.int32)
        else:
            print("WTF")
            raise ValueError("Unknown data type version: {}".format(DATA_TYPE_VERSION)) 

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

        Returns
        -------
        event_data : dict
            Dictionary containing the event data.

        A.P. Colijn 2023
        """
        # Generate random position
        x0 = self.generate_random_position()

        # Get detector parameters
        offset = self.config["npmt_xy"] * self.config["pmt"]["size"] / 2
        npmt = self.config["npmt_xy"]
        nfine = self.config["npmt_xy"] * self.config["pmt"]["ndivs"]
        dx = self.config["pmt"]["size"]
        dy = self.config["pmt"]["size"]

        ndivs = self.config["pmt"]["ndivs"]
        dx_fine = dx / ndivs
        dy_fine = dy / ndivs
        # Initialize pmt signal
        pmt_signal_top = np.zeros((npmt, npmt), dtype=np.int32)
        pmt_signal_bot = np.zeros((npmt, npmt), dtype=np.int32)

        # Initialize fine signal
        fine_signal_top = np.zeros((nfine, nfine), dtype=np.int32)
        fine_signal_bot = np.zeros((nfine, nfine), dtype=np.int32)

        # Generate photons
        nphoton = 0
        if len(self.nph_range) == 1:
            nphoton = self.nph_range[0]
        else:
            # Generate random number of photons flat in log space
            logran = np.random.uniform(self.log0, self.log1)
            nphoton = int(10 ** logran)

        # Loop over photons
        for _ in range(nphoton):
            # Generate photon at position x0
            self.aPhoton.generate_photon(x0)
            # Propagate photon.... this is where the magic happens
            self.aPhoton.propagate()
            # Check if photon is detected
            if self.aPhoton.is_detected():
                # get photon position
                x = self.aPhoton.get_photon_position()
                # get bin in x and y. make sure bin0 does not get overpopulated.....
                if (x[0] + offset >= 0) and (x[1] + offset >= 0):
                    ix = int((x[0] + offset) / dx)
                    iy = int((x[1] + offset) / dy)
                    # if in range
                    if (0 <= ix < npmt) and (0 <= iy < npmt):
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
                    # if ix_fine < npmt*ndivs and iy_fine < npmt*ndivs:
                    if (0 <= ix_fine < npmt * ndivs) and (0 <= iy_fine < npmt * ndivs):
                        if x[2] > 0:
                            # add photon to fine bin in top array
                            fine_signal_top[ix_fine, iy_fine] += 1
                        else:
                            # add photon to fine bin in bottom array
                            fine_signal_bot[ix_fine, iy_fine] += 1

        event_data = {}
        event_data["number"] = self.ievent
        event_data["nphoton"] = nphoton
        event_data["true_position"] = x0
        event_data["pmt_top"] = pmt_signal_top
        event_data["fine_top"] = fine_signal_top
        event_data["pmt_bot"] = pmt_signal_bot
        event_data["fine_bot"] = fine_signal_bot

        return event_data

    def generate_random_position(self):
        """Generates a random position within the detector volume."""
        # Generate random position
        phi = np.random.uniform(0, 2 * np.pi)
        r = self.radius * np.sqrt(np.random.uniform(0, 1))
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = self.config["photon_zgen"]

        return [x, y, z]

    def generate_batch(self):
        """Generates a batch of events."""

        batch_size = self.config["nevents"]
        # Open file
        self.open_file(self.filename)
        self.initialize_file()

        for i in range(batch_size):
            if i % 10 == 0:
                print("Generating event {} of {}".format(i, batch_size))
            self.ievent = i
            # Generate event
            event = self.generate_event()
            # Write event to file
            self.write_event(event)  # Write event to file

        self.close_file()  # Close file

        return 0

    def open_file(self, filename):
        """Opens a file for writing events to."""
        # Check if file exists
        # if os.path.isfile(filename):
        #    raise ValueError("File already exists.")

        # Open file
        self.file = h5py.File(filename, "w")


        return 0

    def write_event(self, event):
        """Writes an event to the file."""

        # Create group for event
        if DATA_TYPE_VERSION == 1.0:
            ev = "event_{}".format(self.ievent)
            event_group = self.event_group.create_group(ev)

            # Write event data
            for key in event.keys():
                event_group[key] = event[key]
        elif DATA_TYPE_VERSION == 2.0:
            idx = self.ievent  # Assume you have initialized this to 0 and update it every time you write an event
    
            self.events_dset["number"][idx] = event["number"]
            self.events_dset["nphoton"][idx] = event["nphoton"]
            self.events_dset["true_position"][idx] = event["true_position"]
            self.events_dset["fine_top"][idx] = event["fine_top"]
            self.events_dset["pmt_top"][idx] = event["pmt_top"] 
            self.events_dset["fine_bot"][idx] = event["fine_bot"]   
            self.events_dset["pmt_bot"][idx] = event["pmt_bot"]

        else:
            raise ValueError("Unknown data type version: {}".format(DATA_TYPE_VERSION))

        return 0

    def close_file(self):
        """Closes the file."""
        self.file.close()
        return 0
