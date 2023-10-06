# OptoSim

OptoSim is a Python package for optical simulation of a TPC and for position reconstruction with machine learning. It provides a set of tools for simulating the propagation of light through a scintillator material and the subsequent detection of the scintillation light by a photodetector. The package also includes a set of machine learning models using super-resolution for reconstructing the position of the scintillation event from the detected light.

## Installation

Get your own copy of OptoSim by cloning the repository:

```bash
git clone git@github.com:acolijn/OptoSim.git
```

Move into the OptoSim directory:

```bash
cd OptoSim
```

You can install OptoSim using pip (development mode is recommended):

```bash
pip install -e ./ 
```

or preferably using the setup.sh script, that creates a virtual environment and installs all dependencies (this step is necessary to submit jobs to the cluster):

```bash
source setup.sh
```

Activate the virtual environment that was created:
```bash
source venv_optosim/bin/activate
```

## Structure

The OptoSim package contains two main modules: simulation and super_resolution. The simulation module contains the classes and functions for simulating the propagation of light through a scintillator material and the subsequent detection of the scintillation light by a photodetector in the TPC. The super_resolution module contains the classes and functions for reconstructing the position of the scintillation event from the detected light using machine learning models, including super_resolution models trained with the information of the optical simulation with a fine grid.

## Usage

### Settings

First, make sure that the paths specified in the settings.ini file are the desired ones. The settings.ini file contains the paths to the data, the output, the log files, the configuration files, etc.

### Simulation

To run a simulation, move to the optosim directory and do:

```bash
python simulation_run.py --run_id mc0001 --job_id 0 --config config_example.json 
```

this will run one batch (job_id=0) of the simulation for run mc0001, using the configuration file config_example.json in the config folder. The ouptup of the simulation will be stored in the output folder specified in the settings.ini file.

To submit multiple jobs to the cluster, each processing a different batch of the simulation, do:

```bash
python submit_simulation.py --run_id mc0001 --config config_example.json --n_jobs 10
```

this will submit 10 jobs to the cluster, each processing a different batch of the simulation for run mc0001, using the configuration file config_example.json in the config folder. The ouptup of the simulation will be stored in the data folder specified in the settings.ini file. The jobs will be submitted by default in the short queue on stoomboot. The job submission is handled by the batch_stbc.py module.  Note that in order to be able to submit jobs to the cluster, the setup.sh script must have been run before. Job submission will only work for users with access to the stoomboot cluster. 

### Super-resolution

To run a super-resolution model, move to the optosim directory and do:

```bash
python model_train.py --run_id mc0001 --pmts_per_dim 5 --nmax 1000
```

this will train a super-resolution model for run mc0001 (assuming that the data has been already produced with the previous commands), using the information of the optical simulation with a fine grid of 5x5 PMTs. The nmax argument sets the maximum number of events to read from the files, it deafults at 10 millions. The model will be stored in the model folder specified in the settings.ini file. The name of the model will be f'model_{pmts_per_dim}x{pmts_per_dim}_{run_id}.pkl'.

To submit multiple jobs to the cluster, each training a different super-resolution model, do:

```bash
python model_submit.py --run_id mc0001 --pmts_per_dim 5 10 15 --nmax 1000
```

in this case it is possible to specify multiple values for the pmts_per_dim parameter, and a different job will be submitted for each value. Note that in order to be able to submit jobs to the cluster, the setup.sh script must have been run before. Job submission will only work for users with access to the stoomboot cluster. 

Warning: running a model training with the same run_id and pmts_per_dim as a previous one will overwrite the previous model.

## Contributing
If you would like to contribute to this project, please follow these guidelines:

- Fork the repository.
- Create a new branch for your feature or bug fix.
- Write your code and tests.
- Run the tests using pytest.
- Submit a pull request.



