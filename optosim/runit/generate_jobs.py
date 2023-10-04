import pandas as pd
import os,sys, json

import argparse

#
# this is the directory where you run the jobs
#
run_dir = '/user/z37/OptoSim/runit'
#
# directory where the scripts live
#
script_dir = run_dir +'/scripts/'

#
# data directory base
#
data_base_dir = '/data/xenon/acolijn/optosim/data/'
#
# directory where the logfiles live
#
log_dir = data_base_dir +'/logs/'

def write_config(run_id, job_id, base_config_file):
    """Write the configuration file for a job
    """
    #
    # read the base configuration file
    #
    with open(base_config_file, 'r') as file:
        config = json.load(file)
    #
    # copy the config file to the script directory
    #   
    config_file = "scripts/{}.config{:04}.json".format(run_id,job_id)

    # modify the filename
    data_dir = data_base_dir + run_id
    config["filename"] = data_dir + "/{}.{:04}.hd5f".format(run_id, job_id)
    # Write the dictionary to the JSON file
    with open(config_file, "w") as json_file:
        json.dump(config, json_file, indent=4)  

    return config_file

def check_directories(run_id, base_config_file, force_write):
    """Check if the directories exist, and if not, create them	
    """

    #
    # if the scripts directory does not exist.... make it
    #
    if not os.path.exists(script_dir):
        cmd = 'mkdir '+script_dir
        os.system(cmd)
    #
    # if the logfile directory does not exist.... make it
    #
    if not os.path.exists(log_dir):
        cmd = 'mkdir '+log_dir
        os.system(cmd)
    #
    # check if the data directory exists
    #
    data_dir = data_base_dir + run_id
    if not os.path.exists(data_dir):
        # if it does not exist, create it
        cmd = 'mkdir '+data_dir
        os.system(cmd)
    else:
        # if it does exist, check if force_write is True
        if force_write:
            # if it does exist, and force_write is True, continue
            print('data directory exists, but force_write is True, so continuing')
        else:
            # if it does exist, and force_write is False, exit
            print('data directory exists, exiting')
            sys.exit(0)


def write_script(run_id, job_id, config_file):
    """Write a job shell script and submit to the batch system
    """
    #
    # generate a shell script
    #
    print('start generation of job', job_id)
    scriptfile = script_dir+'/{}.job{:04}.sh'.format(run_id,job_id)
    fout = open(scriptfile,"w")
    fout.write("#/bin/sh \n")
    fout.write("cd " + run_dir + " \n")
    fout.write("python run.py --config={}".format(str(config_file))+"\n")
    fout.close()
    #
    # execute the job
    #
    os.system('chmod +x '+scriptfile)
    os.system('qsub -e '+log_dir+' -o '+log_dir+' '+scriptfile)

#
# main function
#
def main(argv):
    """Write job shell scripts and submit to the batch system
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Configuration file', default='config.json')
    parser.add_argument('--force_write', help='Force write to existing data directory', default=False)
    parser.add_argument('--mcid', help='MC ID', default=-1)
    parser.add_argument('--njobs', help='Number of jobs', default=10)
    args = parser.parse_args()

    mcid = int(args.mcid)
    njobs = int(args.njobs)    
    run_id = 'mc{:04}'.format(mcid)
    base_config_file = args.config
    force_write = args.force_write
    #
    # check directories for scripts and data
    #
    check_directories(run_id, base_config_file, force_write)
    #
    # loop over the number of jobs
    #
    for job_id in range(njobs):    
        #
        # create config .json file
        #
        config_file = write_config(run_id, job_id, base_config_file)

        #
        # write the job script
        #
        write_script(run_id, job_id, config_file)

#
# main function
#
if __name__ == "__main__":
    main(sys.argv[1:])