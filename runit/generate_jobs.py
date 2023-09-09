import pandas as pd
import os,sys, json

import argparse

# default configuration


#
# this is the directory where you run the jobs
#
run_dir = '/user/z37/OptoSim/runit'

#
# data directory base
#
data_base_dir = '/data/xenon/acolijn/optosim/data/'


def write_script(run_id, job_id, base_config_file):
    #
    # this is the directory where your scripts will live
    #
    script_dir = run_dir +'/scripts/'
    #
    # if the scripts directory does not exist.... make it
    #
    if not os.path.exists(script_dir):
        cmd = 'mkdir '+script_dir
        os.system(cmd)

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
    log_dir = run_dir +'/logs'
    #
    # if the logfile directory does not exist.... make it
    #
    if not os.path.exists(log_dir):
        cmd = 'mkdir '+log_dir
        os.system(cmd)

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
    parser.add_argument('--mcid', help='MC ID', default=0)
    parser.add_argument('--njobs', help='Number of jobs', default=10)
    args = parser.parse_args()

    mcid = int(args.mcid)
    njobs = int(args.njobs)    
    run_id = 'mc{:04}'.format(mcid)
    base_config_file = args.config

    data_dir = data_base_dir + run_id
    if not os.path.exists(data_dir):
        cmd = 'mkdir '+data_dir
        os.system(cmd)
    else:
        if args.force_write:
            print('data directory exists, but force_write is True, so continuing')
        else:
            print('data directory exists, exiting')
            sys.exit(0)

    # loop over the number of jobs
    print(njobs)
    for job_id in range(njobs):    
        write_script(run_id, job_id, base_config_file)

#
# main function
#
if __name__ == "__main__":
    main(sys.argv[1:])