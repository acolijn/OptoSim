import pandas as pd
import os,sys, json

# default configuration
config = {
    "filename": "dummy.hd5f",
    "nevents": 10000,
    "nphoton_per_event": 10000,
    "photon_zgen": 0.1,
    "geometry":{
        "type": "cylinder",
        "radius": 2.5,
        "ztop": 1.0,
        "zliq": 0.0,
        "zbot": -5.0
    },
    "npmt_xy": 2,
    "pmt":{
        "type": "square",
        "size": 2.54,
        "ndivs": 10
    }
}

def write_script(id):


    #
    # this is the directory where you run teh jobs
    #
    run_dir = '/user/z37/OptoSim/runit'

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
    # copy the config file to the script directory
    #   
    config_file = "scripts/config{}.json".format(str(id))
    config["filename"] = "/data/xenon/acolijn/optosim/data/event{}.hd5f".format(str(id))
    # Write the dictionary to the JSON file
    with open(config_file, "w") as json_file:
        json.dump(config, json_file, indent=4)

    #
    # generate a shell script
    #

    print('start generation of job', id)
    scriptfile = script_dir+'/job'+str(id)+".sh"
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
def main():
    """Write job shell scripts and submit to the batch system
    """
    for id in range(10):    
        write_script(id)

if __name__ == "__main__":
    main()
