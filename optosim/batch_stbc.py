"""
Carlo Fuselli
cfuselli@nikhef.nl
-------------------

Module that handles job submission on stomboot, adapted from utilix sbatchq (XENON)

"""

import argparse
import os
import tempfile
import subprocess
import shlex


sbatch_template = """#!/bin/bash

#PBS -N {jobname}
#PBS -j oe
#PBS -o {log}
#PBS -l pmem={mem_per_cpu}

echo "starting script!"

{job}

echo "Script complete, bye!"

"""

TMPDIR = os.path.join(os.environ.get('user', '.'), 'tmp')

def submit_job(jobstring,
               log='job.log',
               jobname='somejob',
               sbatch_file=None,
               dry_run=False,
               mem_per_cpu=1000,
               cpus_per_task=1,
               hours=None,
               **kwargs
               ):
    
    """
    
    See XENONnT utilix function sbatcth for info
    
    :param jobstring: the command to execute
    :param log: where to store the log file of the job
    :param jobname: how to name this job
    :param sbatch_file: where to write the job script to
    :param dry_run: only print how the job looks like
    :param mem_per_cpu: mb requested for job
    :param container: name of the container to activate
    :param bind: which paths to add to the container
    :param cpus_per_task: cpus requested for job
    :param hours: max hours of a job
    :param kwargs: are ignored
    :return: None
    """
    

    sbatch_script = sbatch_template.format(jobname=jobname, 
                                           log=log, 
                                           qos=qos, 
                                           partition=partition,
                                           account=account, 
                                           job=jobstring, 
                                           mem_per_cpu=mem_per_cpu,
                                           cpus_per_task=cpus_per_task, 
                                           hours=hours
                                          )

    if dry_run:
        print("=== DRY RUN ===")
        print(sbatch_script)
        return

    if sbatch_file is None:
        remove_file = True
        _, sbatch_file = tempfile.mkstemp(suffix='.sh')
    else:
        remove_file = False

    with open(sbatch_file, 'w') as f:
        f.write(sbatch_script)

    command = "qsub %s" % sbatch_file
    if not sbatch_file:
        print("Executing: %s" % command)
    subprocess.Popen(shlex.split(command)).communicate()

    if remove_file:
        os.remove(sbatch_file)

  
    