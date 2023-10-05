import argparse
from batch_stbc import submit_job
import os

from optosim.settings import OPTOSIM_DIR
run_mc_file = os.path.join(OPTOSIM_DIR, 'simulation_run.py')


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Configuration file', default='config_example.json')
    parser.add_argument('--force_write', help='Force write to existing data directory', default=False)
    parser.add_argument('--run_id', help='Run ID', default=-1)
    parser.add_argument('--njobs', help='Number of jobs', default=10)
    args = parser.parse_args()

    return args

def main():

    # Parse the command line arguments
    args = parse_args()

    for i in range(args.njobs):

        # Make jobstring 
        jobstring = f"""

        echo "Running job {i} for run {args.run_id} with config {args.config}"

        # Run the job
        {run_mc_file} --run_id={args.run_id} --job_id={i} --config={args.config} --force_write={args.force_write}

        echo "Finished"

        """

if __name__ == '__main__':
    main()