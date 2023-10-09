import argparse
from batch_stbc import submit_job
import os


def parse_args():
    parser = argparse.ArgumentParser()

    # Arguments for the simulation
    parser.add_argument("--config", help="Configuration file", default="config_example.json")
    parser.add_argument("--run_id", help="Run ID", required=True)
    parser.add_argument("--njobs", help="Number of jobs", default=10, type=int)

    # Arguments for the batch system
    parser.add_argument("--mem_per_cpu", help="Memory per CPU", default=4000)
    parser.add_argument("--queue", help="Queue", default="short")

    args = parser.parse_args()

    return args


def main():
    # Parse the command line arguments
    args = parse_args()

    from optosim.settings import OPTOSIM_DIR, PROJECT_DIR, LOG_DIR, DATA_DIR

    run_mc_file = os.path.join(OPTOSIM_DIR, "simulation_run.py")

    # Check if the run ID already exists
    if os.path.exists(os.path.join(DATA_DIR,args.run_id)):
        raise ValueError(f"Run ID {args.run_id} already exists.")

    # Submit the jobs
    for i in range(args.njobs):
        # Make jobstring
        jobstring = f"""

        cd {PROJECT_DIR}
        source venv_optosim/bin/activate

        echo "Running job {i} for run {args.run_id} with config {args.config}"

        # Run the job
        python {run_mc_file} --run_id={args.run_id} --job_id={i} --config={args.config}

        echo "Finished"

        """

        log = os.path.join(LOG_DIR, f"job_sim_{args.run_id}_{i}.log")
        jobname = f"optosim_{args.run_id}_{i:04}"

        # Submit the job
        submit_job(
            jobstring,
            log=log,
            jobname=jobname,
            mem_per_cpu=args.mem_per_cpu,
            queue=args.queue,
        )

        print(f"Submitted job {i} for run {args.run_id} with config {args.config}")


if __name__ == "__main__":
    main()
