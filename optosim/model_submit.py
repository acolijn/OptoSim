import argparse
from batch_stbc import submit_job
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Train a neural network to do super resolution")
    parser.add_argument("--run_id", help="Run ID", required=True)
    parser.add_argument("--nmax", help="Maximum number of events to read", default=10_000_000)
    parser.add_argument("--mem_per_cpu", help="Memory per CPU", default=8000)
    parser.add_argument("--queue", help="Queue", default="generic")
    parser.add_argument(
        "--pmts_per_dim",
        help="""
                        Number of PMTs per dimension. 
                        It allows to specify a list of values, e.g. --pmts_per_dim 5 10 15. 
                        One job will be submitted for each value.""",
        nargs="*",
        default=[
            5,
        ],
    )

    args = parser.parse_args()

    return args


def main():
    # Parse the command line arguments
    args = parse_args()

    from optosim.settings import OPTOSIM_DIR, PROJECT_DIR, LOG_DIR

    model_train_file = os.path.join(OPTOSIM_DIR, "model_train.py")

    for i_pmts_per_dim in args.pmts_per_dim:
        # Make jobstring
        jobstring = f"""

        cd {PROJECT_DIR}
        source venv_optosim/bin/activate

        echo "Running job with {i_pmts_per_dim} pmts for run {args.run_id}"

        # Run the job
        python {model_train_file} --run_id={args.run_id} --nmax={args.nmax} --pmts_per_dim={i_pmts_per_dim}

        echo "Finished"

        """

        # Submit the job
        log = os.path.join(LOG_DIR, f"job_train_{args.run_id}_{i_pmts_per_dim}pmt.log")
        jobname = f"optosim_train_{args.run_id}_{i_pmts_per_dim}pmt"

        # Submit the job
        submit_job(
            jobstring,
            log=log,
            jobname=jobname,
            mem_per_cpu=args.mem_per_cpu,
            queue=args.queue,
        )

        print(f"Submitted job with {i_pmts_per_dim}pmts for run {args.run_id}")


if __name__ == "__main__":
    main()
