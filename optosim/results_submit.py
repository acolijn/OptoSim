from batch_stbc import submit_job
import os


from optosim.settings import OPTOSIM_DIR, PROJECT_DIR, LOG_DIR, DATA_DIR

make_results_file = os.path.join(OPTOSIM_DIR, "results_make.py")

jobstring = f"""

cd {PROJECT_DIR}
source venv_optosim/bin/activate

echo "Running job to make results"

# Run the job
python {make_results_file} 

echo "Finished"

"""

# Make log and jobname
log = os.path.join(LOG_DIR, f"job_results.log")
jobname = f"optosim_results"

# Submit the job
submit_job(
    jobstring,
    log=log,
    jobname=jobname,
    mem_per_cpu=8000,
    queue="short",
)

print(f"Submitted job")
