import argparse
import sys, os



def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_id', help='Run ID', required=True)
    parser.add_argument('--job_id', type=int, help='Job ID', required=True)
    parser.add_argument('--config', help='Configuration file', default='config.json')
    parser.add_argument('--force_write', help='Force write', default=False)
    return parser.parse_args()

def main():
    """
    Main function for running the generator.

    """

    args = parse_args()

    import optosim
    from optosim.settings import LOG_DIR, DATA_DIR, TMP_DIR

    # Parse the arguments
    
    # Create the directories if it doesn't exist
    for _dir in [LOG_DIR, DATA_DIR, TMP_DIR]:
        if not os.path.exists(_dir):
            os.makedirs(_dir)

    # TODO
    # Check if the run_id and job_id already exists
    # Manage force_write somehow
    

    # read the configuration filename
    config_file = args.config
    print('Reading configuration from file: {}'.format(config_file))

    run_id = f"{args.run_id}"
    job_id = args.job_id

    run_id_dir = os.path.join(DATA_DIR, run_id)

    if not os.path.exists(run_id_dir):
        os.makedirs(run_id_dir)

    filename = os.path.join(run_id_dir, f"{run_id}.{job_id:04}.hd5f")

    # initialize the generator
    gen = optosim.simulation.generator.Generator(filename, config=config_file)
    print('Initialized generator.')
    
    # generate events
    print('Generating events...')
    gen.generate_batch()

    print('Done.')

if __name__ == "__main__":
    main()
