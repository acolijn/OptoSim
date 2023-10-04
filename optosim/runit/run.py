import argparse

import sys
sys.path.insert(0, "/user/z37/OptoSim/python/")

from Generator import *

def main(argv):
    """Main function for running the generator.

    """

    print('Running generator...')
    #
    # parse the arguments
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Configuration file', default='config.json')
    args = parser.parse_args()

    # read the configuration filename
    config_file = args.config
    print('Reading configuration from file: {}'.format(config_file))

    # initialize the generator
    gen = Generator(config=config_file)
    print('Initialized generator.')
    
    # generate events
    print('Generating events...')
    gen.generate_batch()

    print('Done.')
#
# main function
#
if __name__ == "__main__":
    main(sys.argv[1:])
