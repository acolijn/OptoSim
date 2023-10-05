import configparser
import os

config = configparser.ConfigParser()


OPTOSIM_DIR = os.path.dirname(os.path.abspath(__file__))

config.read(os.path.join(OPTOSIM_DIR, 'settings.ini'))

LOG_DIR = config.get('General', 'log_dir')
DATA_DIR = config.get('General', 'data_dir')
TMP_DIR = config.get('General', 'tmp_dir')

CONFIG_DIR = os.path.join(OPTOSIM_DIR, 'config')