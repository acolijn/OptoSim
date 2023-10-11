import configparser
import os

config = configparser.ConfigParser()


OPTOSIM_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(OPTOSIM_DIR)

config.read(os.path.join(OPTOSIM_DIR, "settings.ini"))

LOG_DIR = config.get("General", "log_dir")
DATA_DIR = config.get("General", "data_dir")
TMP_DIR = config.get("General", "tmp_dir")
MODEL_DIR = config.get("General", "model_dir")

CONFIG_DIR = os.path.join(OPTOSIM_DIR, "config")

# can I make DATA_TYPE_VERSION a float?
DATA_TYPE_VERSION = float(config.get("General", "data_type_version"))
