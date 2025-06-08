import sys
import os
from pathlib import Path
# import boto3 - AWS SDK for python provided by Amazon
import boto3
import yaml

# sys.path.append('src')

# Define path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE_PATH = os.path.join(PROJECT_ROOT, 'config.yml')
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
TRAINED_MODEL_PATH = os.path.join(PROJECT_ROOT, 'model')
TRAINING_MODEL_PATH = os.path.join(PROJECT_ROOT, 'training')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
DATA_RAW_PATH = os.path.join(DATA_PATH, 'raw')
DATA_PROCESSED_PATH = os.path.join(DATA_PATH, 'processed')
PIPELINE_PATH = os.path.join(SRC_PATH, 'pipeline')
ANALYSIS_PATH = os.path.join(SRC_PATH, 'analysis')
IMAGE_PATH = os.path.join(PROJECT_ROOT, 'image')


S3_LOCATION_CONSTRAINT = 'us-east-2'
S3_DATA_PATH = 'datool-data'
S3_DATA_RAW_PATH = 'raw/'
S3_DATA_PROCESSED_PATH = 'processed/'
S3_DATA_CRYPTO_PATH = 'crypto'


# Version control
house_price_version = 'v1'

def fetch_config_from_yaml(cfg_path: Path = CONFIG_FILE_PATH) -> yaml:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = yaml.load(conf_file, Loader=yaml.FullLoader)
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def update_yaml_config_file(config, cfg_path: Path = CONFIG_FILE_PATH):
    with open(cfg_path, 'w') as yamlfile:
        yaml.dump(config, yamlfile)
        st.write("Write successful")    


data = fetch_config_from_yaml()



