"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import yaml
from easydict import EasyDict
from utils import mkdir_if_missing

def create_config(config_file_env):
    # Config for environment path
    with open(config_file_env, 'r') as stream:
            config = yaml.safe_load(stream)
   
    cfg = EasyDict()
   
    # Copy
    for k, v in config.items():
        cfg[k] = v
    root_dir = cfg['root_dir']
    # Set paths for pretext task (These directories are needed in every stage)
    base_dir = os.path.join(root_dir, cfg['train_db_name'])
    mkdir_if_missing(base_dir)
    cfg['dataset_dir'] = os.path.join(root_dir, 'Datasets')
    mkdir_if_missing(cfg['dataset_dir'])

    cfg['checkpoint'] = os.path.join(base_dir, 'checkpoint.pth.tar')
    cfg['progress_path'] = os.path.join(base_dir, 'progress')
    mkdir_if_missing(cfg['progress_path'])
    cfg['embeddings_path'] = os.path.join(base_dir, 'embeddings')
    mkdir_if_missing(cfg['embeddings_path'])
    print(cfg['checkpoint'])

    return cfg 
