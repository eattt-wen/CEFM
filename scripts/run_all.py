import sys
import os
root_dir = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.insert(0, root_dir)

import yaml
from training.train_contrastive import train_projection_head_stage
from training.train_classifiers import train_classifiers_stage

if __name__ == '__main__':
    cfg_path = os.path.join(root_dir, 'configs', 'default.yaml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    train_projection_head_stage(cfg)
    train_classifiers_stage(cfg)
