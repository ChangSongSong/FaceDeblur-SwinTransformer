import os
import os.path as osp
import argparse
import yaml
from pprint import pprint
from trainer import Trainer


def main(config_path):
    # Read training configuration
    with open(config_path) as f:
        config = yaml.full_load(f)
        pprint(config)

    if(config['main']['mode'] == 'train'):
        trainer = Trainer()
        trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="path to configuration file")
    args = vars(parser.parse_args())
    main(args['config'])
