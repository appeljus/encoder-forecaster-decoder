import argparse
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("configfile", help="Name of the config file.", type=str)
args = parser.parse_args()

config_file_path = "./upgraded-guacamole/run_configs/{}".format(args.configfile)
with open(config_file_path, 'r') as fp:
    config = json.load(fp)
    state_dict_path = config['model_params']['encoder_decoder_state_dict']
    if not os.path.exists(state_dict_path):
        sys.exit('State dict does not exist: {}'.format(state_dict_path))
    else:
        print('State dict exists!')
