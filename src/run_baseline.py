import logging
import argparse
import json
import time
import importlib
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils.create_run_directory import create_run_directory
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument("configfile", help="Name of the config file.", type=str)
args = parser.parse_args()


def run_experiment(config_file_path, model_directory):
    with open(config_file_path, 'r') as fp:
        config = json.load(fp)
        model_wrapper = importlib.import_module("wrapper.{}_wrapper".format(config['model']))
        model = model_wrapper.Model
        run_name = config['run_name']
        new_run_folder_path = create_run_directory(model_directory, run_name)
        logging.basicConfig(filename=new_run_folder_path + '/log.log', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s -\t %(message)s')

        logging.info("Config file path: {}".format(config_file_path))
        data_path = config['data_path']
        data_utils = model_wrapper.DataUtils(data_path, **config['data_utils_params'])
        writer = SummaryWriter(new_run_folder_path)

        run = model(
            run_folder_path=new_run_folder_path,
            writer=writer,
            data_utils=data_utils,
            mse=model_wrapper.MSE,
            ssim=model_wrapper.SSIM,
            forecast_horizon=config['data_utils_params'].get('forecast_horizon', 1),
            model_params=config.get('model_params', None),
        )

        test_stamp1 = time.time()
        run.test()
        test_stamp2 = time.time()
        passed_training_time = test_stamp2 - test_stamp1
        logging.info('Runtime: {:.2f} seconds, {:.0f} minutes, {:.1f} hours'.format(passed_training_time,
                                                                                    passed_training_time / 60,
                                                                                    passed_training_time / 3600))

        average_test_mse_loss = np.average(np.load("{}/arrays/test_mse_loss.npy".format(new_run_folder_path)))
        average_test_ssim_loss = np.average(np.load("{}/arrays/test_ssim_loss.npy".format(new_run_folder_path)))

        logging.info('Total runtime: {:.0f} minutes, average test mse loss: {:.3f}, average test ssim loss: {:.3f}'.format(
            passed_training_time / 60,
            average_test_mse_loss,
            average_test_ssim_loss))


if os.path.isdir("./upgraded-guacamole/run_configs/{}".format(args.configfile)):
    for file in os.listdir("./upgraded-guacamole/run_configs/{}".format(args.configfile)):
        run_experiment("./upgraded-guacamole/run_configs/{}/{}".format(args.configfile, file), args.configfile)
else:
    model_directory = args.configfile
    model_directory = re.sub('\/(?:.(?!\/))+$', '', model_directory)
    run_experiment("./upgraded-guacamole/run_configs/{}".format(args.configfile), model_directory)
