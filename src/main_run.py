import logging
import argparse
import json
import time
import torch
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
        logging.info('Testing: {}'.format(config['test']))
        data_path = config['data_path']
        data_utils = model_wrapper.DataUtils(data_path, **config['data_utils_params'])
        run = model_wrapper.Run
        writer = SummaryWriter(new_run_folder_path)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if config['model_params'].get('soft_start', False):
            config['model_params']['encoder_decoder_state_dict'] = torch.load(
                config['model_params']['encoder_decoder_state_dict'])

        run = run(
            writer=writer,
            model=model,
            data_utils=data_utils,
            device=device,
            run_folder_path=new_run_folder_path,
            do_validation=config['do_validation'],
            save_checkpoints=config.get('save_checkpoints', False),
            lr=config['lr'],
            optimizer=model_wrapper.Optimizer,
            mse=model_wrapper.MSE,
            ssim=model_wrapper.SSIM,
            do_persistence=config.get('do_persistence', False),
            nr_of_input_steps=config.get('data_utils_params', {}).get('nr_of_input_steps', 3),
            backprop_all_outputs=config.get('backprop_all_outputs', False),
            **config['model_params']
        )

        train_stamp1 = time.time()
        run.train()
        train_stamp2 = time.time()
        passed_training_time = train_stamp2 - train_stamp1
        logging.info('Runtime: {:.2f} seconds, {:.0f} minutes, {:.1f} hours'.format(passed_training_time,
                                                                                    passed_training_time / 60,
                                                                                    passed_training_time / 3600))

        if config['test']:
            test_stamp1 = time.time()
            run.test()
            test_stamp2 = time.time()
            passed_test_time = test_stamp2 - test_stamp1
            logging.info('Runtime: {:.2f} seconds, {:.0f} minutes, {:.1f} hours'.format(passed_test_time,
                                                                                        passed_test_time / 60,
                                                                                        passed_test_time / 3600))

        average_train_mse_loss = np.average(np.load("{}/arrays/train_mse_loss.npy".format(new_run_folder_path)))
        average_test_mse_loss = np.average(np.load("{}/arrays/test_mse_loss.npy".format(new_run_folder_path)))
        average_test_ssim_loss = np.average(np.load("{}/arrays/test_ssim_loss.npy".format(new_run_folder_path)))

        average_persistence_test_loss = -1
        if config.get('do_persistence', False):
            average_persistence_test_loss = np.average(np.load("{}/arrays/persistence_test_mse_loss.npy".format(new_run_folder_path)))

        logging.info("Config file path: {}".format(config_file_path))
        logging.info('Total runtime: {:.0f} minutes, number of parameters: {}, average training loss: {:.3f}, '
                     'average mse test loss: {:.3f}, average ssim test loss: {:.3f}, persistence loss: {:.3f}'.format(
                      (passed_training_time + passed_test_time) / 60,
                      run.count_parameters(),
                      average_train_mse_loss,
                      average_test_mse_loss,
                      average_test_ssim_loss,
                      average_persistence_test_loss))


if os.path.isdir("./upgraded-guacamole/run_configs/{}".format(args.configfile)):
    for file in os.listdir("./upgraded-guacamole/run_configs/{}".format(args.configfile)):
        if os.path.isdir("./upgraded-guacamole/run_configs/{}/{}".format(args.configfile, file)):
            for subfile in os.listdir("./upgraded-guacamole/run_configs/{}/{}".format(args.configfile, file)):
                if file != 'combinations':
                    run_experiment("./upgraded-guacamole/run_configs/{}/{}/{}".format(args.configfile, file, subfile),
                                   args.configfile + "/" + file)
        else:
            run_experiment("./upgraded-guacamole/run_configs/{}/{}".format(args.configfile, file), args.configfile)
else:
    model_directory = args.configfile
    model_directory = re.sub('\/(?:.(?!\/))+$', '', model_directory)
    run_experiment("./upgraded-guacamole/run_configs/{}".format(args.configfile), model_directory)
