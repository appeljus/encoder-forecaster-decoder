import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from run.run import Run
from utils.loss import SSIM
from utils.plot_utils import plot_two_columns_of_tensors
from statistics import mean
import logging


class RunEncoderDecoder(Run):
    def __init__(self,
                 writer,
                 model,
                 data_utils,
                 device,
                 run_folder_path,
                 do_validation=True,
                 lr=0.001,
                 save_checkpoints=True,
                 optimizer=optim.Adam,
                 mse=nn.MSELoss,
                 ssim=SSIM,
                 do_persistence=False,
                 nr_of_input_steps=3,
                 backprop_all_outputs=False,
                 **model_params):

        self.net = model(**model_params)

        super().__init__(writer,
                         model,
                         data_utils,
                         device,
                         run_folder_path,
                         do_validation,
                         lr,
                         optimizer,
                         mse,
                         ssim,
                         save_checkpoints,
                         do_persistence,
                         nr_of_input_steps,
                         **model_params)

    def train(self):
        logging.info('Started training')
        for i in range(self.estimate_total_nr_train_data_points):
            # Step
            batch, loss, outputs = self.train_step()

            # Outliers
            if loss.item() > 3 * mean(self.train_mse_loss_array):
                self.plot_outlier(i, batch, outputs, loss)

            # Validation
            if self.do_validation:
                if i > self.when_validate:
                    self.validation_step(i)

            # Checkpoint
            if i % int(self.estimate_total_nr_train_data_points / 100) == 0:
                self.checkpoint(i, batch, outputs, 'train')

        self.checkpoint(i, batch, outputs, 'train')

        # Save state dict
        torch.save(self.net.state_dict(), '{0}/checkpoints/state_dict.pth'.format(self.run_folder_path))

        # Save train and validation loss arrays
        np.save('{}/arrays/train_mse_loss'.format(self.run_folder_path), self.train_mse_loss_array)
        np.save('{}/arrays/train_ssim_loss'.format(self.run_folder_path), self.train_ssim_loss_array)
        if self.do_validation:
            np.save('{}/arrays/validation_loss'.format(self.run_folder_path), self.validation_loss_array)

        logging.info('Finished training')

    def test(self, state_dict=None):
        with torch.no_grad():
            logging.info('Started testing')
            if state_dict is not None:
                self.net = self.model(self.model_params)
                self.net = self.net.to(self.device)
                self.net.load_state_dict(state_dict)
            self.net.eval()

            for i in range(self.estimate_total_nr_test_data_points):
                batch, outputs = self.test_step()

                if i % int(self.estimate_total_nr_test_data_points / 100) == 0:
                    self.checkpoint(i, batch, outputs, 'test')

            np.save('{0}/arrays/test_mse_loss'.format(self.run_folder_path), self.test_mse_loss_array)
            np.save('{0}/arrays/test_ssim_loss'.format(self.run_folder_path), self.test_ssim_loss_array)
            logging.info('Finished testing')

    def step(self, batch):
        batch = batch.to(self.device)
        outputs = self.net(batch)

        outputs = outputs.squeeze()
        ground_truth = batch.squeeze()
        mse_loss = self.mse(outputs, ground_truth)
        ssim_loss = self.ssim(outputs, ground_truth)

        return outputs, mse_loss, ssim_loss

    def train_step(self):
        self.optimizer.zero_grad()
        batch = self.data_utils.get_next_train_data_point()
        outputs, mse_loss, ssim_loss = self.step(batch)
        mse_loss.backward()
        self.optimizer.step()
        self.train_mse_loss_array.append(mse_loss.item())
        self.running_train_mse_loss += mse_loss.item()
        self.train_ssim_loss_array.append(ssim_loss.item())

        return batch, mse_loss, outputs

    def test_step(self):
        batch = self.data_utils.get_next_test_data_point()
        outputs, mse_loss, ssim_loss = self.step(batch)
        self.test_mse_loss_array.append(mse_loss.item())
        self.running_test_mse_loss += mse_loss.item()
        self.test_ssim_loss_array.append(ssim_loss.item())

        return batch, outputs

    def validation_step(self, i):
        batch = self.data_utils.get_next_validation_data_point()
        outputs, mse_loss, _ = self.step(batch)
        self.running_validation_loss += mse_loss.item()
        self.validation_loss_array.append([i, mse_loss.item()])
        self.when_validate += self.validation_train_ratio

    def checkpoint(self, i, batch, outputs, case):
        estimate_total_nr_data_points = getattr(self, "estimate_total_nr_{}_data_points".format(case))
        running_loss = getattr(self, "running_{}_mse_loss".format(case))

        logging.info('Epoch: {}/{}, {:.0%}'.format(i, estimate_total_nr_data_points, (i / estimate_total_nr_data_points)))
        plot_two_columns_of_tensors(outputs, batch.squeeze(), i, '{}/plots/{}'.format(self.run_folder_path, case))

        if i != 0:
            self.writer.add_scalar('{}_loss'.format(case), running_loss / i, i)
            if case == 'train' and self.do_validation:
                self.writer.add_scalar('validation_loss', self.running_validation_loss / len(self.validation_loss_array), i)
            self.writer.close()

        if case == 'train' and self.save_checkpoints:
            save_dict = {
                'epoch': i,
                'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss_array': self.train_mse_loss_array,
            }
            if self.do_validation:
                save_dict['validation_loss_array'] = self.validation_loss_array

            torch.save(save_dict, '{0}/checkpoints/{1}.pth'.format(self.run_folder_path, i))

    def plot_outlier(self, i, batch, outputs, loss):
        logging.warning(
            "Higher than average loss for index {}: {}, average loss: {}".format(i, loss.item(),
                                                                                 mean(self.train_mse_loss_array)))
        plot_two_columns_of_tensors(outputs, batch.squeeze(), i, self.run_folder_path + '/plots/outliers')
