import torch
from run.run import Run
from utils.loss import SSIM
from utils.plot_utils import plot_two_columns_of_tensors, plot_three_columns_of_tensors, plot_two_columns_of_tensors_with_input
from statistics import mean
import logging
import torch.optim as optim
import torch.nn as nn
import numpy as np


class RunRecurrent(Run):
    def __init__(self,
                 writer,
                 model,
                 data_utils,
                 device,
                 run_folder_path,
                 do_validation=True,
                 lr=0.001,
                 save_checkpoints=False,
                 optimizer=optim.Adam,
                 mse=nn.MSELoss,
                 ssim=SSIM,
                 do_persistence=False,
                 nr_of_input_steps=3,
                 backprop_all_outputs=False,
                 **model_params):

        self.backprop_all_outputs = backprop_all_outputs
        logging.info('RunRecurrent - backprop on all timesteps: {}'.format(backprop_all_outputs))
        self.net = model(device=device,
                         do_persistence=do_persistence,
                         backprop_all_outputs=backprop_all_outputs,
                         **model_params)

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

        self.batch = None
        if self.do_persistence:
            self.persistence_train_loss_array = []
            self.persistence_test_loss_array = []
            self.running_persistence_train_loss = 0.0
            self.running_persistence_test_loss = 0.0

    def train(self):
        logging.info('Started training')
        self.net.train()

        if self.do_persistence:
            train_step_func = self.train_step_with_persistence
            validation_step_func = self.validate_step_with_persistence
        else:
            train_step_func = self.train_step
            validation_step_func = self.validate_step
        for i in range(self.estimate_total_nr_train_data_points):
            # Step
            batch, ground_truth, loss, outputs, persistence_output = train_step_func()

            # Outliers
            if loss.item() > 3 * mean(self.train_mse_loss_array):
                self.plot_outlier(ground_truth, i, loss, outputs, 'train')

            if self.persistence_train_loss_array[-1] > 3 * mean(self.persistence_train_loss_array):
                plot_three_columns_of_tensors(outputs, ground_truth, persistence_output, i, self.run_folder_path + '/plots/persistence/{}'.format('train_outliers'))

            # Specific outliers
            if "1h" in self.run_folder_path:
                if 42450 < i < 43000:
                    self.plot_specific_outliers(ground_truth, i, loss, outputs, batch, 'train')
                elif 86200 < i < 86700:
                    self.plot_specific_outliers(ground_truth, i, loss, outputs, batch, 'train')

            # Validation
            if self.do_validation:
                if i > self.when_validate:
                    validation_step_func(i)

            # Checkpoint
            if i % int(self.estimate_total_nr_train_data_points / 100) == 0:
                self.checkpoint(ground_truth, outputs, persistence_output, i, 'train')

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.25)
            # for p in self.net.parameters():
            #     p.data.add_(-self.lr, p.grad.data)

        self.checkpoint(ground_truth, outputs, persistence_output, i, 'train')

        # Save state dict
        torch.save(self.net.state_dict(), '{0}/checkpoints/state_dict.pth'.format(self.run_folder_path))

        # Save train, validation, and persistence loss arrays
        np.save('{}/arrays/train_mse_loss'.format(self.run_folder_path), self.train_mse_loss_array)
        np.save('{}/arrays/train_ssim_loss'.format(self.run_folder_path), self.train_ssim_loss_array)
        if self.do_validation:
            np.save('{}/arrays/validation_mse_loss'.format(self.run_folder_path), self.validation_loss_array)
        if self.do_persistence:
            np.save('{}/arrays/persistence_train_mse_loss'.format(self.run_folder_path), self.persistence_train_loss_array)

        logging.info('Finished training')

    def plot_specific_outliers(self,  ground_truth, i, loss, outputs, batch, case):
        loss_array = getattr(self, '{}_mse_loss_array'.format(case))
        logging.info(
            "Plotting specific outliers for index {} with loss: {}, average loss: {}".format(i, loss.item(), mean(loss_array)))
        plot_two_columns_of_tensors_with_input(outputs, ground_truth, batch, i, '{}/plots/outliers/{}'.format(self.run_folder_path, case))

    def plot_outlier(self, ground_truth, i, loss, outputs, case):
        loss_array = getattr(self, '{}_mse_loss_array'.format(case))
        logging.warning(
            "Higher than average loss for index {}: {}, average loss: {}".format(i, loss.item(), mean(loss_array)))
        plot_two_columns_of_tensors(outputs, ground_truth, i, '{}/plots/outliers/{}'.format(self.run_folder_path, case))

    def test(self, state_dict=None):
        with torch.no_grad():
            logging.info('Started testing')
            if state_dict is not None:
                self.net = self.model(self.model_params)
                self.net = self.net.to(self.device)
                self.net.load_state_dict(state_dict)
            self.net.eval()
            self.optimizer.zero_grad()

            if self.do_persistence:
                test_step_func = self.test_step_with_persistence
            else:
                test_step_func = self.test_step

            for i in range(self.estimate_total_nr_test_data_points):
                # Step
                ground_truth, loss, outputs, persistence_output = test_step_func()

                # Outliers
                if loss.item() > 3 * mean(self.test_mse_loss_array):
                    self.plot_outlier(ground_truth, i, loss, outputs, 'test')

                # Checkpoint
                if i % int(self.estimate_total_nr_test_data_points / 100) == 0:
                    self.checkpoint(ground_truth, outputs, persistence_output, i, 'test')

            np.save('{0}/arrays/test_mse_loss'.format(self.run_folder_path), self.test_mse_loss_array)
            np.save('{0}/arrays/test_ssim_loss'.format(self.run_folder_path), self.test_ssim_loss_array)
            if self.do_persistence:
                np.save('{0}/arrays/persistence_test_mse_loss'.format(self.run_folder_path), self.persistence_test_loss_array)
            logging.info('Finished testing')

    def checkpoint(self, ground_truth, outputs, persistence_output, i, case):
        estimate_total_nr_data_points = getattr(self, "estimate_total_nr_{}_data_points".format(case))
        running_loss = getattr(self, "running_{}_mse_loss".format(case))

        logging.info('Epoch: {}/{}, {:.0%}'.format(i, estimate_total_nr_data_points, (i / estimate_total_nr_data_points)))

        plot_two_columns_of_tensors(outputs, ground_truth, i, self.run_folder_path + '/plots/{}'.format(case))

        if self.do_persistence:
            plot_three_columns_of_tensors(outputs, ground_truth, persistence_output, i, self.run_folder_path + '/plots/persistence/{}'.format(case))

        if i != 0:
            self.writer.add_scalar('{}_loss'.format(case), running_loss / i, i)
            self.writer.close()

            if self.do_persistence:
                running_persistence_loss = getattr(self, "running_persistence_{}_loss".format(case))
                self.writer.add_scalar('persistence_{}_loss'.format(case), running_persistence_loss / i, i)
                self.writer.close()

            if case == 'train' and self.do_validation:
                self.writer.add_scalar('validation_loss', self.running_validation_loss / len(self.validation_loss_array), i)
                self.writer.close()

        if case == 'train' and self.save_checkpoints:
            # Save state
            save_dict = {
                'epoch': i,
                'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss_array': self.train_mse_loss_array,
            }
            if self.do_validation:
                save_dict['validation_loss_array'] = self.validation_loss_array
            torch.save(save_dict, '{0}/checkpoints/{1}.pth'.format(self.run_folder_path, i))

    def train_step(self):
        batch, ground_truth = self.data_utils.get_next_train_data_point()
        _, ground_truth, mse_loss, ssim_loss, outputs = self.step(batch, ground_truth)
        self.optimizer.zero_grad()
        mse_loss.backward()
        self.train_mse_loss_array.append(mse_loss.item())
        self.running_train_mse_loss += mse_loss.item()
        self.train_ssim_loss_array.append(ssim_loss.item())
        self.optimizer.step()

        return None, ground_truth, mse_loss, outputs, None

    def train_step_with_persistence(self):
        batch, ground_truth = self.data_utils.get_next_train_data_point()
        batch, ground_truth, mse_loss, ssim_loss, persistence_loss, outputs, persistence_output = self.step_with_persistence(batch, ground_truth)

        self.optimizer.zero_grad()
        mse_loss.backward()
        self.train_mse_loss_array.append(mse_loss.item())
        self.running_train_mse_loss += mse_loss.item()
        self.persistence_train_loss_array.append(persistence_loss.item())
        self.running_persistence_train_loss += persistence_loss.item()
        self.train_ssim_loss_array.append(ssim_loss.item())
        self.optimizer.step()

        return batch, ground_truth, mse_loss, outputs, persistence_output

    def test_step(self):
        batch, ground_truth = self.data_utils.get_next_test_data_point()
        _, ground_truth, mse_loss, ssim_loss, outputs = self.step(batch, ground_truth)
        self.test_mse_loss_array.append(mse_loss.item())
        self.test_ssim_loss_array.append(ssim_loss.item())
        self.running_test_mse_loss += mse_loss.item()

        return ground_truth, mse_loss, outputs, None

    def test_step_with_persistence(self):
        batch, ground_truth = self.data_utils.get_next_test_data_point()
        _, ground_truth, mse_loss, ssim_loss, persistence_loss, outputs, persistence_output = self.step_with_persistence(batch, ground_truth)

        self.test_mse_loss_array.append(mse_loss.item())
        self.running_test_mse_loss += mse_loss.item()
        self.persistence_test_loss_array.append(persistence_loss.item())
        self.running_persistence_test_loss += persistence_loss.item()
        self.test_ssim_loss_array.append(ssim_loss.item())

        return ground_truth, mse_loss, outputs, persistence_output

    def validate_step(self, i):
        batch, ground_truth = self.data_utils.get_next_validation_data_point()
        _, _, mse_loss, _, _ = self.step(batch, ground_truth)
        self.running_validation_loss += mse_loss.item()
        self.validation_loss_array.append([i, mse_loss.item()])
        self.when_validate += self.validation_train_ratio

    def validate_step_with_persistence(self, i):
        batch, ground_truth = self.data_utils.get_next_validation_data_point()
        _, _, mse_loss, _, _, _, _ = self.step_with_persistence(batch, ground_truth)
        self.running_validation_loss += mse_loss.item()
        self.validation_loss_array.append([i, mse_loss.item()])
        self.when_validate += self.validation_train_ratio

    def step(self, batch, ground_truth):
        batch = batch.to(self.device)
        ground_truth = ground_truth.to(self.device)
        batch_input_images = {}
        for i in range(self.nr_of_input_steps):
            batch_input_images['x_{}'.format(i)] = batch[:, i]
        outputs = self.net(**batch_input_images)

        outputs = outputs.squeeze()
        if self.backprop_all_outputs:
            ground_truth = ground_truth.unsqueeze(0)
            ground_truth = torch.cat([batch[:, 1:], ground_truth], dim=1)
            ground_truth = ground_truth.squeeze()
        else:
            ground_truth = ground_truth.to(self.device)
            ground_truth = ground_truth.squeeze()

        mse_loss = self.mse(outputs, ground_truth)
        ssim_loss = self.ssim(outputs, ground_truth)

        return batch, ground_truth, mse_loss, ssim_loss, outputs

    def step_with_persistence(self, batch, ground_truth):
        batch = batch.to(self.device)
        self.batch = batch
        ground_truth = ground_truth.to(self.device)
        batch_input_images = {}
        for i in range(self.nr_of_input_steps):
            batch_input_images['x_{}'.format(i)] = batch[:, i]
        outputs, persistence_output = self.net(**batch_input_images)

        outputs = outputs.squeeze()
        persistence_output = persistence_output.squeeze()

        if self.backprop_all_outputs:
            ground_truth = ground_truth.unsqueeze(0)
            ground_truth = torch.cat([batch[:, 1:], ground_truth], dim=1)
            ground_truth = ground_truth.squeeze()
        else:
            ground_truth = ground_truth.to(self.device)
            ground_truth = ground_truth.squeeze()

        mse_loss = self.mse(outputs, ground_truth)
        ssim_loss = self.ssim(outputs, ground_truth)

        if self.backprop_all_outputs:
            if len(ground_truth.shape) < 3:
                persistence_loss = self.mse(persistence_output, ground_truth)
            else:
                persistence_loss = self.mse(persistence_output, ground_truth[-1])
        else:
            persistence_loss = self.mse(persistence_output, ground_truth)

        return batch, ground_truth, mse_loss, ssim_loss, persistence_loss, outputs, persistence_output

    def repackage_hidden(self, hidden_state):
        """Wraps hidden states in new Tensors, to detach them from their history."""

        if isinstance(hidden_state, torch.Tensor):
            return hidden_state.detach()
        elif isinstance(hidden_state, list):
            return [self.repackage_hidden(v) for v in hidden_state]
        else:
            return tuple(self.repackage_hidden(v) for v in hidden_state)
