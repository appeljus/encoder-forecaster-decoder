import logging
import numpy as np
from utils.plot_utils import plot_two_columns_of_numpy_matrics


class Evaluation:
    def __init__(self,
                 run_folder_path,
                 data_utils,
                 writer,
                 mse,
                 ssim):

        self.mse = mse()
        self.ssim = ssim()
        self.writer = writer
        self.data_utils = data_utils
        self.run_folder_path = run_folder_path
        self.estimated_number_of_test_data_points = self.data_utils.get_estimate_total_nr_test_data_points()
        self.running_test_mse_loss = 0.0
        self.test_ssim_loss_array = []
        self.test_mse_loss_array = []
        self.i = 0

    def step(self):
        batch, ground_truth = self.data_utils.get_next_test_data_point()
        ground_truth = ground_truth.squeeze().numpy()
        batch = batch.squeeze().numpy()
        forecast = self.forecast(batch)

        mse_loss = self.mse(forecast, ground_truth)
        ssim_loss = self.ssim(forecast, ground_truth)

        self.running_test_mse_loss += mse_loss
        self.test_mse_loss_array.append(mse_loss)
        self.test_ssim_loss_array.append(ssim_loss)

        return forecast, ground_truth, mse_loss

    def forecast(self, batch):
        return NotImplementedError

    def test(self):
        for i in range(self.estimated_number_of_test_data_points):
            self.i = i
            forecast, ground_truth, loss = self.step()

            if loss > 3 * np.mean(self.test_mse_loss_array):
                self.plot_outlier(ground_truth, i, loss, forecast)

            # Checkpoint
            if i % int(self.estimated_number_of_test_data_points / 100) == 0:
                self.checkpoint(ground_truth, forecast, i)
        np.save(self.run_folder_path + '/arrays/test_mse_loss.npy', self.test_mse_loss_array)
        np.save(self.run_folder_path + '/arrays/test_ssim_loss.npy', self.test_ssim_loss_array)

    def checkpoint(self, ground_truth, outputs, i):
        np.save(self.run_folder_path + '/arrays/test_mse_loss.npy', self.test_mse_loss_array)
        logging.info(
            'Epoch: {}/{}, {:.0%}'.format(i, self.estimated_number_of_test_data_points,
                                          (i / self.estimated_number_of_test_data_points)))

        plot_two_columns_of_numpy_matrics(outputs, ground_truth, i, self.run_folder_path + '/plots/test')

        if i != 0:
            self.writer.add_scalar('test_loss', self.running_test_mse_loss / i, i)
            self.writer.close()

    def plot_outlier(self, ground_truth, i, loss, outputs):
        logging.warning(
            "Higher than average loss for index {}: {}, average loss: {}".format(i, loss.item(),
                                                                                 np.mean(self.test_mse_loss_array)))
        plot_two_columns_of_numpy_matrics(outputs, ground_truth, i, '{}/plots/outliers'.format(self.run_folder_path))

    @staticmethod
    def get_name():
        return NotImplementedError
