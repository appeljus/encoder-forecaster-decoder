import math
import numpy as np
import os
import torch
import logging


class DataUtils:
    def __init__(self,
                 data_folder_path,
                 batch_size=1,
                 test_train_split=0.9,
                 train_validation_split=0.8,
                 max_data_points=138889):
        self.data_path = data_folder_path
        self.files = os.listdir(self.data_path)

        self.batch_size = batch_size

        nr_of_data_files = len(self.files)
        train_index_end = int(nr_of_data_files * test_train_split * train_validation_split)
        validation_index_start = train_index_end
        validation_index_end = int(nr_of_data_files * test_train_split)
        test_index_start = validation_index_end

        self.train_file_names = self.files[:train_index_end]
        self.validation_file_names = self.files[validation_index_start:validation_index_end]
        self.test_file_names = self.files[test_index_start:]

        self.nr_of_train_files = len(self.train_file_names)
        self.nr_of_test_files = len(self.test_file_names)
        self.nr_of_validation_files = len(self.validation_file_names)

        self.train_file_index = 0
        self.train_data_point_index = 0
        self.train_file = np.load(self.data_path + '/' + self.train_file_names[self.train_file_index])
        self.current_nr_of_data_points_in_train_file = len(self.train_file)
        self.estimate_total_nr_of_train_data_points = self.nr_of_train_files * self.current_nr_of_data_points_in_train_file


        self.validation_file_index = 0
        self.validation_data_point_index = 0
        self.validation_file = np.load(self.data_path + '/' + self.validation_file_names[self.validation_file_index])
        self.current_nr_of_data_points_in_validation_file = len(self.validation_file)
        self.estimate_total_nr_of_validation_data_points = self.nr_of_validation_files * self.current_nr_of_data_points_in_validation_file


        self.test_file_index = 0
        self.test_data_point_index = 0
        self.test_file = np.load(self.data_path + '/' + self.test_file_names[self.test_file_index])
        self.current_nr_of_data_points_in_test_file = len(self.test_file)
        self.estimate_total_nr_of_test_data_points = self.nr_of_test_files * self.current_nr_of_data_points_in_test_file

        if self.estimate_total_nr_of_train_data_points > max_data_points * test_train_split * train_validation_split:
            self.estimate_total_nr_of_train_data_points = int(max_data_points * test_train_split * train_validation_split)
            self.estimate_total_nr_of_validation_data_points = int(max_data_points * test_train_split * (1 - train_validation_split))
            self.estimate_total_nr_of_test_data_points = int(max_data_points * (1 - test_train_split))
        elif self.estimate_total_nr_of_train_data_points < 10000:
            self.estimate_total_nr_of_train_data_points = int(max_data_points * test_train_split * train_validation_split)
            self.estimate_total_nr_of_validation_data_points = int(max_data_points * test_train_split * (1 - train_validation_split))
            self.estimate_total_nr_of_test_data_points = int(max_data_points * (1 - test_train_split))

        try:
            self.image_shape = self.train_file[0].shape
        except IndexError:
            self.image_shape = (363, 363)

        logging.info("Data Utils params - {}".format({
            "data_folder_path": data_folder_path,
            "batch_size": batch_size,
            "test_train_split": test_train_split,
            "train_validation_split": train_validation_split,
            "max_data_points": max_data_points,
            "train_index_end": train_index_end,
            "validation_index_start": validation_index_start,
            "validation_index_end": validation_index_end,
            "test_index_start": test_index_start,
            "nr_of_data_files": nr_of_data_files,
            "nr_of_train_files": self.nr_of_train_files,
            "nr_of_test_files": self.nr_of_test_files,
            "nr_of_validation_files": self.nr_of_validation_files,
            "estimate_total_nr_of_train_data_points": self.estimate_total_nr_of_train_data_points,
            "estimate_total_nr_of_validation_data_points": self.estimate_total_nr_of_validation_data_points,
            "estimate_total_nr_of_test_data_points": self.estimate_total_nr_of_test_data_points
        }))

    def _compose_data_series(self, data_set_name):
        raise NotImplementedError

    def _check_data_index_out_of_bounds(self, data_set_name):
        data_set_data_point_index = getattr(self, "{}_data_point_index".format(data_set_name))
        current_nr_of_data_points_in_data_set_file = getattr(self, "current_nr_of_data_points_in_{}_file".format(
            data_set_name))
        data_set_file_index = getattr(self, "{}_file_index".format(data_set_name))
        data_set = getattr(self, "{}_file_names".format(data_set_name))
        nr_of_data_set_files = getattr(self, "nr_of_{}_files".format(data_set_name))

        # If the data point index will be out of bounds, fetch new data set file
        if data_set_data_point_index + self.batch_size >= current_nr_of_data_points_in_data_set_file:
            # Try to fetch next data set file
            setattr(self, "{}_file_index".format(data_set_name), data_set_file_index + 1)
            try:
                setattr(self, "{}_file".format(data_set_name),
                        np.load(self.data_path + '/' + data_set[data_set_file_index]))
            except IndexError:
                logging.warning('{0} file index out of bounds. Please check code. Resetting {0} file '
                                'index, this is undesirable. {0} file index: {1}. {0} data point index: '
                                '{2}. Nr of {0} files: {3}'.format(data_set_name, data_set_file_index,
                                                                   data_set_data_point_index,
                                                                   nr_of_data_set_files))
                # Reset file index of data set
                setattr(self, "{}_file_index".format(data_set_name), 0)
                # Fetch new data set file
                setattr(self, "{}_file".format(data_set_name),
                        np.load(self.data_path + '/' + data_set[0]))
            # Reset data point index
            data_set_file = getattr(self, "{}_file".format(data_set_name))
            setattr(self, '{}_data_point_index'.format(data_set_name), 0)
            # Set length of data set file
            setattr(self, 'current_nr_of_data_points_in_{}_file'.format(data_set_name), len(data_set_file))

    def get_next_validation_data_point(self):
        self._check_data_index_out_of_bounds('validation')
        result = self._compose_data_series('validation')
        self.validation_data_point_index += 1
        return result

    def get_next_train_data_point(self):
        self._check_data_index_out_of_bounds('train')
        result = self._compose_data_series('train')
        self.train_data_point_index += 1
        return result

    def get_next_test_data_point(self):
        self._check_data_index_out_of_bounds('test')
        result = self._compose_data_series('test')
        self.test_data_point_index += 1
        return result

    def get_current_nr_of_data_points_in_train_file(self):
        return self.current_nr_of_data_points_in_train_file

    def get_estimate_total_nr_train_data_points(self):
        return self.estimate_total_nr_of_train_data_points

    def get_estimate_total_nr_test_data_points(self):
        return self.estimate_total_nr_of_test_data_points

    def get_validation_train_ratio(self):
        return self.estimate_total_nr_of_train_data_points / self.estimate_total_nr_of_validation_data_points

    def determine_input_size(self, number_of_conv_stages):
        result_x = self.image_shape[0]
        result_y = self.image_shape[1]
        for i in range(number_of_conv_stages):
            result_x = math.floor(result_x / 2)
            result_y = math.floor(result_y / 2)
        return result_x * result_y * 1024


class EncoderDecoderDataUtils(DataUtils):
    def __init__(self, data_folder_path, batch_size=1, test_train_split=0.9, train_validation_split=0.8,
                 max_data_points=138889):
        super().__init__(data_folder_path, batch_size, test_train_split, train_validation_split, max_data_points)

    def _compose_data_series(self, data_set_name):
        data_file = getattr(self, "{}_file".format(data_set_name))
        data_point_index = getattr(self, "{}_data_point_index".format(data_set_name))
        if self.batch_size < 2:
            result = data_file[data_point_index]
            result = torch.from_numpy(result).float().unsqueeze(0).unsqueeze(0)
        else:
            result = data_file[data_point_index:data_point_index + self.batch_size]
            result = torch.from_numpy(result).float().unsqueeze(1)
        return result


class RecurrentDataUtils(DataUtils):
    def __init__(self,
                 data_folder_path,
                 forecast_horizon=1,
                 batch_size=1,
                 test_train_split=0.9,
                 train_validation_split=0.8,
                 nr_of_input_steps=1,
                 input_frequency=1,
                 max_data_points=138889,
                 data_points_per_file=51
                 ):
        super().__init__(data_folder_path, batch_size, test_train_split, train_validation_split, max_data_points)
        self.data_points_per_file = data_points_per_file
        self.input_frequency = input_frequency
        self.forecast_horizon = forecast_horizon
        self.nr_of_input_steps = nr_of_input_steps

        self._get_next_files('train')
        self._get_next_files('validation')
        self._get_next_files('test')

        logging.info("Recurrent Data Utils params: {}".format({
            "input_frequency": self.input_frequency,
            "forecast_horizon": self.forecast_horizon,
            "nr_of_input_steps": self.nr_of_input_steps
        }))

    def _check_data_index_out_of_bounds(self, data_set_name):
        data_point_index = getattr(self, '{}_data_point_index'.format(data_set_name))
        file_index = getattr(self, '{}_file_index'.format(data_set_name))
        nr_of_files = getattr(self, 'nr_of_{}_files'.format(data_set_name))
        # Data point index out of bounds?
        if data_point_index >= self.data_points_per_file:
            # File index out of bounds?
            if file_index + self.nr_of_input_steps * self.input_frequency + 2 * self.batch_size + self.forecast_horizon >= nr_of_files:
                logging.warning('{data_set_name} file index out of bounds, please check code. '
                                'Resetting {data_set_name} file index, this is undesirable. '
                                '{data_set_name} file index: {file_index}.'
                                '{data_set_name} data point index: {data_point_index}. '
                                'Nr of {data_set_name} files: {nr_of_files}'.format(data_set_name=data_set_name,
                                                                                    file_index=file_index,
                                                                                    data_point_index=data_point_index,
                                                                                    nr_of_files=nr_of_files))
                # Reset file index and files:
                setattr(self, '{}_file_index'.format(data_set_name), 0)
                setattr(self, '{}_data_point_index'.format(data_set_name), 0)
                self._get_next_files(data_set_name)
            else:
                # increment file index and fetch new files
                setattr(self, '{}_file_index'.format(data_set_name), file_index + 1)
                setattr(self, '{}_data_point_index'.format(data_set_name), 0)
                self._get_next_files(data_set_name)

    def _get_next_files(self, data_set_name):
        data_set_file_names = getattr(self, '{}_file_names'.format(data_set_name))
        file_index = getattr(self, '{}_file_index'.format(data_set_name))
        input_end_index = file_index + self.nr_of_input_steps * self.input_frequency + self.batch_size - 1
        ground_truth_start_index = input_end_index + self.forecast_horizon - 1
        ground_truth_end_index = input_end_index + self.batch_size + self.forecast_horizon - 1
        setattr(self, '{}_files'.format(data_set_name),
                np.array([np.load(self.data_path + '/' + x) for x in data_set_file_names[file_index:
                                                                                         input_end_index:
                                                                                         self.input_frequency]]))
        setattr(self, '{}_ground_truth_files'.format(data_set_name),
                np.array([np.load(self.data_path + '/' + x) for x in data_set_file_names[ground_truth_start_index:
                                                                                         ground_truth_end_index]]))

    def _compose_data_series(self, data_set_name):
        data_files = getattr(self, "{}_files".format(data_set_name))
        data_point_index = getattr(self, "{}_data_point_index".format(data_set_name))
        # Output data shape:
        # [batch, nr_of_input_steps, channel, x, y]
        if self.batch_size < 2:
            # [1, nr_of_input_steps, 1, x, y]
            result = data_files[:, data_point_index]
            result = torch.from_numpy(result).float().unsqueeze(1).unsqueeze(0)
        else:
            # [ batch, nr_of_input_steps, 1, x, y]
            result = np.empty([self.batch_size, self.nr_of_input_steps, *self.image_shape])
            for batch_index in range(self.batch_size):
                result[batch_index] = data_files[batch_index:batch_index + self.nr_of_input_steps, data_point_index]
            result = torch.from_numpy(result).float().unsqueeze(2)
        return result

    def _get_ground_truth(self, data_set_name):
        ground_truth_data_files = getattr(self, "{}_ground_truth_files".format(data_set_name))
        data_point_index = getattr(self, "{}_data_point_index".format(data_set_name))
        # Data shape:
        # [batch, 1, x, y]
        if self.batch_size < 2:
            # [1, 1, x, y]
            result = ground_truth_data_files[:, data_point_index]
            result = torch.from_numpy(result).float().unsqueeze(0)
        else:
            # [batch, 1, x, y]
            result = np.empty([self.batch_size, 1, *self.image_shape])
            for batch_index in range(self.batch_size):
                result[batch_index] = ground_truth_data_files[batch_index, data_point_index]
            result = torch.from_numpy(result).float()
        return result

    def get_next_train_data_point(self):
        self._check_data_index_out_of_bounds('train')
        result = self._compose_data_series('train'), self._get_ground_truth('train')
        self.train_data_point_index += 1
        return result

    def get_next_test_data_point(self):
        self._check_data_index_out_of_bounds('test')
        result = self._compose_data_series('test'), self._get_ground_truth('test')
        self.test_data_point_index += 1
        return result

    def get_next_validation_data_point(self):
        self._check_data_index_out_of_bounds('validation')
        result = self._compose_data_series('validation'), self._get_ground_truth('validation')
        self.validation_data_point_index += 1
        return result


class SISRecurrentDataUtils(RecurrentDataUtils):
    def __init__(self,
                 data_folder_path,
                 forecast_horizon=1,
                 batch_size=1,
                 test_train_split=0.9,
                 train_validation_split=0.8,
                 nr_of_input_steps=1,
                 input_frequency=1,
                 max_data_points=138889,
                 debug=False,
                 ):
        logging.info('SISRecurrentDataUtils params: {}'.format({'debug': debug}))

        super().__init__(data_folder_path,
                         forecast_horizon,
                         batch_size,
                         test_train_split,
                         train_validation_split,
                         nr_of_input_steps,
                         input_frequency,
                         max_data_points,
                         data_points_per_file=49)

        self.debug = debug

    def _check_data_index_out_of_bounds(self, data_set_name):
        super()._check_data_index_out_of_bounds(data_set_name)
        incorrect_data = self._check_data_correctness(data_set_name)
        while incorrect_data:
            data_point_index = getattr(self, "{}_data_point_index".format(data_set_name))
            setattr(self, "{}_data_point_index".format(data_set_name), data_point_index + 1)
            super()._check_data_index_out_of_bounds(data_set_name)
            incorrect_data = self._check_data_correctness(data_set_name)

    def _check_data_correctness(self, data_set_name):
        data_series = self._compose_data_series(data_set_name)
        ground_truth = self._get_ground_truth(data_set_name)
        result = torch.any(data_series < 1.0).item() or torch.any(ground_truth < 1.0).item()
        if self.debug:
            if result:
                data_point_index = getattr(self, '{}_data_point_index'.format(data_set_name))
                file_index = getattr(self, "{}_file_index".format(data_set_name))
                logging.info('Data point {} in file {} contains incorrect data'.format(data_point_index, file_index))
        return result
