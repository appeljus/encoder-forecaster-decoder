import logging
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.data_utils import DataUtils


class Run:
    def __init__(self,
                 writer: SummaryWriter,
                 model,
                 data_utils: DataUtils,
                 device,
                 run_folder_path,
                 do_validation,
                 lr,
                 optimizer,
                 mse,
                 ssim,
                 save_checkpoints=False,
                 do_persistence=False,
                 nr_of_input_steps=3,
                 **model_params):

        self.data_utils = data_utils
        self.estimate_total_nr_train_data_points = self.data_utils.get_estimate_total_nr_train_data_points()
        self.estimate_total_nr_test_data_points = self.data_utils.get_estimate_total_nr_test_data_points()
        self.validation_train_ratio = data_utils.get_validation_train_ratio()
        self.net = self.net.to(device)

        logging.info('Model params - {}, number of parameters in net: {}, '.format(
            ['{}: {}'.format(x, y) for x, y in model_params.items() if 'state_dict' not in x],
            self.count_parameters())
        )

        logging.info('Run params - device: {}, run folder: {}, validation: {},'
                     ' learning rate: {}, save checkpoints: {}, do persistence: {}'.format(device,
                                                                                           run_folder_path,
                                                                                           do_validation,
                                                                                           lr,
                                                                                           save_checkpoints,
                                                                                           do_persistence,
                                                                                           ))

        logging.info(self.net)
        self.nr_of_input_steps = nr_of_input_steps
        self.do_persistence = do_persistence
        self.save_checkpoints = save_checkpoints
        self.lr = lr
        self.do_validation = do_validation
        self.model_params = model_params
        self.run_folder_path = run_folder_path
        self.device = device
        self.model = model
        self.writer = writer
        self.optimizer = optimizer(self.net.parameters(), lr=self.lr)
        self.ssim = ssim()
        self.mse = mse()
        self.train_mse_loss_array = []
        self.train_ssim_loss_array = []
        self.test_mse_loss_array = []
        self.test_ssim_loss_array = []
        self.running_train_mse_loss = 0.0
        self.running_test_mse_loss = 0.0
        self.hidden_state = None

        # writer.add_graph(net, data_utils.get_next_train_data_point().to(device))
        # writer.close()

        if self.do_validation:
            self.running_validation_loss = 0.0
            self.validation_loss_array = []
            self.when_validate = data_utils.get_validation_train_ratio()

    def train(self):
        return NotImplementedError

    def train_step(self, *args):
        return NotImplementedError

    def step(self, *args):
        return NotImplementedError

    def test(self, state_dict=None):
        return NotImplementedError

    def count_parameters(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def load_state_dict(self):
        self.net.load_state_dict(torch.load('{}/checkpoints/state_dict.pth'.format(self.run_folder_path)))
