from evaluation.models.Evaluation import Evaluation


class Persistence(Evaluation):
    def __init__(self,
                 run_folder_path,
                 data_utils,
                 writer,
                 mse,
                 ssim,
                 forecast_horizon=1,
                 model_params=None):

        super().__init__(run_folder_path,
                         data_utils,
                         writer,
                         mse,
                         ssim)

    def forecast(self, batch):
        return batch[-1]

    @staticmethod
    def get_name():
        return 'persistence'
