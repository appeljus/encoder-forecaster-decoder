import cv2
import numpy as np
from evaluation.models.Evaluation import Evaluation
import logging
from scipy.ndimage import gaussian_filter


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


class TVL1(Evaluation):
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

        self.forecast_horizon = forecast_horizon
        if model_params is None:
            model_params = {}
        logging.info('Model params - {} '.format(
            ['{}: {}'.format(x, y) for x, y in model_params.items() if 'state_dict' not in x]
        ))
        self.tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        self.tvl1.setTheta(model_params.get('theta', 0.3))
        self.tvl1.setTau(model_params.get('tau', 0.1))
        self.tvl1.setLambda(model_params.get('lambda', 0.03))
        self.tvl1.setScalesNumber(model_params.get('scales', 3))
        self.tvl1.setWarpingsNumber(model_params.get('warpings', 3))
        self.tvl1.setEpsilon(model_params.get('epsilon', 0.01))
        self.tvl1.setInnerIterations(model_params.get('inner_iterations', 10))
        self.tvl1.setOuterIterations(model_params.get('outer_iterations', 2))
        self.tvl1.setScaleStep(model_params.get('scale_step', 0.5))
        self.tvl1.setGamma(model_params.get('gamma', 0.1))
        self.tvl1.setMedianFiltering(model_params.get('median_filtering', 5))
        self.tvl1.setUseInitialFlow(False)

    @staticmethod
    def transform_images(image):
        image[image == 1] = 0
        image[image == 2] = 127.5
        image[image == 3] = 255
        return image

    @staticmethod
    def detransform_images(image):
        return (image / 127.5) + 1

    def make_forecast(self, previous_image, current_image):
        previous_image = cv2.UMat(previous_image)
        current_image = cv2.UMat(current_image)
        flow = self.tvl1.calc(previous_image, current_image, None)
        moved_vectors = self.move_vectors(flow.get())
        forecast_image = self.warp(current_image.get(), moved_vectors)
        return forecast_image

    def forecast(self, batch):
        previous_image, current_image = batch[-2], batch[-1]
        previous_image = self.transform_images(previous_image)
        current_image = self.transform_images(current_image)
        for i in range(self.forecast_horizon):
            forecast = self.make_forecast(previous_image, current_image)
            previous_image = current_image
            current_image = forecast
        forecast = gaussian_filter(forecast, 0.5)
        forecast = self.detransform_images(forecast)
        return forecast

    def move_vectors(self, flow):
        moved_vectors = np.zeros(flow.shape, dtype='float32')
        for i in range(flow.shape[0]):
            for j in range(flow.shape[1]):
                new_x = i + int(flow[i, j, 0])
                new_y = j + int(flow[i, j, 1])
                if 0 <= new_x < moved_vectors.shape[0] and 0 <= new_y < moved_vectors.shape[1]:
                    try:
                        moved_vectors[i + int(flow[i, j, 0]), j + int(flow[i, j, 1]), :] = flow[i, j, :]
                    except IndexError as e:
                        logging.warning('Index error: {}, for test index: {}'.format(e, self.i))
        return moved_vectors

    @staticmethod
    def warp(current_image, flow):
        h, w = flow.shape[:2]
        inverse_flow = -flow
        inverse_flow[:, :, 0] += np.arange(w)
        inverse_flow[:, :, 1] += np.arange(h)[:, np.newaxis]
        forecast = cv2.remap(current_image, inverse_flow, None, cv2.INTER_CUBIC)
        return forecast

    @staticmethod
    def get_name():
        return 'tvl1'
