import logging
import numpy as np
import torch
from matplotlib import pyplot as plt


def plot_numpy_matrix(matrix, title='', cmap='Greys'):
    if cmap == 'Greys':
        vmin = 1
        vmax = 3
    else:
        vmin = np.min(matrix)
        vmax = np.max(matrix)
    fig, axes = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(10, 10)
    axes.imshow(matrix, cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)
    axes.set_title(title)
    plt.show()


def plot_tensor(tensor, cmap='Greys'):
    if cmap == 'Greys':
        vmin = 1
        vmax = 3
    else:
        vmin = torch.min(tensor)
        vmax = torch.max(tensor)

    if tensor.is_cuda:
        tensor = tensor.cpu()
    matrix = tensor.squeeze().detach().numpy()
    plt.imshow(matrix, cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)
    plt.show()


def plot_two_numpy_matrices(matrix1, matrix2, cmap='Greys'):
    if cmap == 'Greys':
        vmin = 1
        vmax = 3
    else:
        vmin = np.min(matrix1)
        vmax = np.max(matrix2)
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].imshow(matrix1, cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)
    axes[1].imshow(matrix2, cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)
    plt.show()


def plot_two_columns_of_numpy_matrics(matrix1, matrix2, index=0, folder_path='default', cmap='Greys'):
    if cmap == 'Greys':
        vmin = 1
        vmax = 3
    else:
        vmin = min(np.min(matrix1), np.min(matrix2))
        vmax = max(np.max(matrix1), np.max(matrix2))

    if matrix1.shape == matrix2.shape:
        if len(matrix1.shape) <= 2:
            matrices_len = 1
        else:
            matrices_len = matrix1.shape[0]

        fig, axes = plt.subplots(nrows=matrices_len, ncols=2)
        fig.set_size_inches(10, matrices_len * 5)

        if matrices_len < 2:
            axes[0].imshow(matrix1, cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)
            axes[0].label_outer()
            axes[1].imshow(matrix2, cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)
            axes[1].label_outer()
        else:
            for i in range(matrices_len):
                axes[i, 0].imshow(matrix1[i], cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)
                axes[i, 0].label_outer()
                axes[i, 1].imshow(matrix2[i], cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)
                axes[i, 1].label_outer()
        if folder_path != 'default':
            plt.savefig('{0}/prediction_ground_truth_plot_{1}.png'.format(folder_path, index))
        else:
            plt.show()
        plt.close('all')
    else:
        logging.error('matrix 1 shape != matrix 2 shape: {}, {},'
                      ' for index {}'.format(matrix1.shape,
                                             matrix2.shape,
                                             index))


def plot_three_columns_of_tensors(predicted_tensor, ground_truth_tensor, persistence_tensor, index=0, folder_path='default', cmap='Greys'):
    if cmap == 'Greys':
        vmin = 1
        vmax = 3
    else:
        vmin = min(torch.min(predicted_tensor), torch.min(ground_truth_tensor), torch.min(persistence_tensor))
        vmax = max(torch.max(predicted_tensor), torch.max(ground_truth_tensor), torch.max(persistence_tensor))

    predicted_tensor = predicted_tensor.squeeze()
    ground_truth_tensor = ground_truth_tensor.squeeze()
    persistence_tensor = persistence_tensor.squeeze()
    if predicted_tensor.shape == ground_truth_tensor.shape == persistence_tensor.shape:
        pass
    else:
        temp = torch.zeros(ground_truth_tensor.shape)
        temp[-1] = persistence_tensor
        persistence_tensor = temp
    if predicted_tensor.shape == ground_truth_tensor.shape == persistence_tensor.shape:
        if len(predicted_tensor.shape) <= 2:
            matrices_len = 1
        else:
            matrices_len = predicted_tensor.shape[0]

        if predicted_tensor.is_cuda:
            predicted_tensor = predicted_tensor.cpu()
        if ground_truth_tensor.is_cuda:
            ground_truth_tensor = ground_truth_tensor.cpu()
        if persistence_tensor.is_cuda:
            persistence_tensor = persistence_tensor.cpu()

        predicted_tensor = predicted_tensor.detach().numpy()
        ground_truth_tensor = ground_truth_tensor.numpy()
        persistence_tensor = persistence_tensor.detach().numpy()

        fig, axes = plt.subplots(nrows=matrices_len, ncols=3)
        fig.set_size_inches(10, matrices_len * 5)
        if matrices_len < 2:
            axes[0].imshow(predicted_tensor, cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)
            axes[0].label_outer()
            axes[1].imshow(ground_truth_tensor, cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)
            axes[1].label_outer()
            axes[2].imshow(persistence_tensor, cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)
            axes[2].label_outer()
        else:
            for i in range(matrices_len):
                axes[i, 0].imshow(predicted_tensor[i], cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)
                axes[i, 0].label_outer()
                axes[i, 1].imshow(ground_truth_tensor[i], cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)
                axes[i, 1].label_outer()
                axes[i, 2].imshow(persistence_tensor[i], cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)
                axes[i, 2].label_outer()
        if folder_path != 'default':
            plt.savefig('{0}/prediction_ground_truth_plot_{1}.png'.format(folder_path, index))
        else:
            plt.show()
        plt.close('all')
    else:
        logging.error('predicted tensor shape != ground truth tensor != persistence tensor shape: {}, {}, {},'
                      ' for index {}'.format(predicted_tensor.shape,
                                             ground_truth_tensor.shape,
                                             persistence_tensor.shape,
                                             index))


def plot_two_columns_of_tensors(predicted_tensor, ground_truth_tensor, index=0, folder_path='default', cmap='Greys'):
    if cmap == 'Greys':
        vmin = 1
        vmax = 3
    else:
        vmin = min(torch.min(predicted_tensor), torch.min(ground_truth_tensor))
        vmax = max(torch.max(predicted_tensor), torch.max(ground_truth_tensor))

    predicted_tensor = predicted_tensor.squeeze()
    ground_truth_tensor = ground_truth_tensor.squeeze()
    if predicted_tensor.shape == ground_truth_tensor.shape:
        if len(predicted_tensor.shape) <= 2:
            matrices_len = 1
        else:
            matrices_len = predicted_tensor.shape[0]
        if predicted_tensor.is_cuda:
            predicted_tensor = predicted_tensor.cpu()
        if ground_truth_tensor.is_cuda:
            ground_truth_tensor = ground_truth_tensor.cpu()
        predicted_tensor = predicted_tensor.detach().numpy()
        ground_truth_tensor = ground_truth_tensor.numpy()
        fig, axes = plt.subplots(nrows=matrices_len, ncols=2)
        fig.set_size_inches(10, matrices_len * 5)
        if matrices_len < 2:
            axes[0].imshow(predicted_tensor, cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)
            axes[0].label_outer()
            axes[1].imshow(ground_truth_tensor, cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)
            axes[1].label_outer()
        else:
            for i in range(matrices_len):
                axes[i, 0].imshow(predicted_tensor[i], cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)
                axes[i, 0].label_outer()
                axes[i, 1].imshow(ground_truth_tensor[i, :], cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)
                axes[i, 1].label_outer()
        if folder_path != 'default':
            plt.savefig('{0}/prediction_ground_truth_plot_{1}.png'.format(folder_path, index))
        else:
            plt.show()
        plt.close('all')
    else:
        logging.error('predicted_matrices.shape != ground_truth_matrices.shape: {}, {}, for index {}'.format(predicted_tensor.shape, ground_truth_tensor.shape, index))


def plot_two_columns_of_tensors_with_input(predicted_tensor, ground_truth_tensor, inputs_tensor, index=0, folder_path='default', cmap='Greys'):
    if cmap == 'Greys':
        vmin = 1
        vmax = 3
    else:
        vmin = min(torch.min(predicted_tensor), torch.min(ground_truth_tensor))
        vmax = max(torch.max(predicted_tensor), torch.max(ground_truth_tensor))

    inputs_tensor = inputs_tensor.squeeze()
    predicted_tensor = predicted_tensor.squeeze()
    ground_truth_tensor = ground_truth_tensor.squeeze()
    if predicted_tensor.shape == ground_truth_tensor.shape:
        if predicted_tensor.is_cuda:
            predicted_tensor = predicted_tensor.cpu()
        if ground_truth_tensor.is_cuda:
            ground_truth_tensor = ground_truth_tensor.cpu()
        if inputs_tensor.is_cuda:
            inputs_tensor = inputs_tensor.cpu()

        predicted_tensor = predicted_tensor.detach().numpy()
        ground_truth_tensor = ground_truth_tensor.numpy()
        inputs_tensor = inputs_tensor.numpy()
        nr_of_inputs = inputs_tensor.shape[0]

        fig, axes = plt.subplots(nrows=nr_of_inputs+1, ncols=2)
        fig.set_size_inches(10, 4 * 5)

        for i in range(nr_of_inputs):
            axes[i, 0].axis('off')
            axes[i, 1].imshow(inputs_tensor[i, :], cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)

        axes[nr_of_inputs, 0].imshow(predicted_tensor, cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)
        axes[nr_of_inputs, 0].label_outer()
        axes[nr_of_inputs, 1].imshow(ground_truth_tensor, cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)
        axes[nr_of_inputs, 1].label_outer()
        if folder_path != 'default':
            plt.savefig('{0}/prediction_ground_truth_plot_{1}.png'.format(folder_path, index))
        else:
            plt.show()
        plt.close('all')
    else:
        logging.error('predicted_matrices.shape != ground_truth_matrices.shape: {}, {}, for index {}'.format(predicted_tensor.shape, ground_truth_tensor.shape, index))


def plot_time_series_numpy_matrix(time_series_matrix, cmap='Greys'):
    if cmap == 'Greys':
        vmin = 1
        vmax = 3
    else:
        vmin = np.min(time_series_matrix)
        vmax = np.max(time_series_matrix)
    matrices_len = time_series_matrix.shape[0]
    fig, axes = plt.subplots(nrows=matrices_len, ncols=1)
    fig.set_size_inches(10, 10 * matrices_len)
    for i in range(matrices_len):
        axes[i].imshow(time_series_matrix[i], cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)
    plt.show()


def plot_time_series_tensor(time_series_tensor, folder_path='', cmap='Greys', file_name='series_plot'):
    if cmap == 'Greys':
        vmin = 1
        vmax = 3
    else:
        vmin = torch.min(time_series_tensor)
        vmax = torch.max(time_series_tensor)
    if time_series_tensor.is_cuda:
        time_series_tensor = time_series_tensor.cpu()
    time_series_tensor = time_series_tensor.detach()
    time_series_tensor = time_series_tensor.squeeze()
    if len(time_series_tensor.shape) <= 2:
        matrices_len = 1
    else:
        matrices_len = time_series_tensor.shape[0]
    fig, axes = plt.subplots(nrows=matrices_len, ncols=1)
    fig.set_size_inches(10, 10*matrices_len)
    if matrices_len < 2:
        axes.imshow(time_series_tensor, cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)
    else:
        for i in range(matrices_len):
            axes[i].imshow(time_series_tensor[i], cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)
    if folder_path == '':
        plt.show()
    else:
        plt.savefig('{0}/{1}.png'.format(folder_path, file_name))


def plot_loss_numpy_array(loss_array, folder_path='default'):
    fig, axes = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(10, 10)
    axes = plt.plot(loss_array)
    if folder_path != 'default':
        plt.savefig('{}/loss_plot.png'.format(folder_path))
    else:
        plt.show()
    plt.close('all')


def plot_train_validation_loss_numpy(train_loss_array, validation_loss_array, folder_path='default'):
    fig, axes = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(10, 10)
    axes = plt.plot(train_loss_array)
    axes = plt.plot(validation_loss_array[:, 0], validation_loss_array[:, 1])
    if folder_path != 'default':
        plt.savefig('{}/train_validation_loss_plot.png'.format(folder_path))
    else:
        plt.show()
    plt.close('all')
