import os


def create_run_directory(model_directory, run_name):
    runs = os.listdir('./upgraded-guacamole/runs/{}'.format(model_directory))
    runs.sort()
    if len(runs) < 1:
        last_run = 'run_0'
    else:
        actual_runs = []
        for run in runs:
            if "run" in run:
                actual_runs.append(run)
        if len(actual_runs) < 1:
            last_run = 'run_0'
        else:
            actual_runs.sort()
            last_run = actual_runs[-1]
    latest_run_number = [int(s) for s in last_run.split('_') if s.isdigit()][0]
    new_run_number = latest_run_number + 1
    new_run_folder_path = './upgraded-guacamole/runs/{}/run_{:02d}_{}'.format(model_directory, new_run_number, run_name)
    os.mkdir(new_run_folder_path)
    os.mkdir(new_run_folder_path + '/arrays')
    os.mkdir(new_run_folder_path + '/checkpoints')
    os.mkdir(new_run_folder_path + '/plots')
    os.mkdir(new_run_folder_path + '/plots/outliers')
    os.mkdir(new_run_folder_path + '/plots/outliers/train')
    os.mkdir(new_run_folder_path + '/plots/outliers/test')
    os.mkdir(new_run_folder_path + '/plots/filtered')
    os.mkdir(new_run_folder_path + '/plots/train')
    os.mkdir(new_run_folder_path + '/plots/test')
    os.mkdir(new_run_folder_path + '/plots/persistence')
    os.mkdir(new_run_folder_path + '/plots/persistence/train')
    os.mkdir(new_run_folder_path + '/plots/persistence/train_outliers')
    os.mkdir(new_run_folder_path + '/plots/persistence/test')

    return new_run_folder_path
