import os
# 3: Filters out INFO, WARNING, and ERROR messages. Shows only FATAL errors.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import datetime
import sys

import keras

from sudoku_solver.config.config import AppConfig
from sudoku_solver.data.dataset import prepare_dataset
from sudoku_solver.model.training import train_model
from sudoku_solver.model.evaluation import plot_histories
from sudoku_solver.model.model import prepare_model

EXPERIMENTS_DIR = "experiments"
MODEL_FILE_NAME = "sudoku_solver.keras"


def main():
    experiment_to_run = sys.argv[1]
    if experiment_to_run:
        print(f"Running experiment {experiment_to_run}...")
        run_experiment(experiment_to_run)
    else:
        print("Running all experiments...")
        run_all_experiments()


def run_all_experiments():
    for experiment in os.listdir(EXPERIMENTS_DIR):
        run_experiment(experiment)


def run_experiment(experiment):
    start_time = time.time()

    print(f"Loading config for experiment {experiment}...")
    experiment_dir_path = os.path.join(EXPERIMENTS_DIR, experiment)
    app_config = AppConfig.from_toml(os.path.join(experiment_dir_path, "config.toml"))
    app_config.ROOT_DIR = experiment_dir_path

    print("Preparing datasets...")
    train_datasets, val_dataset, test_dataset = _prepare_dataset(app_config)
    print(f"Datasets loaded after {str(datetime.timedelta(seconds=(time.time() - start_time)))}")

    print("Preparing model...")
    model = _prepare_model(app_config)

    print("Training model...")
    histories = train_model(model, train_datasets, val_dataset, app_config)
    print(f"Model trained after {str(datetime.timedelta(seconds=(time.time() - start_time)))}")

    print("Saving model...")
    model.save(os.path.join(app_config.ROOT_DIR, MODEL_FILE_NAME))

    print("Saving plot figures...")
    plot_histories(histories, save_to_folder=experiment_dir_path)

    print("Loading model to make sure saved version is valid...")
    model = keras.saving.load_model(os.path.join(app_config.ROOT_DIR, MODEL_FILE_NAME))

    print("Evaluating model...")
    loss, accuracy, *rest = model.evaluate(test_dataset)

    print(f"On test set, model achieved accuracy: {accuracy} and loss: {loss}")

    print(f"Experiment finished after {str(datetime.timedelta(seconds=(time.time() - start_time)))}")


def _prepare_dataset(app_config):
    return prepare_dataset(
        app_config.BATCH_SIZE,
        size_limit=app_config.DATA_SIZE_LIMIT,
        use_disk_cache=app_config.USE_DISK_CACHE,
    )


def _prepare_model(app_config):
    return prepare_model(
        use_residual=app_config.USE_RESIDUAL,
        use_fixed_number_layer=app_config.USE_FIXED_NUMBER_LAYER,
        learning_rate=app_config.LEARNING_RATE,
        constraint_weight=app_config.CONSTRAINT_WEIGHT,
        fixed_cell_weight=app_config.FIXED_CELL_WEIGHT,
    )


if __name__ == "__main__":
    sys.exit(main())
