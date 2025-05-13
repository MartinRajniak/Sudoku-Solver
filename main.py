import os

# 3: Filters out INFO, WARNING, and ERROR messages. Shows only FATAL errors.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Force colors even when output is piped
os.environ["FORCE_COLOR"] = "1"

import time
import datetime
import sys

import keras

# Enable mixed precision training to speed up computation
# WARNING: turn off if you run this on CPU - it will significantly slow down training
# https://keras.io/api/mixed_precision/
#
# Enable when training takes too long - it lowered training time by 14% and increased error rate by 16%
keras.mixed_precision.set_global_policy("mixed_float16")

# Set the seed using keras.utils.set_random_seed. This will set:
# 1) `numpy` seed
# 2) backend random seed
# 3) `python` random seed
keras.utils.set_random_seed(42)

from sudoku_solver.config.config import AppConfig
from sudoku_solver.data.dataset import prepare_dataset
from sudoku_solver.model.training import train_model
from sudoku_solver.model.evaluation import (
    plot_histories,
    evaluate_on_difficulties,
    evaluate_replacing_fixed_positions,
    evaluate_puzzle,
)
from sudoku_solver.model.model import prepare_model

EXPERIMENTS_DIR = "experiments"
MODEL_FILE_NAME = "sudoku_solver.keras"

OOS_PUZZLE = (
    "800250704"
    "420000000"
    "000008065"
    "000045300"
    "004603100"
    "007910000"
    "540700000"
    "000000089"
    "209086001"
)

OOS_SOLUTION = (
    "863259714"
    "425167938"
    "791438265"
    "612845397"
    "984673152"
    "357912846"
    "548791623"
    "176324589"
    "239586471"
)

from termcolor import colored


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

    print(f"\nLoading config for experiment {experiment}...")
    experiment_dir_path = os.path.join(EXPERIMENTS_DIR, experiment)
    app_config = AppConfig.from_toml(os.path.join(experiment_dir_path, "config.toml"))
    app_config.ROOT_DIR = experiment_dir_path

    print("\nPreparing datasets...")
    train_datasets, val_dataset, test_dataset = _prepare_dataset(app_config)
    print(f"Dataset loaded after {_format_seconds(time.time() - start_time)}")

    print("\nPreparing model...")
    model = _prepare_model(app_config)

    print("\nTraining model...")
    histories = train_model(model, train_datasets, val_dataset, app_config)
    print(f"Model trained after {_format_seconds(time.time() - start_time)}")

    print("\nSaving model...")
    model.save(os.path.join(app_config.ROOT_DIR, MODEL_FILE_NAME))

    print("\nSaving plot figures...")
    plot_histories(histories, save_to_folder=experiment_dir_path)

    print("\nLoading model to make sure saved version is valid...")
    model = keras.saving.load_model(os.path.join(app_config.ROOT_DIR, MODEL_FILE_NAME))

    print("\nEvaluating model on out of sample puzzle...")
    evaluate_puzzle(model, OOS_PUZZLE, OOS_SOLUTION)

    print("\nEvaluating model on different train dataset difficulties...")
    evaluate_on_difficulties(model, train_datasets)

    print("\nEvaluating model on test set...")
    loss, accuracy, *rest = model.evaluate(test_dataset, verbose=0)
    print(f"On test set, model achieved accuracy: {accuracy} and loss: {loss}")

    print("\nEvaluating model on test set after copying fixed numbers from puzzle...")
    evaluate_replacing_fixed_positions(model, test_dataset)

    print(f"\nFinished after {_format_seconds(time.time() - start_time)}")


def _format_seconds(seconds):
    return str(datetime.timedelta(seconds=seconds))


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
