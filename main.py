import os

# 3: Filters out INFO, WARNING, and ERROR messages. Shows only FATAL errors.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Force colors even when output is piped
os.environ["FORCE_COLOR"] = "1"

from argparse import ArgumentParser
from dataclasses import asdict
import datetime
import time
import tomllib
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

import mlflow

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
EXPERIMENT_TRIALS_DIR = "trials"
EXPERIMENT_DESCRIPTOR = "description.toml"
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


def main():
    # Define arguments
    parser = ArgumentParser(
        description="Run specified experiment."
        " If trial is specified, only that trial is used."
        " Otherwise all trials in experiment are run."
    )
    parser.add_argument("--experiment", required=True, help="Experiment's directory")
    parser.add_argument("--trial", required=False, help="Trial's directory")

    # Parse arguments
    args = parser.parse_args()
    experiment_dir = args.experiment
    experiment_trial = args.trial

    # Initialize experiment
    experiment_path = os.path.join(EXPERIMENTS_DIR, experiment_dir)
    experiment_desc_path = os.path.join(experiment_path, EXPERIMENT_DESCRIPTOR)

    with open(experiment_desc_path, "rb") as f:
        config_data = tomllib.load(f)

    experiment_name = config_data.get("NAME", "Unknown")

    print(f"Starting experiment {experiment_name}...")
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment(experiment_name=experiment_name)

    experiment_trials_path = os.path.join(experiment_path, EXPERIMENT_TRIALS_DIR)
    if experiment_trial:
        print(f"Running trial {experiment_trial}...")
        run_trial(experiment_trials_path, experiment_trial)
    else:
        print("Running all trials...")
        run_all_trials(experiment_trials_path)


def run_all_trials(experiment_trials_path):
    for trial in os.listdir(experiment_trials_path):
        run_trial(experiment_trials_path, trial)


def run_trial(experiment_trials_path, trial):
    start_time = time.time()

    print(f"\nLoading config for trial {trial}...")
    experiment_trial_path = os.path.join(experiment_trials_path, trial)
    app_config = AppConfig.from_toml(os.path.join(experiment_trial_path, "config.toml"))
    app_config.ROOT_DIR = experiment_trial_path

    mlflow.start_run()
    mlflow.log_params(asdict(app_config))

    print("\nPreparing datasets...")
    train_datasets, val_dataset, test_dataset = _prepare_dataset(app_config)
    print(f"Dataset loaded after {_format_seconds(time.time() - start_time)}")

    print("\nPreparing model...")
    model = prepare_model(app_config)

    print("\nTraining model...")
    histories = train_model(model, train_datasets, val_dataset, app_config)
    print(f"Model trained after {_format_seconds(time.time() - start_time)}")

    print("\nSaving model...")
    model.save(os.path.join(app_config.ROOT_DIR, MODEL_FILE_NAME))

    print("\nSaving plot figures...")
    plot_histories(histories, save_to_folder=experiment_trial_path)

    print("\nLoading model to make sure saved version is valid...")
    model = keras.saving.load_model(os.path.join(app_config.ROOT_DIR, MODEL_FILE_NAME))

    print("\nEvaluating model on out of sample puzzle...")
    evaluate_puzzle(model, OOS_PUZZLE, OOS_SOLUTION)

    print("\nEvaluating model on different train dataset difficulties...")
    difficulty_to_accuracy = evaluate_on_difficulties(model, train_datasets)
    mlflow.log_metrics(difficulty_to_accuracy)

    print("\nEvaluating model on test set...")
    test_loss, test_accuracy, *rest = model.evaluate(test_dataset, verbose=0)
    mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_accuracy})
    print(f"On test set, model achieved accuracy: {test_accuracy} and loss: {test_loss}")

    print("\nEvaluating model on test set after copying fixed numbers from puzzle...")
    fixed_accuracy = evaluate_replacing_fixed_positions(model, test_dataset)
    mlflow.log_metric("fixed_accuracy", fixed_accuracy)

    mlflow.end_run()

    print(f"\nFinished after {_format_seconds(time.time() - start_time)}")


def _format_seconds(seconds):
    return str(datetime.timedelta(seconds=seconds))


def _prepare_dataset(app_config):
    return prepare_dataset(
        app_config.BATCH_SIZE,
        size_limit=app_config.DATA_SIZE_LIMIT,
        use_disk_cache=app_config.USE_DISK_CACHE,
    )


if __name__ == "__main__":
    sys.exit(main())
