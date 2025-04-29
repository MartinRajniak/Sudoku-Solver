from sudoku_solver.utils.dataset import print_pipeline_performance
from sudoku_solver.data.preprocess import preprocess_dataset
from sudoku_solver.data.serialization import load_from_disk, save_to_disk

import kagglehub
import os
import pathlib

import numpy as np
import tensorflow as tf

import shutil

ROOT_CACHE_DIR = "cache"

# Input pipeline


def configure_for_performance(ds, shuffle, batch_size, use_disk_cache=False, cache_dir=None):
    if cache_dir:
        # if memory is not an issue, do not specify disk folder so that everything is loaded to memory
        if use_disk_cache:
            # Clear any cache from previous runs
            cache_path = os.path.join(ROOT_CACHE_DIR, cache_dir)
            shutil.rmtree(cache_path, ignore_errors=True)
            os.makedirs("cache/train", exist_ok=True)
            ds = ds.cache(cache_path)
        else:
            ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=100_000)
    ds = ds.batch(
        batch_size, drop_remainder=True  # drop_remainder=True = better for TPU/GPU
    )
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def download_from_network():

    # Download from Kaggle
    path = kagglehub.dataset_download("rohanrao/sudoku")
    print("Path to dataset files:", path)
    print(os.listdir(path))
    FILE_PATH = os.path.join(path, "sudoku.csv")

    # Read downloaded text
    sudoku_text = pathlib.Path(FILE_PATH).read_text()
    sudoku_lines = sudoku_text.split("\n")[1:-1]  # Drop header
    del sudoku_text

    # Decode CSV to Tensor list
    sudoku_tensors = np.array(
        tf.io.decode_csv(sudoku_lines, record_defaults=[str()] * 2)
    )
    print(f"Decoded CSV shape: {sudoku_tensors.shape}")
    del sudoku_lines

    X_tensors, y_tensors = sudoku_tensors
    return X_tensors, y_tensors


def prepare_dataset(batch_size: int, size_limit: int = None, use_disk_cache=False):

    print("Trying to prepare dataset from disk")
    train_loaded_datasets, val_loaded_dataset, test_loaded_dataset = load_from_disk()

    if not train_loaded_datasets or not val_loaded_dataset or not test_loaded_dataset:
        print("Pre-processed sudoku data not found on disk. Downloading new version...")
        X_tensors, y_tensors = download_from_network()

        # TODO: remove
        X_tensors, y_tensors = X_tensors[:300_000], y_tensors[:300_000]

        print("Download complete. Starting preprocess...")
        (
            train_preprocessed_datasets,
            val_preprocessed_dataset,
            test_preprocessed_dataset,
        ) = preprocess_dataset(X_tensors, y_tensors)
        del X_tensors, y_tensors

        print("Preprocess complete. Starting to save data to disk...")
        save_to_disk(train_preprocessed_datasets, val_preprocessed_dataset, test_preprocessed_dataset)
        del train_preprocessed_datasets
        del val_preprocessed_dataset
        del test_preprocessed_dataset

        print("Saving complete. Trying to prepare dataset from disk again...")
        train_loaded_datasets, val_loaded_dataset, test_loaded_dataset = (
            load_from_disk()
        )

    print("Dataset ready.")

    # Configure input pipeline for performance
    train_datasets = []
    for index, train_loaded_dataset in enumerate(train_loaded_datasets):
        train_dataset = configure_for_performance(
            train_loaded_dataset,
            shuffle=True,
            batch_size=batch_size,
            use_disk_cache=use_disk_cache,
            cache_dir="train_{:02d}".format(index),
        )
        train_datasets.append(train_dataset)

    val_dataset = configure_for_performance(
        val_loaded_dataset,
        shuffle=False,
        batch_size=batch_size,
        use_disk_cache=use_disk_cache,
        cache_dir="val",
    )

    test_dataset = configure_for_performance(
        test_loaded_dataset, 
        shuffle=False,
        batch_size=batch_size,
    )
    del train_loaded_datasets, val_loaded_dataset, test_loaded_dataset

    # Limit training size for faster training
    if size_limit:
        n_train_ds = len(train_datasets)
        for index in range(n_train_ds):
            train_datasets[index] = train_datasets[index].take(size_limit // n_train_ds)

    print_pipeline_performance(train_datasets[0])

    return train_datasets, val_dataset, test_dataset
