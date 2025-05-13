import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# We have enough data so 2.5% for test and val sets is enough (still 225k)
TEST_TRAIN_SPLIT = 0.05
VAL_TEST_SPLIT = 0.5


@tf.function
def preprocess(puzzle_tensor):
    byte_values = tf.io.decode_raw(puzzle_tensor, tf.uint8)

    # Convert ASCII digit byte values to integer numbers (0-9)
    # '0' is ASCII 48, '1' is 49, ..., '9' is 57
    # Subtract the ASCII value of '0' (48) to get the integer digit
    # This works correctly because the digits '0'-'9' are consecutive in ASCII
    numbers = byte_values - tf.constant(ord("0"), dtype=tf.uint8)

    numbers = tf.cast(numbers, tf.int32)

    return tf.reshape(numbers, (9, 9, 1))


def preprocess_input(puzzle_tensor):
    return (preprocess(puzzle_tensor) / 9) - 0.5


def preprocess_target(puzzle_tensor):
    return (
        preprocess(puzzle_tensor) - 1
    )  # 0-based predictions (when presenting results, do not forget to add +1)


@tf.function
def preprocess_map(X, y):
    return (preprocess_input(X), preprocess_target(y))


def inverse_preprocess_input(X):
    denormalize = denormalize_input(X)
    reshape = tf.reshape(denormalize, (9, 9))
    return tf.cast(reshape, X.dtype)


def inverse_preprocess_target(y):
    scale = denormalize_label(y)
    reshape = tf.reshape(scale, (9, 9))
    return tf.cast(reshape, y.dtype)


def denormalize_input(X):
    return (X + 0.5) * 9


def denormalize_label(y):
    return y + 1


def replace_rare_difficulties(difficulties):
    # Replace difficulties that are rare with the most common one so that we can split evenly
    difficulty_counts = np.bincount(difficulties)
    rare_difficulties = np.where(difficulty_counts < 100)[0]
    most_common_difficulty = np.argmax(difficulty_counts)
    return np.where(
        np.isin(difficulties, rare_difficulties), most_common_difficulty, difficulties
    )


def split_based_on_difficulty(X_tensors, y_tensors, difficulties):
    train_sss = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_TRAIN_SPLIT, random_state=42
    )
    train_index, test_index = next(train_sss.split(X_tensors, difficulties))
    difficulties_train = difficulties[train_index]
    difficulties_train_dist = np.round(
        100 * np.bincount(difficulties_train) / len(difficulties_train), 2
    )
    print("Train difficulty distribution (percentage):", difficulties_train_dist)

    X_train_tensors, X_temp, y_train_tensors, y_temp = (
        X_tensors[train_index],
        X_tensors[test_index],
        y_tensors[train_index],
        y_tensors[test_index],
    )
    difficulties_temp = difficulties[test_index]

    test_sss = StratifiedShuffleSplit(
        n_splits=1, test_size=VAL_TEST_SPLIT, random_state=42
    )
    val_index, test_index = next(test_sss.split(X_temp, difficulties_temp))
    X_val_tensors, X_test_tensors, y_val_tensors, y_test_tensors = (
        X_temp[val_index],
        X_temp[test_index],
        y_temp[val_index],
        y_temp[test_index],
    )

    difficulties_val = difficulties_temp[val_index]
    difficulties_val_dist = np.round(
        100 * np.bincount(difficulties_val) / len(difficulties_val), 2
    )
    print("Val difficulty distribution (percentage):", difficulties_val_dist)

    difficulties_test = difficulties_temp[test_index]
    difficulties_test_dist = np.round(
        100 * np.bincount(difficulties_test) / len(difficulties_test), 2
    )
    print("Test difficulty distribution (percentage):", difficulties_test_dist)

    # Calculate the difference between the distributions
    train_val_diff = difficulties_train_dist - difficulties_val_dist
    train_test_diff = difficulties_train_dist - difficulties_test_dist
    val_test_diff = difficulties_val_dist - difficulties_test_dist

    # Print the differences
    print(
        "Difference between train and validation difficulty distribution (percentage):",
        train_val_diff,
    )
    print(
        "Difference between train and test difficulty distribution (percentage):",
        train_test_diff,
    )
    print(
        "Difference between validation and test difficulty distribution (percentage):",
        val_test_diff,
    )

    del X_temp, y_temp

    # Validate splits
    train_difficulty_10_sample = X_train_tensors[np.where(difficulties_train == 10)[0]][
        :5
    ]
    print(train_difficulty_10_sample)
    [puzzle.decode("utf-8").count("0") for puzzle in train_difficulty_10_sample]

    val_difficulty_20_sample = X_val_tensors[np.where(difficulties_val == 20)[0]][:5]
    print(val_difficulty_20_sample)
    [puzzle.decode("utf-8").count("0") for puzzle in val_difficulty_20_sample]

    test_difficulty_30_sample = X_test_tensors[np.where(difficulties_test == 30)[0]][:5]
    print(test_difficulty_30_sample)
    [puzzle.decode("utf-8").count("0") for puzzle in test_difficulty_30_sample]

    print(f"Train size: {len(X_train_tensors)}")
    print(f"Validation size: {len(X_val_tensors)}")
    print(f"Test size: {len(X_test_tensors)}")

    # Sort train data by difficulty
    X_train_tensors_sorted, y_train_tensors_sorted = sort_by_difficulty(
        X_train_tensors, y_train_tensors, difficulties_train
    )

    return (
        X_train_tensors_sorted,
        y_train_tensors_sorted,
        X_val_tensors,
        y_val_tensors,
        X_test_tensors,
        y_test_tensors,
    )


def preprocess_dataset(X_tensors, y_tensors):
    difficulties = calculate_difficulties(X_tensors)
    print("Original difficulty distribution:", np.bincount(difficulties))

    (
        X_train_tensors_sorted,
        y_train_tensors_sorted,
        X_val_tensors,
        y_val_tensors,
        X_test_tensors,
        y_test_tensors,
    ) = split_based_on_difficulty(X_tensors, y_tensors, difficulties)

    print(f"First few items of sorted train dataset:{X_train_tensors_sorted[:5]}")
    print(f"Last few items of sorted train dataset:{X_train_tensors_sorted[-5:]}")

    train_preprocessed_datasets = []
    train_size = len(X_train_tensors_sorted)
    n_splits = 10
    for i in range(n_splits):
        start = (train_size // n_splits) * i
        end = (train_size // n_splits) * (i + 1)

        X_slice = X_train_tensors_sorted[start:end]
        y_slice = y_train_tensors_sorted[start:end]

        train_preprocessed_dataset = tf.data.Dataset.from_tensor_slices(
            (X_slice, y_slice)
        ).map(preprocess_map, num_parallel_calls=tf.data.AUTOTUNE)
        train_preprocessed_datasets.append(train_preprocessed_dataset)

    val_preprocessed_dataset = tf.data.Dataset.from_tensor_slices(
        (X_val_tensors, y_val_tensors)
    ).map(preprocess_map, num_parallel_calls=tf.data.AUTOTUNE)

    test_preprocessed_dataset = tf.data.Dataset.from_tensor_slices(
        (X_test_tensors, y_test_tensors)
    ).map(preprocess_map, num_parallel_calls=tf.data.AUTOTUNE)

    return (
        train_preprocessed_datasets,
        val_preprocessed_dataset,
        test_preprocessed_dataset,
    )


# Limit data size
def limit_dataset(size_limit, train_datasets, val_dataset, test_dataset):
    train_size_limit = int(size_limit * (1 - TEST_TRAIN_SPLIT))
    # Make sure we have at least one batch
    val_size_limit = max(int(size_limit * TEST_TRAIN_SPLIT * VAL_TEST_SPLIT), 1)
    test_size_limit = max(int(size_limit * TEST_TRAIN_SPLIT * VAL_TEST_SPLIT), 1)

    n_train_ds = len(train_datasets)
    for index in range(n_train_ds):
        n_batches = train_size_limit // n_train_ds
        train_datasets[index] = train_datasets[index].take(n_batches)

    return (
        train_datasets,
        val_dataset.take(val_size_limit),
        test_dataset.take(test_size_limit),
    )


# Difficulty


def calculate_difficulties(X_tensors):
    # Count zero entries (given clues) - the more zeros the more difficult is the puzzle
    difficulties = np.array([puzzle.decode("utf-8").count("0") for puzzle in X_tensors])
    return replace_rare_difficulties(difficulties)


def sort_by_difficulty(X_tensors, y_tensors, difficulties):
    # Sort data by increasing difficulty
    sorted_indices = np.argsort(difficulties)
    X_tensors_sorted = X_tensors[sorted_indices]
    y_tensors_sorted = y_tensors[sorted_indices]
    return X_tensors_sorted, y_tensors_sorted
