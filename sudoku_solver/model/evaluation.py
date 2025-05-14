from matplotlib import pyplot as plt
import os
from termcolor import colored

import numpy as np

import tensorflow as tf

import keras

import mlflow

from sudoku_solver.data.preprocess import (
    preprocess_input,
    preprocess_target,
    denormalize_input,
    denormalize_label,
)


def plot_histories(histories, save_to_folder=None):
    merged_histories = merge_histories(histories)

    total_epochs = len(merged_histories["loss"])
    epochs_in_run = total_epochs // len(histories)

    n_columns = 4 if "sudoku_constraint" in merged_histories else 3
    height = 6
    width = 24 if n_columns == 4 else 18

    figure = plt.figure(figsize=(width, height))

    plot_accuracy(merged_histories, n_columns)
    plot_run_lines(total_epochs, epochs_in_run)

    plot_loss(merged_histories, n_columns)
    plot_run_lines(total_epochs, epochs_in_run)

    plot_learning_rate(merged_histories, n_columns)
    plot_run_lines(total_epochs, epochs_in_run)

    if n_columns > 3:
        plot_constraint_loss(merged_histories, n_columns)
        plot_run_lines(total_epochs, epochs_in_run)

    mlflow.log_figure(figure, "learning_curves.png")

    if save_to_folder:
        plt.savefig(os.path.join(save_to_folder, "learning_curves.png"))


def merge_histories(histories):
    merged_histories = histories[0].history
    for history in histories[1:]:
        current_history = history.history
        for key in current_history.keys():
            merged_histories[key] += current_history[key]
    return merged_histories


def plot_run_lines(total_epochs, epochs_in_run):
    for epoch in range(total_epochs):
        if (epoch + 1) % epochs_in_run == 0:
            plt.vlines(x=epoch, ymin=0, ymax=100, color="r", linestyle="-", linewidth=1)


def plot_accuracy(histories, n_columns):
    min_accuracy = min(min(histories["accuracy"]), min(histories["val_accuracy"]))

    plt.subplot(1, n_columns, 1)
    plt.plot(histories["accuracy"], label="accuracy")
    plt.plot(histories["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([min_accuracy, 1])
    plt.legend(loc="lower left")


def plot_loss(histories, n_columns):
    max_loss = max(max(histories["loss"]), max(histories["val_loss"]))

    plt.subplot(1, n_columns, 2)
    plt.plot(histories["loss"], label="loss")
    plt.plot(histories["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim([0, max_loss + max_loss / 10])  # buffer so max value is visible
    plt.legend(loc="lower left")


def plot_learning_rate(histories, n_columns):
    max_learning_rate = max(histories["learning_rate"])

    plt.subplot(1, n_columns, 3)
    plt.plot(histories["learning_rate"], label="learning_rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.ylim(
        [0, max_learning_rate + max_learning_rate / 10]
    )  # buffer so max value is visible
    plt.legend(loc="lower left")


def plot_constraint_loss(histories, n_columns):
    max_constraint_loss = max(histories["sudoku_constraint"])

    plt.subplot(1, n_columns, 4)
    plt.plot(histories["sudoku_constraint"], label="sudoku_constraint")
    plt.xlabel("Epoch")
    plt.ylabel("Sudoku constraint loss")
    plt.ylim(
        [0, max_constraint_loss + max_constraint_loss / 10]
    )  # buffer so max value is visible
    plt.legend(loc="lower left")


def evaluate_on_difficulties(model, train_datasets):
    difficulty_to_accuracy = {}
    for index, train_dataset in enumerate(train_datasets):
        batch_count = 10
        total_loss = 0.0
        total_accuracy = 0.0
        total_non_zero_count = 0

        for X_batch, y_batch in train_dataset.take(batch_count):
            y_batch_without_flag = tf.cast(
                tf.where(y_batch >= 10, y_batch - 10, y_batch), tf.float32
            )

            loss, accuracy, *rest = model.evaluate(
                X_batch, y_batch_without_flag, verbose=0
            )
            total_loss += loss
            total_accuracy += accuracy
            total_non_zero_count += tf.reduce_mean(
                tf.math.count_nonzero(denormalize_input(X_batch), axis=(1, 2))
            )

        avg_loss = total_loss / batch_count
        avg_accuracy = total_accuracy / batch_count
        print(f"Difficulty {index + 1}: loss={avg_loss}, accuracy={avg_accuracy}")
        difficulty_to_accuracy.update({f"Difficulty {index + 1}": avg_accuracy})

        avg_non_zero = int(total_non_zero_count / batch_count)
        print(f"Average non-zero numbers in puzzle in one batch: {avg_non_zero}\n")
    return difficulty_to_accuracy


def evaluate_replacing_fixed_positions(model, test_dataset):
    x_test = []
    y_true = []

    for X, y in test_dataset:
        x_test.append(X.numpy())
        y_true.append(y.numpy())

    x_test = np.concatenate(x_test, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    y_pred = model.predict(x_test, verbose=0)
    # predictions are 0-based but puzzle solution numbers start with 1
    y_pred_classes = denormalize_label(np.argmax(y_pred, axis=-1))

    # revert preprocessing so numbers are as in puzzles
    x_test_fixed = denormalize_input(x_test).reshape((-1, 9, 9))
    y_true_fixed = denormalize_label(y_true)

    # replace fixed positions in predictions with actual fixed numbers
    y_pred_fixed = np.where(x_test_fixed == 0, y_pred_classes, x_test_fixed)

    accuracy_metric = keras.metrics.Accuracy()
    accuracy_metric.update_state(y_true_fixed, y_pred_fixed)
    fixed_accuracy = accuracy_metric.result().numpy()
    print(f"Test Set Accuracy after copying Fixed Numbers: {fixed_accuracy:.8f}")
    return fixed_accuracy


def evaluate_puzzle(model, puzzle, solution):
    print(f"\nPuzzle:")
    print_in_sudoku_format(puzzle)

    print(f"\nActual solution:")
    print_in_sudoku_format(solution)

    prediction = solve_sudoku(model, puzzle)
    print(f"\nPredicted solution:")
    compare_prediction(puzzle, solution, prediction)

    loss, accuracy, *rest = model.evaluate(
        prepare_input_batch(puzzle), prepare_target_batch(solution), verbose=0
    )
    non_zero_count = len(puzzle) - puzzle.count("0")
    print(f"\nAccuracy on puzzle with {non_zero_count} non-zero numbers is {accuracy}")


def prepare_input_batch(puzzle):
    reshaped = preprocess_input(puzzle)
    reshaped_batch = tf.expand_dims(reshaped, axis=0)
    return reshaped_batch


def prepare_target_batch(solution):
    reshaped = preprocess_target(solution)
    reshaped_batch = tf.expand_dims(reshaped, axis=0)
    return reshaped_batch


def solve_sudoku(model, puzzle):
    reshaped_batch = prepare_input_batch(puzzle)
    predictions = model.predict(reshaped_batch, verbose=0)
    result = denormalize_label(np.argmax(predictions, axis=-1))
    return result


def print_in_sudoku_format(puzzle_text):
    puzzle_list = np.array(list(puzzle_text)).reshape((9, 9)).tolist()
    for row in puzzle_list:
        print(" ".join(map(str, row)))


def compare_prediction(puzzle, solution, prediction):
    puzzle_list = np.array(list(puzzle)).reshape((9, 9)).tolist()
    solution_list = np.array(list(solution)).reshape((9, 9)).tolist()
    prediction_list = np.array(list(prediction)).reshape((9, 9)).tolist()

    for row_index, row in enumerate(prediction_list):
        chars = []
        for column_index, column in enumerate(row):
            puzzle_char = puzzle_list[row_index][column_index]
            solution_char = solution_list[row_index][column_index]
            prediction_char = str(prediction_list[row_index][column_index])
            error_color = "yellow" if puzzle_char == "0" else "red"

            if solution_char == prediction_char:
                chars.append(colored(prediction_char, "green"))
            else:
                chars.append(colored(prediction_char, error_color))

        # Do not use print to preserve colors also when piping output to file
        colored_text = " ".join(chars)
        print(colored_text)
