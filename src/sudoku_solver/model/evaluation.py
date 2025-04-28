from matplotlib import pyplot as plt
import numpy as np


def merge_histories(histories):
    merged_histories = histories[0].history
    for history in histories[1:]:
        current_history = history.history
        for key in current_history.keys():
            merged_histories[key] += current_history[key]
    return merged_histories


def plot_histories(histories):
    merged_histories = merge_histories(histories)

    total_epochs = len(merged_histories["loss"])
    epochs_in_run = total_epochs // len(histories)
    x_ticks = np.arange(0, total_epochs, 1)

    plt.figure(figsize=(20, 6))

    min_accuracy = min(
        min(merged_histories["accuracy"]), min(merged_histories["val_accuracy"])
    )

    plt.subplot(1, 3, 1)
    plt.plot(merged_histories["accuracy"], label="accuracy")
    plt.plot(merged_histories["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([min_accuracy, 1])
    plt.xticks(x_ticks)
    plt.legend(loc="lower left")
    for index, tick in enumerate(x_ticks):
        if (index + 1) % epochs_in_run == 0:
            plt.vlines(x=tick, ymin=0, ymax=10, color="r", linestyle="-", linewidth=1)

    max_loss = max(max(merged_histories["loss"]), max(merged_histories["val_loss"]))

    plt.subplot(1, 3, 2)
    plt.plot(merged_histories["loss"], label="loss")
    plt.plot(merged_histories["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim([0, max_loss + max_loss / 10])  # buffer so max value is visible
    plt.xticks(x_ticks)
    plt.legend(loc="lower left")
    for index, tick in enumerate(x_ticks):
        if (index + 1) % epochs_in_run == 0:
            plt.vlines(x=tick, ymin=0, ymax=10, color="r", linestyle="-", linewidth=1)

    max_learning_rate = max(merged_histories["learning_rate"])

    plt.subplot(1, 3, 3)
    plt.plot(merged_histories["learning_rate"], label="learning_rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.ylim(
        [0, max_learning_rate + max_learning_rate / 10]
    )  # buffer so max value is visible
    plt.xticks(x_ticks)
    plt.legend(loc="lower left")
    for index, tick in enumerate(x_ticks):
        if (index + 1) % epochs_in_run == 0:
            plt.vlines(x=tick, ymin=0, ymax=10, color="r", linestyle="-", linewidth=1)
