from matplotlib import pyplot as plt


def plot_histories(histories):
    merged_histories = merge_histories(histories)

    total_epochs = len(merged_histories["loss"])
    epochs_in_run = total_epochs // len(histories)

    n_columns = 4 if "sudoku_constraint" in merged_histories else 3
    height = 6
    width = 24 if n_columns == 4 else 18

    plt.figure(figsize=(width, height))

    plot_accuracy(merged_histories, n_columns)
    plot_run_lines(total_epochs, epochs_in_run)

    plot_loss(merged_histories, n_columns)
    plot_run_lines(total_epochs, epochs_in_run)

    plot_learning_rate(merged_histories, n_columns)
    plot_run_lines(total_epochs, epochs_in_run)

    if n_columns > 3:
        plot_constraint_loss(merged_histories, n_columns)
        plot_run_lines(total_epochs, epochs_in_run)


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
