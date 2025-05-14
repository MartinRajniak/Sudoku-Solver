import os
import keras
from keras import callbacks
import shutil
from datetime import datetime
import numpy as np

import mlflow

from sudoku_solver.data.dataset import mix_datasets, TOTAL_DATASET_SIZE

LOGS_DIR = "logs"
MODEL_CHECKPOINT_NAME = "sudoku_model_checkpoint.keras"

CURRICULUM_BASE_LEARNING_RATE = 5e-4

def train_model(model, train_datasets, val_dataset, app_config):
    if app_config.USE_CURRICULUM_LEARNING:
        histories = perform_curriculum_training(
            model, train_datasets, val_dataset, app_config
        )
    else:
        histories = perform_normal_training(
            model, train_datasets, val_dataset, app_config
        )
    return histories


def perform_normal_training(model, train_datasets, val_dataset, app_config):
    training_callbacks = prepare_callbacks(app_config)

    train_dataset = None
    # TODO: remove 1 - only for testing
    for dataset in train_datasets[1:]:
        if train_dataset == None:
            train_dataset = dataset
        else:
            train_dataset = train_dataset.concatenate(dataset)

        train_dataset = train_dataset.shuffle(
            app_config.DATA_SIZE_LIMIT
            if app_config.DATA_SIZE_LIMIT
            else TOTAL_DATASET_SIZE
        )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=app_config.EPOCHS,
        callbacks=training_callbacks,
    )
    return [history]


def perform_curriculum_training(model, train_datasets, val_dataset, app_config):
    num_of_difficulties = len(train_datasets)
    # TODO: compare static and progressive loss weights
    constraint_weights = np.linspace(0.1, 0.6, num_of_difficulties)
    fixed_cell_weights = np.linspace(10, 5, num_of_difficulties)

    histories = []
    initial_epoch = 0
    for main_dataset_index in range(num_of_difficulties):
        print(f"Train on difficulty {main_dataset_index}")
        
        # Prepare callbacks
        training_callbacks = prepare_callbacks(app_config)

        # Prepare dataset
        if app_config.USE_DATASET_MIXING:
            train_dataset = mix_datasets(
                train_datasets, main_dataset_index, app_config.PRIMARY_DATASET_SPLIT
            )
        else:
            train_dataset = train_datasets[main_dataset_index]

        # Prepare Learning Rate
        # Start every difficulty with lower learning rate
        print("Resetting learning rate to:", CURRICULUM_BASE_LEARNING_RATE)
        model.optimizer.learning_rate = CURRICULUM_BASE_LEARNING_RATE

        # TODO: prepare LR callback for easy, medium and hard difficulties

        # Prepare Loss Weights
        if (app_config.USE_WEIGHT_SCHEDULING):
            print("Previous constraint loss weight:", model.loss.constraint_weight.numpy())
            print("Previous fixed cell loss weight:", model.loss.fixed_cell_weight.numpy())

            model.loss.constraint_weight = constraint_weights[main_dataset_index]
            model.loss.fixed_cell_weight = fixed_cell_weights[main_dataset_index]

            print("New constraint loss weight:", model.loss.constraint_weight.numpy())
            print("New fixed cell loss weight:", model.loss.fixed_cell_weight.numpy())

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=initial_epoch + app_config.EPOCHS,
            callbacks=training_callbacks,
            initial_epoch=initial_epoch,
        )
        histories.append(history)

        initial_epoch += app_config.EPOCHS
    
    return histories


def prepare_callbacks(app_config):
    # Clear any logs from previous runs
    shutil.rmtree(os.path.join(app_config.ROOT_DIR, LOGS_DIR), ignore_errors=True)

    # TODO: see if it is OK that we create new callback every difficulty
    mlflow_callback = mlflow.keras.MlflowCallback(mlflow.active_run())

    return [
        callbacks.ModelCheckpoint(
            os.path.join(app_config.ROOT_DIR, MODEL_CHECKPOINT_NAME), save_best_only=True, monitor="val_loss"
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
        callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        callbacks.TensorBoard(
            log_dir=os.path.join(
                os.path.join(app_config.ROOT_DIR, LOGS_DIR), "fit", datetime.now().strftime("%Y%m%d-%H%M%S")
            ),
            histogram_freq=1,
            # Uncomment for profile data
            #
            # write_graph=True,
            # write_steps_per_second=True,
            # update_freq="batch",
            # profile_batch='20000,20005'
        ),
        PrintPenalties(),
        # SudokuRulesWeightScheduler(),
        mlflow_callback
    ]


class SudokuRulesWeightScheduler(keras.callbacks.Callback):

    def __init__(self, epochs_per_run):
        super().__init__()
        self.epochs_per_run = epochs_per_run

    def on_epoch_end(self, epoch, logs=None):
        print("constraint_weight ", self.model.loss.constraint_weight)
        print("fixed_cell_weight ", self.model.loss.fixed_cell_weight)

        if epoch == 30:
            self.model.loss.constraint_weight = 1.0
        elif epoch == 60:
            self.model.loss.constraint_weight = 10.0


class PrintPenalties(keras.callbacks.Callback):

    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        print("\nPenalties:")
        print("crossentropy_tracker: ", self.model.loss.crossentropy_tracker.result().numpy())
        print(
            "total_penalty_tracker: ",
            self.model.loss.constraint_penalty_tracker.result().numpy(),
        )
        print("row_penalty_tracker: ", self.model.loss.row_penalty_tracker.result().numpy())
        print("col_penalty_tracker: ", self.model.loss.col_penalty_tracker.result().numpy())
        print("box_penalty_tracker: ", self.model.loss.box_penalty_tracker.result().numpy())
        print("cell_penalty_tracker: ", self.model.loss.cell_penalty_tracker.result().numpy())


# def pretrain_model():
#     if (USE_PRE_TRAINING):
#     pretrain_model = models.clone_model(model)

#     training_callbacks = prepare_callbacks()

#     # Pre-train dataset
#     pretrain_sudoku_ds = tf.data.Dataset.from_tensor_slices(
#         # Learn basic rules only on solutions
#         (y_train_tensors, y_train_tensors)
#     )
#     # Rest is same as during training
#     pretrain_sudoku_reshaped_ds = pretrain_sudoku_ds.map(
#         lambda X, y: (preprocess_input(X), preprocess_target(y)),
#         num_parallel_calls=tf.data.AUTOTUNE,
#     )
#     pretrain_dataset = configure_for_performance(pretrain_sudoku_reshaped_ds, cache_dir="pretrain", shuffle=True)

#     # Pre-train val dataset
#     pretrain_val_sudoku_ds = tf.data.Dataset.from_tensor_slices(
#         # Learn basic rules only on solutions
#         (y_val_tensors, y_val_tensors)
#     )
#     # Rest is same as during training
#     pretrain_val_sudoku_reshaped_ds = pretrain_val_sudoku_ds.map(
#         lambda X, y: (preprocess_input(X), preprocess_target(y)),
#         num_parallel_calls=tf.data.AUTOTUNE,
#     )
#     pretrain_val_dataset = configure_for_performance(pretrain_val_sudoku_reshaped_ds, cache_dir="pretrain_val", shuffle=True)

#     # Train
#     pretrain_history = pretrain_model.fit(
#         pretrain_dataset,
#         epochs=EPOCHS,
#         batch_size=BATCH_SIZE,
#         validation_data=pretrain_val_dataset,
#         callbacks=training_callbacks
#     )

#     del pretrain_sudoku_ds, pretrain_sudoku_reshaped_ds, pretrain_dataset

#     # Transfer learned weights to our actual model
#     # (we only transfer convolutional layers, not the final dense layer)
#     for i, layer in enumerate(model.layers):
#         if isinstance(layer, layers.Conv2D) or isinstance(layer, layers.BatchNormalization):
#             model.layers[i].set_weights(pretrain_model.layers[i].get_weights())

#     del pretrain_model
#     del X_train_tensors, y_train_tensors, X_val_tensors, y_val_tensors, X_test_tensors, y_test_tensors


# Include some percentage of earlier samples in each new batch
def create_mixed_dataset(x_data, y_data, prev_indices, new_indices, mix_ratio=0.2):
    # Sample some previous examples
    num_prev = int(len(new_indices) * mix_ratio)
    if num_prev > 0 and len(prev_indices) > 0:
        sampled_prev = np.random.choice(
            prev_indices, size=min(num_prev, len(prev_indices)), replace=False
        )
        combined_indices = np.concatenate([sampled_prev, new_indices])
    else:
        combined_indices = new_indices

    return x_data[combined_indices], y_data[combined_indices]


# def train_progressively():
#     # Percentage of data set to use
#     data_size_per = [1]

#     training_callbacks = prepare_callbacks()

#     # Progressive training
#     prev_size = 0
#     history_list = []
#     for percentage in data_size_per:
#         # Calculate how many samples to use
#         n_samples = int(percentage * full_dataset_size)
#         # Don't exceed available training samples
#         n_samples = min(n_samples, len(X_train_tensors_sorted))
#         print(
#             f"\n{percentage*100}% of samples included in training, however training only with new {n_samples - prev_size} samples..."
#         )

#         # Get subset of training data
#         X_train_current = X_train_tensors_sorted[prev_size:n_samples]
#         y_train_current = y_train_tensors_sorted[prev_size:n_samples]

#         train_sudoku_ds = tf.data.Dataset.from_tensor_slices(
#             (X_train_current, y_train_current)
#         )
#         train_sudoku_reshaped_ds = train_sudoku_ds.map(
#             lambda X, y: (preprocess_input(X), preprocess_target(y)),
#             num_parallel_calls=tf.data.AUTOTUNE,
#         )
#         train_dataset = configure_for_performance(train_sudoku_reshaped_ds, cache_dir="train", shuffle=True)

#         del X_train_current
#         del y_train_current

#         # Adjust learning rate based on dataset size
#         if percentage <= 0.1:
#             lr = 1e-3
#         elif percentage <= 0.5:
#             lr = 5e-4
#         else:
#             lr = 1e-4

#         # Update optimizer learning rate
#         model.optimizer.learning_rate = lr

#         history = model.fit(
#             train_dataset,
#             validation_data=val_dataset,
#             epochs=EPOCHS,
#             callbacks=training_callbacks,
#         )
#         history_list.append(history)

#         del train_sudoku_ds
#         del train_sudoku_reshaped_ds
#         del train_dataset

#         # Update previous size for next iteration
#         prev_size = n_samples

#     model.save(MODEL_FILE_NAME)
