import keras
from keras import layers, optimizers, models, regularizers, losses, saving, Metric

import keras_tuner

import tensorflow as tf

from sudoku_solver.data.mask import *
from sudoku_solver.model.loss import SudokuLoss
from sudoku_solver.data.preprocess import denormalize_input
from sudoku_solver.config.config import AppConfig


def prepare_model(
    app_config: AppConfig, hp: keras_tuner.HyperParameters = None
) -> models.Model:
    inputs = layers.Input((9, 9, 1))

    # Initial feature extraction
    x = layers.Conv2D(64, kernel_size=3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Residual blocks for deeper learning
    if app_config.USE_RESIDUAL:
        for _ in range(4):  # Multiple residual blocks to improve feature extraction
            residual = x
            x = layers.Conv2D(64, kernel_size=3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.Conv2D(64, kernel_size=3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            # Skip connection
            x = layers.Add()([x, residual])
            x = layers.Activation("relu")(x)
    else:
        # Alternative deeper network without residual connections
        for _ in range(3):
            x = layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
            x = layers.BatchNormalization()(x)

    # Increase feature capacity with 1x1 convolutions
    x = layers.Conv2D(128, kernel_size=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Incorporate Sudoku structure awareness
    # Row awareness
    row_conv = layers.Conv2D(16, kernel_size=(1, 9), padding="same")(x)
    # Column awareness
    col_conv = layers.Conv2D(16, kernel_size=(9, 1), padding="same")(x)
    # Box awareness (3x3 boxes with dilation)
    box_conv = layers.Conv2D(16, kernel_size=3, padding="same", dilation_rate=3)(x)

    # Combine structure-aware features
    x = layers.Concatenate()([x, row_conv, col_conv, box_conv])
    x = layers.Conv2D(128, kernel_size=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Final prediction layers
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(9 * 9 * 9, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Reshape((9, 9, 9))(x)

    outputs = layers.Softmax()(x)

    if app_config.USE_FIXED_NUMBER_LAYER:
        outputs = FixedNumberLayer()(inputs, outputs)

    model = models.Model(inputs, outputs)

    if hp:
        constraint_weight = hp.Float(
            "constraint_weight", min_value=0.1, max_value=2.0, sampling="log"
        )
        fixed_cell_weight = hp.Float(
            "fixed_cell", min_value=1.0, max_value=20.0, sampling="log"
        )
        learning_rate = hp.Float(
            "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"
        )
    else:
        constraint_weight = app_config.CONSTRAINT_WEIGHT_START
        fixed_cell_weight = app_config.FIXED_CELL_WEIGHT_START
        learning_rate = app_config.LEARNING_RATE

    custom_loss = SudokuLoss(constraint_weight, fixed_cell_weight)

    model.compile(
        optimizer=optimizers.Adam(learning_rate),
        loss=custom_loss,
        # TODO: add simple CE loss as an option in app config
        # loss=losses.SparseCategoricalCrossentropy(),
        metrics=[SudokuAccuracyMetric(), SudokuRulesMetric()],
    )

    return model


@keras.saving.register_keras_serializable()
class SudokuAccuracyMetric(keras.metrics.SparseCategoricalAccuracy):
    def __init__(self, name="accuracy", **kwargs):
        super().__init__(name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_digits = remove_fixed_number_mask(y_true)
        super().update_state(y_true_digits, y_pred, sample_weight)


@keras.saving.register_keras_serializable()
class SudokuRulesMetric(Metric):
    def __init__(self, name="sudoku_rules", **kwargs):
        super().__init__(name=name, **kwargs)
        self.row_penalty_tracker = keras.metrics.Mean(name="row_penalty")
        self.col_penalty_tracker = keras.metrics.Mean(name="col_penalty")
        self.box_penalty_tracker = keras.metrics.Mean(name="box_penalty")

    def update_state(self, y_true, y_pred, sample_weight=None):
        row_penalty = self._row_penalty(y_pred)
        col_penalty = self._column_penalty(y_pred)
        box_penalty = self._box_penalty(y_pred)

        self.row_penalty_tracker.update_state(row_penalty, sample_weight)
        self.col_penalty_tracker.update_state(col_penalty, sample_weight)
        self.box_penalty_tracker.update_state(box_penalty, sample_weight)

    def result(self):
        return {
            self.row_penalty_tracker.name: self.row_penalty_tracker.result(),
            self.col_penalty_tracker.name: self.col_penalty_tracker.result(),
            self.box_penalty_tracker.name: self.box_penalty_tracker.result(),
        }

    # TODO: copied from loss function - find common place

    def _row_penalty(self, y_pred):
        # Row constraint: Sum of probabilities for each digit in a row should be 1.
        row_sums = tf.reduce_sum(
            y_pred, axis=2
        )  # Sum over columns -> (batch, 9, 9_digits)
        ones = tf.ones_like(row_sums)
        row_penalty = tf.reduce_mean(tf.square(row_sums - ones))

        return row_penalty

    def _column_penalty(self, y_pred):
        # Column constraint: Sum of probabilities for each digit in a column should be 1.
        col_sums = tf.reduce_sum(
            y_pred, axis=1
        )  # Sum over rows -> (batch, 9, 9_digits)
        ones = tf.ones_like(col_sums)
        col_penalty = tf.reduce_mean(tf.square(col_sums - ones))

        return col_penalty

    def _box_penalty(self, y_pred):
        # Box constraint: Sum of probabilities for each digit in a 3x3 box should be 1.
        # Reshape to easily sum over boxes: (batch, 3, 3, 3, 3, 9_digits)
        box_probs = tf.reshape(y_pred, [-1, 3, 3, 3, 3, 9])
        # Sum over cells within each box (axes 2 and 4) -> (batch, 3_row_blocks, 3_col_blocks, 9_digits)
        box_sums = tf.reduce_sum(box_probs, axis=[2, 4])
        ones = tf.ones_like(box_sums)
        box_penalty = tf.reduce_mean(tf.square(box_sums - ones))

        return box_penalty


@keras.saving.register_keras_serializable()
class FixedNumberLayer(layers.Layer):
    def __init__(self):
        super().__init__()

    @tf.function
    def call(self, inputs, x):

        # 1. Preprocess Inputs: Scale, round, clip, and cast to integer
        # Assuming original inputs might be e.g., [-0.5, 0.5] or [0, 1]
        # Adjust scaling/offset if your normalization is different
        scaled_inputs = denormalize_input(inputs)
        rounded_inputs = tf.round(scaled_inputs)
        # Ensure values are within the expected range [0, 9]
        clipped_inputs = tf.clip_by_value(rounded_inputs, 0.0, 9.0)
        # Cast to integer type for indexing and comparison
        # Squeeze the last dimension (1) to make it (batch, 9, 9)
        input_digits = tf.squeeze(tf.cast(clipped_inputs, tf.int32), axis=-1)

        # 2. Handle the condition where input_digit == 0
        # Create a boolean mask where input_digits is 0
        is_zero_mask = tf.equal(input_digits, 0)
        # Expand mask dims to match x's rank for broadcasting: (-1, 9, 9, 1)
        is_zero_mask_expanded = tf.expand_dims(is_zero_mask, axis=-1)
        # Where mask is True, take values from x; otherwise, take zeros.
        output_part_if_zero = tf.where(is_zero_mask_expanded, x, tf.zeros_like(x))

        # 3. Handle the condition where input_digit != 0
        # Create one-hot vectors based on input_digits.
        # The depth is 9 (for indices 0-8).
        # input_digits are 1-9. Subtract 1 to get indices 0-8.
        # tf.one_hot handles negative indices (-1 when input_digit was 0) by producing all zeros.
        indices_for_one_hot = input_digits - 1
        output_part_if_nonzero = tf.one_hot(
            indices_for_one_hot,
            depth=9,  # Corresponds to the last dimension size of x
            on_value=1.0,
            off_value=0.0,
            dtype=x.dtype,  # Ensure output dtype matches x
        )

        # 4. Combine the results
        # Where input was 0, output_part_if_nonzero is all zeros (due to index -1).
        # Where input was non-zero, output_part_if_zero is all zeros.
        # So, we can simply add them.
        final_outputs = output_part_if_zero + output_part_if_nonzero

        return final_outputs
