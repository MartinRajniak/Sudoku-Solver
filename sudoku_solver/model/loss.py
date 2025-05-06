import tensorflow as tf
import keras


@keras.saving.register_keras_serializable()
class SudokuLoss(keras.losses.Loss):
    def __init__(self, constraint_weight=0.1, name="sudoku_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self._constraint_weight = keras.Variable(
            constraint_weight,
            name="constraint_weight",
            dtype=tf.float32,
            trainable=False,
            aggregation="only_first_replica",
        )

        # Initialize metric trackers
        # These will be automatically collected by Keras during training
        self.crossentropy_tracker = keras.metrics.Mean(name="crossentropy")
        self.cell_penalty_tracker = keras.metrics.Mean(name="cell_penalty")
        self.constraint_penalty_tracker = keras.metrics.Mean(name="constraint_penalty")
        self.row_penalty_tracker = keras.metrics.Mean(name="row_penalty")
        self.col_penalty_tracker = keras.metrics.Mean(name="col_penalty")
        self.box_penalty_tracker = keras.metrics.Mean(name="box_penalty")

    def call(self, y_true, y_pred):
        # Cast to float32 to avoid mixed precision issues
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        # --- Masking ---

        # Create a mask for fixed cells based on y_true.
        is_fixed_mask = tf.cast(
            tf.where(y_true >= 10, 0, 1), tf.float32
        )  # 1 for fixed, 0 for empty
        is_fixed_mask = tf.reshape(is_fixed_mask, (-1, 9, 9))

        # Create a mask for empty cells based on y_true.
        is_empty_mask = tf.cast(
            tf.where(y_true >= 10, 1, 0), tf.float32
        )  # 0 for fixed, 1 for empty
        is_empty_mask = tf.reshape(is_empty_mask, (-1, 9, 9))

        # Subtract mask so we have target digits again
        y_true_digits = tf.reshape(
            tf.cast(tf.where(y_true >= 10, y_true - 10, y_true), tf.int32), (-1, 9, 9)
        )

        # Standard Crossentropy for Empty/Predicted Cells
        mean_cross_entropy = self._masked_cross_entropy(
            y_true_digits, y_pred, is_empty_mask
        )

        fixed_cell_penalty = self._fixed_number_penalty(
            y_true_digits, y_pred, is_fixed_mask
        )
        row_penalty = self._row_penalty(y_pred)
        col_penalty = self._column_penalty(y_pred)
        box_penalty = self._box_penalty(y_pred)

        total_constraint_penalty = self.constraint_weight * (
            row_penalty + col_penalty + box_penalty
        )

        total_loss = (
            # Fixed cell penalty only makes sense if we increase weight against cross-entropy,
            # otherwise we could just use cross-entropy to check fixed number predictions
            mean_cross_entropy
            + total_constraint_penalty
            + fixed_cell_penalty * 10
        )

        # --- Update Metrics ---
        # Use the Loss class's metric attributes directly
        self.crossentropy_tracker.update_state(mean_cross_entropy)
        self.constraint_penalty_tracker.update_state(total_constraint_penalty)
        self.row_penalty_tracker.update_state(row_penalty)
        self.col_penalty_tracker.update_state(col_penalty)
        self.box_penalty_tracker.update_state(box_penalty)
        self.cell_penalty_tracker.update_state(fixed_cell_penalty)

        return total_loss

    @property
    def constraint_weight(self):
        return self._constraint_weight

    @constraint_weight.setter
    def constraint_weight(self, value):
        self._constraint_weight.assign(value)

    # --- Sudoku Constraint Penalties ---

    def _row_penalty(self, y_pred):
        """Row constraint: Sum of probabilities for each digit in a row should be 1."""
        ones = tf.constant(1.0, dtype=tf.float32)

        # Sum over columns -> (batch, 9, 9_digits)
        row_sums = tf.reduce_sum(y_pred, axis=2)
        row_penalty = tf.reduce_mean(tf.square(row_sums - ones))

        return row_penalty

    def _column_penalty(self, y_pred):
        """Column constraint: Sum of probabilities for each digit in a column should be 1."""
        ones = tf.constant(1.0, dtype=tf.float32)

        col_sums = tf.reduce_sum(
            y_pred, axis=1
        )  # Sum over rows -> (batch, 9, 9_digits)
        col_penalty = tf.reduce_mean(tf.square(col_sums - ones))

        return col_penalty

    def _box_penalty(self, y_pred):
        """Box constraint: Sum of probabilities for each digit in a 3x3 box should be 1."""
        ones = tf.constant(1.0, dtype=tf.float32)

        # Reshape to easily sum over boxes: (batch, 3, 3, 3, 3, 9_digits)
        box_probs = tf.reshape(y_pred, [-1, 3, 3, 3, 3, 9])
        # Sum over cells within each box (axes 2 and 4) -> (batch, 3_row_blocks, 3_col_blocks, 9_digits)
        box_sums = tf.reduce_sum(box_probs, axis=[2, 4])
        box_penalty = tf.reduce_mean(tf.square(box_sums - ones))

        return box_penalty

    # --- Masked Penalties (only fixed or only empty cells) ---

    def _fixed_number_penalty(self, y_true, y_pred, is_fixed_mask):
        """Penalty for failed prediction of fixed numbers."""

        tf.debugging.assert_equal(
            tf.reduce_all(tf.shape(y_true)[1:] == tf.constant((9, 9), dtype=tf.int32)),
            tf.constant(True, dtype=tf.bool),
            message="Shape mismatch (ignoring first dimension)",
        )

        # One hot target to be able to compare with predictions
        target_one_hot = tf.one_hot(y_true, 9)

        penalty = tf.reduce_mean(
            tf.reduce_sum(tf.square(y_pred - target_one_hot), axis=-1) * is_fixed_mask
        )
        return penalty

    def _masked_cross_entropy(self, y_true, y_pred, mask):
        """Cross-entropy that only checks empty cells."""

        cross_entropy = keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=False
        )
        masked_cross_entropy = cross_entropy * mask
        mean_cross_entropy = tf.reduce_mean(masked_cross_entropy)
        return mean_cross_entropy

    # TODO: metrics don't show up in model's metrics - I think this only works for Layer/Model subclasses
    # The `metrics` property allows Keras to automatically discover the trackers
    @property
    def metrics(self):
        return [
            self.crossentropy_tracker,
            self.cell_penalty_tracker,
            self.constraint_penalty_tracker,
            self.row_penalty_tracker,
            self.col_penalty_tracker,
            self.box_penalty_tracker,
        ]

    def get_config(self):
        """Returns the config dictionary for serialization."""
        base_config = super().get_config()
        return {
            **base_config,
            "constraint_weight": self.constraint_weight,
        }

    @classmethod
    def from_config(cls, config):
        """Creates a loss instance from its config."""
        return cls(**config)
