import keras
from keras import layers, optimizers, losses, models, regularizers
import tensorflow as tf

def prepare_model(
        use_residual = True,
        learning_rate = 1e-3,
        use_constraint_layer = False,
        constraint_weight = 0.2,
):
    inputs = layers.Input((9, 9, 1))

    # Initial feature extraction
    x = layers.Conv2D(64, kernel_size=3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Residual blocks for deeper learning
    if use_residual:
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
    
    if (use_constraint_layer):
        # Add the constraint layer before the softmax
        x = SudokuConstraintLayer(constraint_weight=constraint_weight)(x)

    outputs = layers.Softmax()(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=optimizers.Adam(learning_rate),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model

# Define the Sudoku Constraint Layer
@keras.saving.register_keras_serializable()
class SudokuConstraintLayer(layers.Layer):
    """Custom layer that enforces Sudoku rules using differentiable constraints"""
    
    def __init__(self, constraint_weight=0.1, **kwargs):
        super(SudokuConstraintLayer, self).__init__(**kwargs)
        self.constraint_weight = constraint_weight
        
    def build(self, input_shape):
        super(SudokuConstraintLayer, self).build(input_shape)
    
    def call(self, inputs):
        # We'll compute all constraints in float32 to avoid precision issues
        # then cast back to the input's dtype at the end
        inputs_f32 = tf.cast(inputs, tf.float32)
        
        # Apply softmax in float32
        probs = tf.nn.softmax(inputs_f32, axis=-1)
        ones = tf.constant(1.0, dtype=tf.float32)
        
        # Row constraint: each number should appear once in each row
        row_sums = tf.reduce_sum(probs, axis=1)  # (batch, 9, 9)
        row_penalty = tf.reduce_mean(tf.square(row_sums - ones))
        
        # Column constraint: each number should appear once in each column
        col_sums = tf.reduce_sum(probs, axis=2)  # (batch, 9, 9)
        col_penalty = tf.reduce_mean(tf.square(col_sums - ones))
        
        # Box constraint: each number should appear once in each 3x3 box
        # Reshape to get boxes: (batch, 3, 3, 3, 3, 9)
        box_shape = tf.reshape(probs, [-1, 3, 3, 3, 3, 9])
        box_sums = tf.reduce_sum(box_shape, axis=[2, 4])  # (batch, 3, 3, 9)
        box_penalty = tf.reduce_mean(tf.square(box_sums - ones))
        
        # Cell constraint: each cell should have exactly one number
        cell_sums = tf.reduce_sum(probs, axis=-1)  # (batch, 9, 9)
        cell_penalty = tf.reduce_mean(tf.square(cell_sums - ones))
        
        # Calculate total penalty in float32
        constraint_weight = tf.constant(self.constraint_weight, dtype=tf.float32)
        total_penalty_f32 = constraint_weight * (row_penalty + col_penalty + box_penalty + cell_penalty)
        
        # Add the loss in float32 (TensorFlow's add_loss expects float32)
        self.add_loss(total_penalty_f32)
        
        # Return the inputs unchanged
        return inputs
    
    # This ensures your layer can be properly serialized and deserialized.
    def get_config(self):
        config = super(SudokuConstraintLayer, self).get_config()
        config.update({"constraint_weight": self.constraint_weight})
        return config