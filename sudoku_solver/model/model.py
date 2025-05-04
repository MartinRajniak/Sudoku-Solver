from keras import layers, optimizers, models, regularizers, losses

from sudoku_solver.model.loss import SudokuLoss


def prepare_model(
    use_residual=True,
    learning_rate=1e-3,
    fixed_cell_penalty_weight=0.1,
    constraint_weight=0.1,
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

    outputs = layers.Softmax()(x)

    model = models.Model(inputs, outputs)

    custom_loss = SudokuLoss(constraint_weight=constraint_weight, fixed_cell_weight=fixed_cell_penalty_weight)

    model.compile(
        optimizer=optimizers.Adam(learning_rate),
        loss=custom_loss,
        # loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model
