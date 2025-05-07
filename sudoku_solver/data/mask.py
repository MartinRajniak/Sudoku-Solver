import tensorflow as tf

@tf.function
def add_fixed_number_mask(X, y):
    """Add fixed number mask from target data"""

    is_empty_mask = X == -0.5  # -0.5 is preprocessed 0

    modified_values = 10 + y  # add 10 so we know that digit was empty in input

    y = tf.where(is_empty_mask, modified_values, y)

    return X, y

def remove_fixed_number_mask(y_true):
    """Remove fixed number mask from target data"""
    return tf.cast(tf.where(y_true >= 10, y_true - 10, y_true), y_true.dtype)
