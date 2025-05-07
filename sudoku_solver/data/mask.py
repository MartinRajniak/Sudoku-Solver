import tensorflow as tf

def mask_target():
    print()

def remove_mask_from_target(y_true):
    return tf.cast(tf.where(y_true >= 10, y_true - 10, y_true), y_true.dtype)
