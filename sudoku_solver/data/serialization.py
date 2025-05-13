import os
import tensorflow as tf

# Directory to save the TFRecord files
ROOT_DIR = "./data/sudoku_tfrecords"
TFRECORD_DIR_TRAIN = "train_{:02d}"
TFRECORD_DIR_VAL = "val"
TFRECORD_DIR_TEST = "test"
# Number of TFRecord files to split the data into (e.g., 100 files for 9M examples = 90k examples per file)
NUM_SHARDS = 10
# Consider using compression for smaller file size (at cost of CPU when reading)
COMPRESSION_TYPE = None

def save_to_disk(train_preprocessed_datasets, val_preprocessed_dataset, test_preprocessed_dataset):
    for index, train_preprocessed_dataset in enumerate(train_preprocessed_datasets):
        _save_to_disk(train_preprocessed_dataset, tf_record_dir=TFRECORD_DIR_TRAIN.format(index))
    _save_to_disk(val_preprocessed_dataset, tf_record_dir=TFRECORD_DIR_VAL)
    _save_to_disk(test_preprocessed_dataset, tf_record_dir=TFRECORD_DIR_TEST)

def _save_to_disk(preprocessed_dataset, tf_record_dir):
    tf_record_dir_path = os.path.join(ROOT_DIR, tf_record_dir)

    # Create output directory
    os.makedirs(tf_record_dir_path, exist_ok=True)

    # --- Serialize to TF Examples ---
    # Map the serialization function over the preprocessed dataset
    # Use parallel calls if serialization itself becomes a bottleneck (less likely than preprocessing)
    serialized_dataset = preprocessed_dataset.map(
        lambda x, y: tf.py_function(
            func=lambda a, b: create_tf_example(a, b).SerializeToString(),
            inp=[x, y],
            Tout=tf.string,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Add a 0-based index to each element
    # The dataset elements are now tuples: (index, serialized_example_string)
    indexed_serialized_dataset = serialized_dataset.enumerate()

    tf.data.Dataset.save(
        indexed_serialized_dataset,
        path=tf_record_dir_path,
        compression=COMPRESSION_TYPE,
        # The lambda function now receives the tuple (index, element)
        # Use the 'index' (the first item in the tuple) for sharding
        shard_func=lambda index, element: index % NUM_SHARDS,
    )

    print(
        f"Preprocessing complete. TFRecords saved to {tf_record_dir_path} in {NUM_SHARDS} shards."
    )

def load_from_disk():
    train_preprocessed_datasets = []
    # TODO: magic number 10 - it is actually number of splits defined in preprocess.py (could make it a configuration parameter)
    for index in range(10):
        train_preprocessed_dataset = _load_from_disk(tf_record_dir=TFRECORD_DIR_TRAIN.format(index))
        train_preprocessed_datasets.append(train_preprocessed_dataset)
    val_preprocessed_dataset = _load_from_disk(tf_record_dir=TFRECORD_DIR_VAL)
    test_preprocessed_dataset = _load_from_disk(tf_record_dir=TFRECORD_DIR_TEST)
    return train_preprocessed_datasets, val_preprocessed_dataset, test_preprocessed_dataset

def _load_from_disk(tf_record_dir):
    tf_record_dir_path = os.path.join(ROOT_DIR, tf_record_dir)

    if (
        not os.path.exists(tf_record_dir_path)
        or not os.path.isdir(tf_record_dir_path)
        or not os.listdir(tf_record_dir_path)
    ):
        print(f"Serialized data folder {tf_record_dir_path} not found or empty.")
        return None

    parsed_dataset = tf.data.Dataset.load(
        tf_record_dir_path,
        compression=COMPRESSION_TYPE,
        element_spec=(
            tf.TensorSpec(shape=(), dtype=tf.int64),  # This is the index
            tf.TensorSpec(shape=(), dtype=tf.string),  # This is the serialized example
        ),
    ).map(parse_tf_example, num_parallel_calls=tf.data.AUTOTUNE)
    return parsed_dataset


# Helper function to create a BytesList feature
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# Function to create a tf.train.Example from preprocessed tensors
def create_tf_example(input_tensor, target_tensor):
    """Creates a tf.train.Example message ready to be written to a file."""
    # Convert tensors to byte strings
    input_bytes = tf.io.serialize_tensor(
        input_tensor
    )  # Efficiently serializes tensor to byte string
    target_bytes = tf.io.serialize_tensor(target_tensor)

    # Create a dictionary mapping feature keys to tf.train.Feature proto objects
    feature = {
        "input": _bytes_feature(input_bytes),
        "target": _bytes_feature(target_bytes),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


# --- Define Parsing Function ---
def parse_tf_example(index, example_proto):
    """Parses a serialized tf.train.Example proto into tensors."""
    # Define the features to be parsed
    feature_description = {
        "input": tf.io.FixedLenFeature([], tf.string),  # Stored as byte string
        "target": tf.io.FixedLenFeature([], tf.string),  # Stored as byte string
    }

    # Parse the example
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)

    # Deserialize the byte strings back into tensors
    # Need to know the original shape and dtype
    input_tensor = tf.io.parse_tensor(parsed_features["input"], out_type=tf.double)
    target_tensor = tf.io.parse_tensor(parsed_features["target"], out_type=tf.int32)

    # Reshape back to the original (9, 9, 1) shape (parse_tensor might flatten)
    input_tensor = tf.reshape(input_tensor, (9, 9, 1))
    target_tensor = tf.reshape(target_tensor, (9, 9, 1))

    return input_tensor, target_tensor
