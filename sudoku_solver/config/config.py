import tomllib
from dataclasses import dataclass


@dataclass
class AppConfig:

    # We need to limit data to be able to run faster experiments
    DATA_SIZE_LIMIT: int

    BATCH_SIZE: int
    LEARNING_RATE: float
    EPOCHS: int

    # Use residual connections in convolutional model architecture
    USE_RESIDUAL: bool

    # Idea is to pretrain model on solution-only data (self-supervised learning) like autoencoder
    #
    # Note: There is no performance improvement on small data sample (10%) and few epochs training (1)
    # Turn on when training with more data and epochs to see if it helps
    USE_PRE_TRAINING: bool

    # Disk cache is slower but a must if we can't load everything into a memory
    USE_DISK_CACHE: bool

    # Idea is to let model learn first on easy problems and then gradually increase problem difficulty
    USE_CURRICULUM_LEARNING: bool

    # Idea is to create a layer that would penalise predictions that break Sudoku rules
    CONSTRAINT_WEIGHT_START: float
    CONSTRAINT_WEIGHT_END: float

    # Fixed cell penalty only makes sense if we increase weight against cross-entropy,
    # otherwise we could just use cross-entropy to check fixed number predictions
    FIXED_CELL_WEIGHT_START: float
    FIXED_CELL_WEIGHT_END: float

    USE_WEIGHT_SCHEDULING: bool

    # Replace predictions for fixed numbers with actuall fixed numbers
    #
    # Idea is to force model to ignore fixed numbers
    #
    # Note: so far it seems that model is unable to learn with this turned on
    USE_FIXED_NUMBER_LAYER: bool

    # Mix train datasets so that in each dataset, there is PRIMARY_DATASET_SPLIT of primary difficulty
    # and 1 - PRIMARY_DATASET_SPLIT of already seen difficulties
    #
    # Note: At least for initial training, mixing datasets is detrimental - slower convergence and lower accuracy
    USE_DATASET_MIXING: bool
    PRIMARY_DATASET_SPLIT: float

    ROOT_DIR = ""

    @classmethod
    def from_toml(cls, file_path):
        with open(file_path, "rb") as f:
            config_data = tomllib.load(f)
        return cls(**config_data)
