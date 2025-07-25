# src/loaders.py
import os
from glob import glob
import numpy as np

def get_training_ids() -> tuple[list, list]:
    """Return sorted list of folder names under data_dir."""
    TRAIN_GLOB = '../data/raw/train_data/*'
    train_paths = sorted(glob(TRAIN_GLOB))
    train_ids = [os.path.basename(p) for p in train_paths]
    return train_paths, train_ids

def get_test_ids() -> tuple[list, list]:
    TEST_GLOB = '../data/raw/test_data/*'
    test_paths = sorted(glob(TEST_GLOB))
    test_ids = [os.path.basename(p) for p in test_paths]
    return test_paths, test_ids

def shuffle_and_split_data(
    ids: list,
    val_ratio: float = 0.2,
    seed: int = 18
) -> tuple[list, list]:
    """Shuffle list of IDs and split into (train_ids, val_ids)."""
    np.random.seed(seed)
    permuted = np.random.permutation(ids)
    n_val = int(len(ids) * val_ratio)
    val_ids = permuted[:n_val].tolist()
    train_ids = permuted[n_val:].tolist()
    return train_ids, val_ids
