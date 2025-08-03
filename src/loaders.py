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

def load_gather(sample_id):
    SRC_IDS = [1, 75, 150, 225, 300]
    DATA_DIR = '../data/raw/train_data'
    mats = []
    for sid in SRC_IDS:
        path = os.path.join(DATA_DIR, sample_id, f'receiver_data_src_{sid}.npy')
        mats.append(np.load(path))
    return np.stack(mats, axis=0)  # (5, time, receivers)

def load_data_array(sample_ids: list, receiver_ids: list = [1, 75, 150, 225, 300], receiver_shape: tuple = (10001, 31), model_shape: tuple = (300, 1259)):
    """
    Loads seismic input data and corresponding velocity model labels into NumPy arrays.

    Parameters
    ----------
    sample_ids : list of str
        List of sample IDs to load (each corresponding to a folder under ./data/train_data/).

    receiver_ids : list of int, optional
        List of source positions to load seismic data from. Each ID corresponds to a file named
        'receiver_data_src_{id}.npy'. Default is [1, 75, 150, 225, 300].

    receiver_shape : tuple of int, optional
        Expected shape of each individual receiver data file. Default is (10001, 31), where
        10001 is the number of time steps and 31 is the number of receivers.

    model_shape : tuple of int, optional
        Shape of the velocity model (label) per sample. Default is (300, 1259), where
        300 is the horizontal axis and 1259 is the depth axis.

    Returns
    -------
    data : np.ndarray
        Array of shape (num_samples, num_receivers, time_steps, num_channels) containing seismic input data.

    labels : np.ndarray
        Array of shape (num_samples, model_shape[0], model_shape[1]) containing velocity model labels.

    Notes
    -----
    This function assumes that the directory structure follows:
        ./data/train_data/{sample_id}/receiver_data_src_{receiver_id}.npy
        ./data/train_data/{sample_id}/vp_model.npy
    and that all files exist and match the expected shape.
    """
    
    num_samples = len(sample_ids)
    num_receivers = len(receiver_ids)

    result = np.zeros((num_samples, num_receivers, receiver_shape[0], receiver_shape[1]), dtype=np.float32)
    result2 = np.zeros((num_samples, model_shape[0], model_shape[1]))

    for i, sample_id in enumerate(sample_ids):
        for j, receiver_id in enumerate(receiver_ids):
            arr = np.load(f'../data/raw/train_data/{sample_id}/receiver_data_src_{receiver_id}.npy')
            arr2 = np.load(f'../data/raw/train_data/{sample_id}/vp_model.npy')
            result[i, j, :, :] = arr
            result2[i, :, :] = arr2
    return result, result2

def load_data_array_test(sample_ids: list, receiver_ids: list = [1, 75, 150, 225, 300], receiver_shape: tuple = (10001, 31), model_shape: tuple = (300, 1259)):
    """
    Loads seismic input data and corresponding velocity model labels into NumPy arrays.

    Parameters
    ----------
    sample_ids : list of str
        List of sample IDs to load (each corresponding to a folder under ./data/train_data/).

    receiver_ids : list of int, optional
        List of source positions to load seismic data from. Each ID corresponds to a file named
        'receiver_data_src_{id}.npy'. Default is [1, 75, 150, 225, 300].

    receiver_shape : tuple of int, optional
        Expected shape of each individual receiver data file. Default is (10001, 31), where
        10001 is the number of time steps and 31 is the number of receivers.

    model_shape : tuple of int, optional
        Shape of the velocity model (label) per sample. Default is (300, 1259), where
        300 is the horizontal axis and 1259 is the depth axis.

    Returns
    -------
    data : np.ndarray
        Array of shape (num_samples, num_receivers, time_steps, num_channels) containing seismic input data.

    labels : np.ndarray
        Array of shape (num_samples, model_shape[0], model_shape[1]) containing velocity model labels.

    Notes
    -----
    This function assumes that the directory structure follows:
        ./data/train_data/{sample_id}/receiver_data_src_{receiver_id}.npy
        ./data/train_data/{sample_id}/vp_model.npy
    and that all files exist and match the expected shape.
    """
    
    num_samples = len(sample_ids)
    num_receivers = len(receiver_ids)

    result = np.zeros((num_samples, num_receivers, receiver_shape[0], receiver_shape[1]), dtype=np.float32)
    result2 = np.zeros((num_samples, model_shape[0], model_shape[1]))

    for i, sample_id in enumerate(sample_ids):
        for j, receiver_id in enumerate(receiver_ids):
            arr = np.load(f'../data/raw/test_data/{sample_id}/receiver_data_src_{receiver_id}.npy')
            
            result[i, j, :, :] = arr
            
    return result
