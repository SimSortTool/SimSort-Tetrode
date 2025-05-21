import torch
import numpy as np
from scipy.signal import butter, lfilter, filtfilt

def split_data(data, num_train, num_val, num_test):
    """
    Split data into train, val, and test sets.
    Parameters
    ----------
    data: numpy.ndarray
        spike data that is formatted in shape (num_spikes, n_chans, T)
    num_train:
        amount of data for the training set
    num_val: int
        amount of data for the validation set
    num_test: int
        amount of data for the test set
    Returns
    -------
    train_set: numpy.ndarray
    val_set: numpy.ndarray
    test_set: numpy.ndarray
    """
    train_set = []
    val_set = []
    test_set = []
    tot_num = num_train + num_val + num_test
    n_div = int(len(data) / tot_num)
    for i in range(n_div):
        # (N, train + test, num_chans_extract, 121)
        start = (num_train + num_val + num_test) * i
        train_set.append(data[start : start + num_train])
        val_set.append(data[start + num_train : start + num_train + num_val])
        test_set.append(
            data[start + num_train + num_val : start + num_train + num_val + num_test]
        )

    return np.concatenate(train_set), np.concatenate(val_set), np.concatenate(test_set)

def reconstruct(data, k=8):
    """
    Reconstruct the data from the first k principal eigenvectors.

    Args:
        data (torch.Tensor): shape (batch_size, num_channels, length) on GPU.
        k (int): Number of principal eigenvectors to use for reconstruction.

    Returns:
        torch.Tensor: shape (batch_size, num_channels, length) containing the reconstructed data.
    """
    data_trans = data.view(data.size(0), -1)
    data_centered = data_trans - data_trans.mean(dim=1, keepdim=True)

    u, d, v = torch.svd(data_centered)

    u_k = u[:, :k] # Take the first k columns of eigenvectors
    d_k = torch.diag(d[:k]) # Take the first k eigenvalues and construct a diagonal matrix
    v_k = v[:, :k]  # Take the first k rows of eigenvectors

    # Reconstruct the data
    data_reconstructed = torch.mm(torch.mm(u_k, d_k), v_k.t())
    data_reconstructed = data_reconstructed.view(data.size())

    return data_reconstructed

def bandpass_filter(data, lowcut, highcut=None, fs=10000, order=4):
    nyquist = 0.5 * fs  # Nyquist frequency
    if highcut is None:
        highcut = 0.5 * fs - 1
    low = lowcut / nyquist
    high = highcut / nyquist
    try:
        b, a = butter(order, [low, high], btype='band')
    except ValueError:
        highcut = 0.5 * fs - 1
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        print(f"Highcut frequency is too high. Set to {highcut} Hz.")
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[1]):  # Iterate over each channel
        filtered_data[:, i] = filtfilt(b, a, data[:, i])
    return filtered_data

def highpass_filter(data, lowcut, fs, order=4):
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist  # Normalized lowcut frequency
    b, a = butter(order, low, btype='high')
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[1]):  # Iterate over each channel
        filtered_data[:, i] = filtfilt(b, a, data[:, i])
    return filtered_data

def spatial_whitening(data):
    cov_matrix = np.cov(data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    whitening_matrix = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    whitened_data = data @ whitening_matrix.T 
    return whitened_data

def zscore(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1
    return (data - mean) / std
