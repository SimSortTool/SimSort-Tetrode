import numpy as np

def calculate_n_match(test_spike_train, gt_spike_train, match_window=0.001, return_match_boolean=False):
    """
    Calculate the number of matches between the sorted spike train and the ground truth spike train.

    Parameters
    ----------
    test_spike_train : np.ndarray
        sorted spike train
    gt_spike_train : np.ndarray
        ground truth spike train
    match_window : float, optional
        matching window, by default 0.001, in seconds (1 ms)
    
    Returns
    -------
    n_match : int
        number of matches
    n_miss : int
        number of missed events
    n_fp : int
        number of false positives
    """
    test_spike_train = np.sort(test_spike_train)
    gt_spike_train = np.sort(gt_spike_train)
    
    n_match = 0
    gt_matched_boolean = np.zeros(len(gt_spike_train), dtype=bool)
    test_matched_boolean = np.zeros(len(test_spike_train), dtype=bool)

    for i, spike in enumerate(test_spike_train):
        matched_indices = np.where(np.abs(gt_spike_train - spike) <= match_window)[0]
        for idx in matched_indices:
            if not gt_matched_boolean[idx]:
                gt_matched_boolean[idx] = True
                n_match += 1
                test_matched_boolean[i] = True
                break

    n_miss = len(gt_spike_train) - n_match
    n_fp = len(test_spike_train) - n_match
    if return_match_boolean:
        return n_match, n_miss, n_fp, test_matched_boolean, gt_matched_boolean
    return n_match, n_miss, n_fp

def calculate_metrics(gt_spike_train, test_spike_train, match_window=0.001):
    """
    Calculate the accuracy, recall, and precision between the sorted spike train and the ground truth spike train.

    Parameters
    ----------
    gt_spike_train : np.ndarray
        ground truth spike train
    test_spike_train : np.ndarray
        sorted spike train
    match_window : float, optional
        matching window, by default 0.001, in seconds (1 ms)
    
    Returns
    -------
    accuracy : float
        accuracy
    recall : float
        recall
    precision : float
        precision
    """
    n_match, n_miss, n_fp = calculate_n_match(test_spike_train, gt_spike_train, match_window)
    
    accuracy = n_match / (n_match + n_miss + n_fp) if (n_match + n_miss + n_fp) > 0 else 0
    precision = n_match / (n_match + n_fp) if (n_match + n_fp) > 0 else 0
    recall = n_match / (n_match + n_miss) if (n_match + n_miss) > 0 else 0
    
    return accuracy, recall, precision

