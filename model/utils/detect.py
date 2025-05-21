import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.functional import max_pool2d, avg_pool2d, conv1d, max_pool1d

# x shape (1, N)
def max1d(X, dt):
    """
    maxpooling 1d

    Parameters
    ----------
    X : torch.Tensor
        input signal, shape (1, t)
    dt : int
        half window size

    Returns
    -------
    Xmax : torch.Tensor
        maxpooling result, shape (1, t)

    """
    X = X.unsqueeze(1)
    Xmax = F.max_pool1d(X, kernel_size=2 * dt + 1, stride=1, padding=dt)
    return Xmax.squeeze(1)

def sum1d(X, dt):
    """
    average pooling 1d

    Parameters
    ----------
    X : torch.Tensor
        input signal, shape (1, t)
    dt : int
        half window size
    
    Returns
    -------
    Xsum : torch.Tensor
        average pooling result, shape (1, t)

    """
    X = X.unsqueeze(1)
    Xsum = F.avg_pool1d(X, kernel_size=2 * dt + 1, stride=1, padding=dt) * (2 * dt + 1)
    return Xsum.squeeze(1)

def get_peak_position(X, nt=61, Th_single_ch=2.5, loc_range=5, 
                     long_range=10, device=torch.device('cuda')):
    """
   Get peak postions from the input signal.

    Parameters
    ----------
    X : torch.Tensor
        input signal, shape (1, t)
    nt : int
        waveform length, by default 61
    twav_min : int
        half window size
    Th_single_ch : float
        threshold to detect spikes, by default 2.5
    loc_range : int, optional
        range for local maxima, by default 5
    long_range : int, optional
        range for long maxima, by default 10
    device : torch.device, optional

    Returns
    -------
    xy : torch.Tensor
        location of snippets, shape (t,)
    
    """
    Xabs   = X.abs()
    Xmax   = max1d(Xabs, loc_range)
    ispeak = torch.logical_and(Xmax==Xabs, Xabs > Th_single_ch).float()
    
    ispeak_sum  = sum1d(ispeak, long_range)
    is_peak_iso = ((ispeak_sum==1) * (ispeak==1))

    is_peak_iso[:nt] = 0
    is_peak_iso[-nt:] = 0
    
    xy = is_peak_iso.nonzero().squeeze(1)

    return xy


def detect_spikes(X, nt=61, Th_single_ch=9, loc_range=5, long_range=10, merge_window=10, device=torch.device('cuda'), largest_channel=False):
    """
    Using threshold to detect spikes from the input signal, and merge adjacent peaks just keep the one with the largest amplitude.

    Parameters
    ----------
    X : torch.Tensor
        input signal, shape (c, t) where c is the number of channels and t is the number of time samples.
    nt : int, optional
        The number of time samples used to represent spike waveforms,the default 61 represents 2ms + 1 bin for a sampling rate of 30kHz.
        For a different sampling rate, you may need to adjust accordingly. For example, nt = 81 would be the 2ms equivalent for 40kHz.
    Th_single_ch : float, optional
        threshold, by default 9, in uV
    loc_range : int, optional
        range for local maxima, by default 5
    long_range : int, optional
        range for long maxima, by default 10
    merge_window : int, optional
        merge window size, by default 10
    
    Returns
    -------
    merged_positions : np.ndarray
        detected spike positions, shape (n,)

    """
    num_channels = X.shape[0]
    peak_positions = []

    for channel in range(num_channels):
        channel_signal = X[channel, :]
        xy = get_peak_position(channel_signal, nt, Th_single_ch, loc_range, long_range, device=device)
        peak_positions.append(xy.cpu().numpy())

    # get the largest channel
    Xabs = X.abs()
    Xmean = Xabs.sum(dim=1)
    if largest_channel:
        largest_channel = Xmean.argmax().item()
        Xlargest = X[largest_channel, :]
        signal = Xlargest
        positions = peak_positions[largest_channel]
        signal_abs = signal.abs()

        # combine adjacent positions, keep the one with the largest amplitude
        merged_positions = []
        current_group = [positions[0]]

        for i in range(1, len(positions)):
            if positions[i] - positions[i-1] <= merge_window:
                current_group.append(positions[i])
            else:
                max_pos = max(current_group, key=lambda x: signal_abs[x])
                merged_positions.append(max_pos)
                current_group = [positions[i]]

        if current_group:
            max_pos = max(current_group, key=lambda x: signal_abs[x])
            merged_positions.append(max_pos)
    else:
        all_pos = []
        for ch in range(num_channels):
            signal = X[ch, :]
            positions = peak_positions[ch]
            signal_abs = signal.abs()
            if len(positions) == 0:
                continue
            # combine adjacent positions, keep the one with the largest amplitude
            merged_positions = []
            current_group = [positions[0]]

            for i in range(1, len(positions)):
                if positions[i] - positions[i-1] <= merge_window:
                    current_group.append(positions[i])
                else:
                    max_pos = max(current_group, key=lambda x: signal_abs[x])
                    merged_positions.append(max_pos)
                    current_group = [positions[i]]

            if current_group:
                max_pos = max(current_group, key=lambda x: signal_abs[x])
                merged_positions.append(max_pos)
            all_pos.append(merged_positions)

        # remove duplicates and sort in all_pos
        all_pos = [element for sublist in all_pos for element in sublist]
        all_pos = np.array(all_pos)
        all_pos = np.unique(all_pos)
        all_pos.sort()

        # then remove the pos that are too close, less than min samples
        merged_positions = []
        if len(all_pos) == 0:
            return np.array(merged_positions)
        
        current_group = [all_pos[0]]
        for i in range(1, len(all_pos)):
            if all_pos[i] - all_pos[i-1] <= merge_window:
                current_group.append(all_pos[i])
            else:
                max_pos = max(current_group, key=lambda x: signal_abs[x])
                merged_positions.append(max_pos)
                current_group = [all_pos[i]]
        if current_group:
            max_pos = max(current_group, key=lambda x: signal_abs[x])
            merged_positions.append(max_pos)

    return np.array(merged_positions)

def extract_snippets(X, positions, nt=61, cuts=[1, 2]):
    """
    Extract snippets from the input signal, according to the detected spike positions.

    Parameters
    ----------
    X : np.ndarray
        input signal, shape (c, t) where c is the number of channels and t is the number of time samples.
    positions : np.ndarray
        detected spike positions, shape (n,)
    nt : int, optional
        waveform length, by default 61 for 2ms, 30khz sampling rate.
    cuts : list, optional
        the ratio before and after the peak to cut, by default [1, 2]
    Returns
    -------
    snippets : np.ndarray
        extracted snippets, shape (n, c, nt)

    """
    signal = X.T  # shape (t, c)
    cut_range = np.ceil([cuts[0] / np.sum(cuts) * (nt - 1), cuts[1] / np.sum(cuts) * (nt - 1)])

    pad_before = int(cut_range[0])
    pad_after = int(cut_range[1])
    padded_signal = np.pad(signal, ((pad_before, pad_after), (0, 0)), mode='constant')

    snippets = []
    for pos in positions:
        start = int(pos + pad_before - cut_range[0])
        end = int(pos + pad_before + cut_range[1])
        snippet = padded_signal[start:end, :]
        
        if isinstance(snippet, torch.Tensor):
            snippet = snippet.cpu().numpy()
            snippets.append(snippet)
        else:
            snippets.append(snippet)
    snippets = np.array(snippets)
    return snippets

def merge_adjacent_positions(merged_labels, signal_abs, merge_window=15):
    """
    Merge adjacent positions, keep the one with the largest value.

    Parameters
    ----------
    merged_labels : np.ndarray
        labels after mergement, shape (t,)
    signal_abs : np.ndarray
        absolute value of input signal, shape (t,)
    merge_window : int, optional
        merge window size, by default 10

    Returns
    -------
    np.ndarray
        merged positions
    
    """
    merged_positions = []
    positions = np.argwhere(merged_labels == 1).squeeze(1)
    
    current_group = [positions[0]]

    for i in range(1, len(positions)):
        if positions[i] - positions[i-1] <= merge_window:
            current_group.append(positions[i])
        else:
            max_pos = max(current_group, key=lambda x: signal_abs[x])
            merged_positions.append(max_pos)
            current_group = [positions[i]]
    if current_group:
        max_pos = max(current_group, key=lambda x: signal_abs[x])
        merged_positions.append(max_pos)

    return np.array(merged_positions)


def merge_predicted_labels(predicted_labels, input_data, merge_window=10, return_positions=False):
    """
    Merge adjacent predicted labels, keep the one with the largest value.

    Parameters
    ----------
    predicted_labels : np.ndarray
        Labels predicted by the detector model, shape (t,)
    input_data : np.ndarray
        Single channel input data, shape (t,)
    merge_window : int, optional
        Merge window size, by default 10
    return_positions : bool, optional
        Whether to return the merged positions or merged labels, by default False

    Returns
    -------
    np.ndarray
        Merged labels or positions
    """
    merged_labels = np.zeros_like(predicted_labels)
    peak_indices = np.where(predicted_labels == 1)[0]
    
    if len(peak_indices) == 0:
        # If no peaks detected, return empty or unchanged results
        if return_positions:
            return np.array([])  # No positions
        return merged_labels  # All zeros
    
    # Identify groups of adjacent peaks
    groups = []
    group = [peak_indices[0]]
    for idx in range(1, len(peak_indices)):
        if peak_indices[idx] - peak_indices[idx - 1] > 1:
            groups.append(group)
            group = []
        group.append(peak_indices[idx])
    groups.append(group)  # Add the last group

    # Merge within each group
    for group in groups:
        start, end = group[0], group[-1] + 1
        max_idx = start + np.argmax(np.abs(input_data[start:end]))
        merged_labels[max_idx] = 1

    if return_positions:
        # Return merged positions using a sliding window approach
        merged_positions = []
        for group in groups:
            start, end = group[0], group[-1] + 1
            signal_segment = np.abs(input_data[start:end])
            max_idx = start + np.argmax(signal_segment)
            
            # Check within merge_window range to avoid multiple close peaks
            if len(merged_positions) == 0 or max_idx - merged_positions[-1] > merge_window:
                merged_positions.append(max_idx)
        return np.array(merged_positions)
    
    return merged_labels

    