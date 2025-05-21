import numpy as np
import glob
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_waveform_segment_from_label(lfp, label, max_segments=400):
    """
    Get waveform segments from the single channal LFP signal and the label of single unit.

    Parameters
    ----------
    lfp: 1D array
        the LFP signal after summing all the simulated neurons in the same channel
    label: 1D array
        the ground-truth label of single unit
    max_segments: int
        the maximum number of waveform segments to extract
    Returns
    -------
    wf_segment: 2D array
        the waveform segments, shape (max_segments, segment_length)
    wf_label: 1D array
        the unique labels in the segments, shape (max_segments,)
    """
    segment = []
    segment_label = []
    wf_label = []
    wf_segment = []

    for idx, value in enumerate(label):
        if value > 0:
            segment.append(lfp[idx])
            segment_label.append(value)
        else:
            if segment:
                wf_segment.append(segment)
                wf_label.append(segment_label)
                segment = []
                segment_label = []
                if len(wf_segment) >= max_segments:
                    break

    # if segment and len(wf_segment) < max_segments:
    #     wf_segment.append(segment)
    #     wf_label.append(segment_label)

    if len(wf_segment) < max_segments:
        return [], []

    try:
        total_label = np.array(wf_label)
        wf_segment = np.array(wf_segment)

    except ValueError:
        return [], []
    
    total_label = np.unique(total_label, axis=1).flatten()
    # print(np.array(wf_segment).shape, total_label.shape)

    return wf_segment, total_label

def get_spike_array(lfp, label, channel_num, max_segments=400):
    """
    Combine the single unit waveform segments of all channels into a 3D array.

    Parameters
    ----------
    lfp: 2D array
        the summed LFP signal, shape (channel_num, time_length)
    label: 1D array
        the ground-truth label of single unit, shape (time_length,)
    channel_num: int
        the number of channels, shape (channel_num,)
    max_segments: int
        the maximum number of waveform segments to extract
    Returns
    -------
    spikes_array: 3D array
        the waveform segments, shape (channel_num, max_segments, segment_length)
    """
    spikes_array = []
    for i in range(channel_num):
        lfp_sc = lfp[i,:]
        wf = get_waveform_segment_from_label(lfp_sc, label, max_segments=max_segments)[0]

        if len(wf) == 0 or len(wf) < max_segments:
            continue  # skip neurons with insufficient segments
        spikes_array.append(wf)

    if not spikes_array:
        return None
    try:
        spikes_array = np.array(spikes_array)
        
    except ValueError:
        return None

    return spikes_array

def combine_spike_array(data_dir, max_segments=400, sum=True):
    """
    Combine the waveform segments of all units into a 3D array.

    Parameters
    ----------
    data_dir: str
        the directory of the lfp and label files
    max_segments: int
        the maximum number of waveform segments for one unit to extract
    sum: bool
        whether to sum the LFP signals of all neurons in the same channel, default is True
    Returns
    -------
    spikes_array: 3D array
        the waveform segments of total units, shape (unit_num * max_segments, channel_num, segment_length)
    """
    lfp_dir = os.path.join(data_dir, "results")
    label_dir = os.path.join(data_dir, "label")

    lfp_file = glob.glob(os.path.join(lfp_dir, "*ep.npy"))
    v_label_file = glob.glob(os.path.join(label_dir, "*v_label.npy"))

    try:
        lfp = np.load(lfp_file[0])
        v_label = np.load(v_label_file[0])

    except IndexError:
        print(f"Skipping folder {data_dir} due to missing files.")
        return None, None
    
    if sum is True:
        lfp_sum = np.sum(lfp, axis=0)

        try:
            spikes_array = np.concatenate([get_spike_array(lfp_sum, v_label[i], 
                                                        channel_num=lfp.shape[1], max_segments=max_segments) 
                                        for i in range(lfp.shape[0])], axis=1)
        except ValueError:

            return None, None

        spikes_array =  np.transpose(spikes_array, (1, 0, 2))
        label_array = np.concatenate([get_waveform_segment_from_label(lfp_sum[0, :], v_label[i], max_segments=max_segments)[1] for i in range(lfp.shape[0])])

        return spikes_array, label_array
    
    else:
        lfp_sum = lfp

        try:
            spikes_array = np.concatenate([get_spike_array(lfp_sum[i], v_label[i], 
                                                        channel_num=lfp.shape[1], max_segments=max_segments) 
                                        for i in range(lfp.shape[0])], axis=1)
        except ValueError:
                
                return None, None
        
        spikes_array =  np.transpose(spikes_array, (1, 0, 2))

        label_array = np.concatenate([get_waveform_segment_from_label(lfp_sum[i, 0, :], v_label[i], max_segments=max_segments)[1] for i in range(lfp.shape[0])])

        return spikes_array, label_array

def create_datasets(exp_folder, save_folder, max_segments=400, file_name = None, sum=True):
    """
    Create the spike and label datasets and save them.

    Parameters
    ----------
    exp_folder: str
        the directory of the experiment
    save_folder: str
        the directory to save the datasets
    max_segments: int
        the maximum number of waveform segments for one unit to extract
    file_name: str
        the name of the dataset files
    """
    data_folder_list = glob.glob(os.path.join(exp_folder, "*/"))
    label_list = []
    spike_list = []
    num_unique = 0

    for folder in tqdm(data_folder_list, desc="Processing folders"):
        spikes_array, label_array = combine_spike_array(folder, max_segments=max_segments, sum=sum)

        if spikes_array is None or label_array is None:
            continue
        if np.isnan(spikes_array).any() or np.isnan(label_array).any():
            print(f"NaN found in folder: {folder}, skipping.")
            continue

        label_array = num_unique + label_array
        if len(np.unique(label_array)) > 0:
            num_unique = max(np.unique(label_array))

        label_list.append(label_array)
        spike_list.append(spikes_array)

    combined_label = np.array(label_list)
    combined_spike = np.array(spike_list)

    np.save(os.path.join(save_folder, file_name+"_data.npy"), combined_spike)
    np.save(os.path.join(save_folder, file_name+"_labels.npy"), combined_label)

    print("Spike and label datasets have been created successfully")
    print("Spike shape: ", combined_spike.shape)
    print("Label shape: ", combined_label.shape)


if __name__ == "__main__":

    exp_name = '6e5_dt0.1_n5-long-L1-L5-1'
    project_name = 'long_simulator'

    container = r'/mnt/storage'
    save_dir = os.getenv('AMLT_OUTPUT_DIR', './')

    project_folder = os.path.join(container, 'projects', project_name, 'amlt-results')

    save_folder = os.path.join(save_dir, exp_name, 'datasets', 'ds_extraction')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    create_datasets(project_folder, save_folder, max_segments=400, 
                    file_name = 'wf400_trial', sum=True)

