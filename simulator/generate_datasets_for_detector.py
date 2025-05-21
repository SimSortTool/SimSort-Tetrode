import glob
import os
import numpy as np
from tqdm import tqdm

def process_single_rec(data_dir, segment_length=10000, label_range=[1, 4], dt=0.1):

    lfp_dir = os.path.join(data_dir, "results")
    label_dir = os.path.join(data_dir, "label")

    lfp_file = glob.glob(os.path.join(lfp_dir, "*ep.npy"))
    v_label_file = glob.glob(os.path.join(label_dir, "*v_label.npy"))

    try:
        lfp = np.load(lfp_file[0])
        v_label = np.load(v_label_file[0])

    except IndexError:
        print(f"Skipping folder {data_dir} due to missing files.")
        return None, None, None
    
    lfp = lfp[:, :, :segment_length]
    lfp_sum = np.sum(lfp, axis=0)
    lfp_input = lfp_sum.T
    #print(f'max: {np.max(lfp_input)}, min: {np.min(lfp_input)}')
    labels = v_label[:, :segment_length] 

    offset = int(2 * (1/dt))
    spike_starts = np.where((labels[:, 1:] - labels[:, :-1]) > 0) # tuple, (neuron, spike)
    unit_ids = spike_starts[0]
    spike_starts = spike_starts[1]
    
    spike_peaks = spike_starts + offset
    unit_spike_trains = np.array([spike_peaks, unit_ids]).T
    unit_spike_trains_sorted = sorted(unit_spike_trains, key=lambda x: x[1]) # sort by spike_peak timestamps
    label_sum = np.zeros(lfp_input.shape[0], dtype=int)

    for peak in spike_peaks:
        start = max(0, peak-label_range[0])
        end = min(lfp_input.shape[0], peak+label_range[1])  # [peak-range, peak+range]
        label_sum[start:end] = 1

    min_length = min(lfp_input.shape[0], label_sum.shape[0])
    lfp_input = lfp_input[:min_length]
    label_sum = label_sum[:min_length]

    return lfp_input, label_sum, unit_spike_trains_sorted

def create_datasets(exp_folder, save_folder, segment_length=10000, label_range=[1, 4], dt=0.1, file_name = None):

    label_range[0] = int(label_range[0] * int(0.1/dt))
    label_range[1] = int(label_range[1] * int(0.1/dt))

    print(f"Label range: {label_range}")
    data_folder_list = glob.glob(os.path.join(exp_folder, "*/"))
    label_list = []
    lfp_list = []
    unit_spike_trains_list = []
    num_unique = 0

    for folder in tqdm(data_folder_list, desc="Processing folders"):
        
        lfp_input, label_array, unit_spike_trains = process_single_rec(folder, segment_length=segment_length, label_range=label_range, dt=dt)

        if lfp_input is None or label_array is None or unit_spike_trains is None:
            continue
        
        if np.isnan(lfp_input).any() or np.isnan(label_array).any():
            print(f"NaN found in folder: {folder}, skipping.")
            continue

        label_list.append(label_array)
        lfp_list.append(lfp_input * 1e3)
        unit_spike_trains_list.append(unit_spike_trains)

    combined_label = np.array(label_list)
    combined_lfp = np.array(lfp_list)
    combined_spike_trains = np.array(unit_spike_trains_list, dtype=object)

    np.save(os.path.join(save_folder, file_name+"_data.npy"), combined_lfp)
    np.save(os.path.join(save_folder, file_name+"_labels.npy"), combined_label)
    np.save(os.path.join(save_folder, file_name+"_spike_trains.npy"), combined_spike_trains)

if __name__ == "__main__":

    exp_name = '6e5_dt0.1_n5-long-L1-L5-1'
    project_name = 'long_simulator'

    container = r'/mnt/storage'
    save_dir = os.getenv('AMLT_OUTPUT_DIR', './')

    project_folder = os.path.join(container, 'projects', project_name, 'amlt-results')

    save_folder = os.path.join(save_dir, exp_name, 'datasets', 'ds_detection')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    create_datasets(project_folder, save_folder, segment_length=600000, file_name = 'combined_rec', label_range=[1, 4], dt=0.1)




    
    

        

    
    