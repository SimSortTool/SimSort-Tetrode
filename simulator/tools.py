import math
import random
import sys
import os
import pytz
from neuron import h
import numpy as np
import datetime
from pathlib import Path
import matplotlib.pyplot as plt

def clear_all():
    h('forall delete_section()')
    h('objref all')

def get_current_time():
    beijing_tz = pytz.timezone('Asia/Shanghai')
    return datetime.datetime.now(beijing_tz).strftime("%Y-%m-%d %H:%M:%S")

def compile_all_mechanisms(cell_folder, verbose=False):
    os.chdir(Path(cell_folder))
    cell_folder = Path(cell_folder)
    mod_folder = cell_folder / "mods"
    mod_folder.mkdir(exist_ok=True, parents=True)

    neurons = [f for f in cell_folder.iterdir() if "mods" not in str(f) and not f.name.startswith(".")]

    for neuron in neurons:
        for nmodl in (neuron / "mechanisms").iterdir():
            if nmodl.suffix == ".mod":
                while not (cell_folder / "mods" / nmodl.parts[-1]).is_file():
                    if sys.platform == "win32":
                        _command = "copy"
                    else:
                        _command = "cp"
                    if verbose >= 1:
                        print(f"{_command} {nmodl} {cell_folder / 'mods'}")
                    os.system(f"{_command} {nmodl} {cell_folder / 'mods'}")
    starting_dir = os.getcwd()
    os.chdir(str(cell_folder / "mods"))
    os.system("nrnivmodl")
    os.chdir(starting_dir)

def find_spike_idxs(v, thresh=-20, find_max=30):

    spikes_th = [idx for idx in range(len(v) - 1) if v[idx] < thresh < v[idx + 1]]
    spikes = []
    for sp in spikes_th:
        max_idx = np.argmax(v[sp : sp + find_max])
        spikes.append(sp + max_idx)
    return spikes

def get_cutted_spike_waveform(v, num_spikes, cut_out, spike_index):  
    spike_waveform_size = cut_out[0] + cut_out[1]  
    v_spikes = np.zeros((num_spikes, spike_waveform_size))  
    for idx, spike_idx in enumerate(spike_index):  
        start_idx = max(spike_idx - cut_out[0], 0)  
        end_idx = min(spike_idx + cut_out[1], len(v))  
          
        v_spike = v[start_idx:end_idx]  
          
        if len(v_spike) < spike_waveform_size:  
            v_spike = np.pad(v_spike, (0, spike_waveform_size - len(v_spike)), 'constant')  
        v_spikes[idx, :] = v_spike  
      
    return v_spikes  

def label_spikes(v, spike_index, cut_out=[10, 20], label=1):  
    spike_labels = np.zeros(len(v)) # v must be 1D array 
    for spike_idx in spike_index:  
        start_idx = max(spike_idx - cut_out[0], 0)  
        end_idx = min(spike_idx + cut_out[1], len(v))
        spike_labels[start_idx:end_idx] = label
    return spike_labels

def merge_labels(labels):  
    non_zero_labels = list(filter(lambda x: x != 0, labels))  
    if len(non_zero_labels) > 1:  
        merged = '(' + ''.join(map(str, set(non_zero_labels))) + ')'
    elif len(non_zero_labels) == 1:  
        merged = str(int(non_zero_labels[0]))
    else:  
        merged = '0'  
    return merged  

def update_electrode_positions(cell, probe, electrode=None, seed=None):
    if not isinstance(cell, np.ndarray):
        cellx = cell.x.flatten()
        celly = cell.y.flatten()
        cellz = cell.z.flatten()
        electrode.x = probe.positions[:, 1]
        electrode.z = probe.positions[:, 2]
        electrode.y = probe.positions[:, 0]
        x_range = [cellx.min(), cellx.max()]
        #y_range = [celly.min()*0.5, celly.max()*0.5]
        y_range = [-20, 20]
        z_range = [cellz.min(), cellz.max()]
        ele_abs_pos = np.array([electrode.x, electrode.y, electrode.z]).T
        base_point = ele_abs_pos[0]
        relative_positions = [[x - base_point[0], y - base_point[1], z - base_point[2]] for x, y, z in ele_abs_pos]
        if seed is not None:
            np.random.seed(seed)
        rand_base_x = random.uniform(*x_range)
        rand_base_y = random.uniform(*y_range)
        rand_base_z = random.uniform(*z_range)
        random_position = [[rand_base_x + dx, rand_base_y + dy, rand_base_z + dz] for dx, dy, dz in relative_positions]
        electrode.x, electrode.y, electrode.z = np.array(random_position).T
        return electrode.x, electrode.y, electrode.z
    else:
        x_range = cell[0]
        y_range = cell[1]
        z_range = cell[2]
        ele_x = probe.positions[:, 1]
        ele_y = probe.positions[:, 0]
        ele_z = probe.positions[:, 2]
        ele_abs_pos = np.array([ele_x, ele_y, ele_z]).T
        base_point = ele_abs_pos[0]
        relative_positions = [[x - base_point[0], y - base_point[1], z - base_point[2]] for x, y, z in ele_abs_pos]
        if seed is not None:
            np.random.seed(seed)
        rand_base_x = random.uniform(*x_range)
        rand_base_y = random.uniform(*y_range)
        rand_base_z = random.uniform(*z_range)
        random_position = [[rand_base_x + dx, rand_base_y + dy, rand_base_z + dz] for dx, dy, dz in relative_positions]
        probe_x, probe_y, probe_z = np.array(random_position).T
        return probe_x, probe_y, probe_z
    
# functions for population simulation
def random_model_idx(cell_model_list, cell_number):
    folder_size = len(cell_model_list)
    if folder_size == 0:
        raise ValueError("The population model folder is empty")
    if cell_number <= folder_size:
        model_index = np.random.choice(folder_size, 
                                       cell_number, 
                                       replace=False)
    else:
        raise ValueError("The number of models is less than the number of cells")
    return model_index

def random_cell_position(cell_number, Seed=None, **population_parameters):
    x_range = np.array(population_parameters['x_range']) * cell_number
    y_range = np.array(population_parameters['y_range']) * cell_number
    z_range = np.array(population_parameters['z_range']) * cell_number
    if Seed is None:
        seed = population_parameters['seed']
    else:
        seed = Seed
    np.random.seed(seed)
    x_pos = np.random.uniform(x_range[0], x_range[1])
    y_pos = np.random.uniform(y_range[0], y_range[1])
    z_pos = np.random.uniform(z_range[0], z_range[1])
    print(x_pos, y_pos, z_pos)
    return x_pos, y_pos, z_pos


def random_probe_location(cell_number, probe, seed=None):
    # assume the range of cell population 
    x_range = np.array([-10, 10]) * cell_number
    y_range = np.array([-10, 10]) * cell_number
    z_range = np.array([-10, 10]) * cell_number
    
    total_cell_range = np.array([x_range, y_range, z_range])

    probe = update_electrode_positions(total_cell_range, 
                                            probe=probe, 
                                            seed=seed)
    
    return probe

def calculate_sta_waveform(cell, noise, threshold=0, fig=True, **kwargs):
    if 'dt' not in kwargs:
        raise ValueError("Missing required keyword argument 'dt'")
    dt = kwargs['dt']

    if not hasattr(cell, 'somav'):
        raise ValueError("The 'cell' object does not have a 'somav' attribute")
    
    v_mem = cell.somav 
    spike_times = np.where(np.diff((v_mem > threshold).astype(int)) > 0)[0] * dt  
    window_length = 100  # ms
    n_window_points = int(window_length / dt)

    sta = np.zeros(n_window_points)  
    n_spikes = len(spike_times)  

    if n_spikes == 0:
        raise ValueError("No spikes found. Cannot compute STA.")
    
    for spike_time in spike_times:  
        start_index = int((spike_time - window_length) / dt)  
        if start_index >= 0:  
            sta += noise[start_index:start_index + n_window_points]  

    sta /= n_spikes
    if fig:
        fig = plt.figure()
        plt.plot(np.arange(-window_length, 0, dt), sta)  
        plt.axhline(y=0, color='gray', linestyle='--')
        plt.xlabel('Time (ms)')  
        plt.ylabel('Average input (a.u.)')  
        plt.title('Spike-Triggered Average')  
        return sta, fig
    else:
        return sta



