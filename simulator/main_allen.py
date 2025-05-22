import numpy as np
import os
import LFPy
import MEAutility as mu
import argparse
import random
import yaml
import neuron
import glob
import gc
from stimulate import set_noise_input, get_rheobase_current
from tools import (random_cell_position, random_model_idx, find_spike_idxs, label_spikes,
                   random_probe_location, get_current_time, clear_all)
from models.Allen_model import AllenModel

def print_structure():
    for sec in neuron.h.allsec():
        print(f"Section: {sec.name()} L={sec.L} diam={sec.diam} nseg={sec.nseg}")

class Allen_population_simulator:

    def __init__(self, input_filename, output_filename, script_dir=os.getcwd(), root_path='./'):
        
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.root_path = root_path
        self.script_dir = script_dir
        self.CellParameters = yaml.load(open(self.input_filename, 'r'), Loader=yaml.FullLoader)
        self.model_index = self.get_model_index()
        self.probe = mu.return_mea('tetrode_custom')

        self.cell_name_list = []
        self.lfp_array = []
        self.somav_array = []
        self.label_array = []

    def get_model_index(self):
        self.cell_number = self.CellParameters['num']
        self.cell_model_list = glob.glob('allen_models/n*')
        model_index = random_model_idx(self.cell_model_list, self.cell_number)
        print(model_index)
        return model_index
    
    def setup_electrode(self):
        # seed_probe = random.randint(1, 1e4)
        probe_pos = random_probe_location(self.cell_number, self.probe)
        return probe_pos
    
    def setup_simulation(self, idx, probe_pos=None):
        dt = self.CellParameters['dt']
        end_T = self.CellParameters['end_T']
        sigma = self.CellParameters['sigma']
        seed = random.randint(1, 1e6)

        os.chdir(self.script_dir)
        cell_model_folder = self.cell_model_list[idx]
        
        self.neuron_model = AllenModel(cell_model_folder, start_T=0, end_T=end_T, dt=dt)
        with self.neuron_model:
            cell = self.neuron_model.run()

        self.cell_pos = random_cell_position(self.cell_number, Seed=seed, **self.CellParameters)
        cell.set_pos(x=self.cell_pos[0], y=self.cell_pos[1], z=self.cell_pos[2])
        cell.set_rotation(x=self.CellParameters['rotation'][0], y=self.CellParameters['rotation'][1])
    
        rheobase_current = get_rheobase_current(cell, min_current=0.05, 
                                                   max_current=0.3, current_step=0.05)
        cell.tstop = end_T
        noiseVec, cell, syn, pink_noise = set_noise_input(cell, 
                                                        noise_type='pink_noise', 
                                                        rheobase_current=rheobase_current, 
                                                        amplitude=0.05,
                                                        **self.CellParameters)

        cell.simulate(rec_imem=True)

        electrode = LFPy.RecExtElectrode(cell, probe=self.probe)
        electrode.sigma = sigma

        electrode.x = probe_pos[0]
        electrode.y = probe_pos[1]
        electrode.z = probe_pos[2]

        lfp = np.array(electrode.get_transformation_matrix() @ cell.imem, dtype=np.float16)
        self.lfp_array.append(lfp)
        self.cell_name_list.append(cell_model_folder)
        self.somav_array.append(cell.somav)
        # print_structure()
        del lfp, cell, electrode, noiseVec, syn, pink_noise
        gc.collect()

    def control_lfp_gain(self, lfp):
        """
        lfp: ndarray, shape (n_neuron, n_electrode, n_time)
        """
        lfp = np.array(lfp, dtype=np.float16)
        gain_vector = np.random.uniform(0, 1, size=(4,))
        self.lfp_array = lfp * gain_vector[np.newaxis, :, np.newaxis]

        return self.lfp_array

    def generate_label(self):
        dt = self.CellParameters['dt']
        cuts = self.CellParameters['cut_out']
        soma = np.array(self.somav_array)
        cut_out = [int(cuts[0]/dt), int(cuts[1]/dt)]
        v_label = np.zeros((np.shape(soma)[0], np.shape(soma)[1]))

        # generate label file
        for n in range(np.shape(soma)[0]):
            spike_idx = find_spike_idxs(soma[n,:], thresh=-20, find_max=10)
            spike_labels = label_spikes(soma[n,:], spike_idx, cut_out, label=int(n+1))
            v_label[n,:] = spike_labels

        return v_label

    def save_results(self):

        self.CellParameters['index'] = self.model_index.tolist()
        self.CellParameters['cell_name'] = self.cell_name_list

        config_path = os.path.join(self.root_path, 'setting', self.output_filename)
        results_path = os.path.join(self.root_path, 'results', os.path.splitext(self.output_filename)[0] + '_ep.npy')
        soma_path = os.path.join(self.root_path, 'results_soma', os.path.splitext(self.output_filename)[0] + '_soma.npy')
        label_path = os.path.join(self.root_path, 'label', os.path.splitext(self.output_filename)[0] + '_v_label.npy')

        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        os.makedirs(os.path.dirname(soma_path), exist_ok=True)
        os.makedirs(os.path.dirname(label_path), exist_ok=True)

        current_datetime = get_current_time()

        with open(config_path, 'w') as f:
            f.write(f"# exp_time: {current_datetime}\n")
            yaml.dump(self.CellParameters, f)

        np.save(results_path, self.lfp_array)
        np.save(soma_path, self.somav_array)

        v_label = self.generate_label()

        np.save(label_path, v_label)

    def run(self):
        probe_pos = self.setup_electrode()
        for idx in self.model_index:
            self.setup_simulation(idx, probe_pos)
            clear_all()
        # self.control_lfp_gain(self.lfp_array)
        self.save_results()

if __name__ == '__main__':

    script_dir = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config1.yaml')
    parser.add_argument('--output_config', type=str, default='output.yaml')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    sim = Allen_population_simulator(args.config, args.output_config, script_dir=script_dir, 
                                root_path=os.getenv('AMLT_OUTPUT_DIR', './'))

    # sim = Allen_population_simulator(args.config, args.output_config, script_dir=script_dir, 
    #                               root_path=r'/local_test')
    
    sim.run()