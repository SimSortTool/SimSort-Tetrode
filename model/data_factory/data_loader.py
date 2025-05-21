import os
import random
import glob
import numpy as np
import scipy.io
import hashlib
from copy import deepcopy
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.signal import butter, lfilter, filtfilt
from torch.utils.data import Dataset, DataLoader

import spikeinterface.full as si
import spikeforest as sf
from benchmark.HYBRID_JANELIA.get_unitID import UnitIDManager

from utils.nex_api import DataProcessor
from utils.tool import bandpass_filter, spatial_whitening, highpass_filter, zscore

class SFRecordingLoaderBase:
    def __init__(self, 
                 file_path: str = None,
                 label_range: list = [7, 8],
                 preprocess_with_si: bool = False,
                 mounted_blob_dir: str = './',
                 flag: str = 'test',
                 **kwargs):
        if file_path is not None:
            self.file_path = os.path.join(mounted_blob_dir, file_path)
        else:
            self.file_path = self._determine_file_path(mounted_blob_dir, **kwargs)

        self.npy_file_path = os.path.splitext(self.file_path)[0] + ".npy"
        print(f"Loading data from: {self.file_path}")
        
        self.label_range = label_range 
        self.preprocess_with_si = preprocess_with_si
        self.flag = flag if flag is not None else kwargs.get('flag', 'test')
        self.standardize_data = False
        self.study_name = None
        self.recording_name = None
        self.raw_data = None
        self.data = None
        self.recording_extractor = None
        self.unit_spike_trains = None
        self.unit_ids = None
        self.sampling_frequency = None
        self.no_preprocess = False
        
        self._initialize_recording()

    def _determine_file_path(self, mounted_blob_dir, **kwargs):
        raise NotImplementedError("Subclasses must implement _determine_file_path method.")

    def _initialize_recording(self):
        if os.path.exists(self.npy_file_path):
            print(f"Loading data from cached npy file: {self.npy_file_path}")
            self.data = np.load(self.npy_file_path)
            self.unit_spike_trains = np.load(os.path.splitext(self.npy_file_path)[0] + "_spike_trains.npy")
            self.unit_ids = np.load(os.path.splitext(self.npy_file_path)[0] + "_unit_ids.npy")
            self.labels = np.load(os.path.splitext(self.npy_file_path)[0] + "_labels.npy")
            self.sampling_frequency = np.load(os.path.splitext(self.npy_file_path)[0] + "_sampling_frequency.npy").item()
        else:
            print(f"Cached npy file not found. Loading raw data from: {self.file_path}")
            self._load_and_cache_raw_data()

    def _load_and_cache_raw_data(self):
        self.study_name = os.path.basename(os.path.dirname(self.file_path))
        self.recording_name = os.path.splitext(os.path.basename(self.file_path))[0]
        self.raw_data = self._load_raw_data()
        self.recording_extractor = self.raw_data.get_recording_extractor()
        self.sampling_frequency = self.recording_extractor.get_sampling_frequency()
        self.data = self.preprocess()
        self.unit_spike_trains, self.unit_ids = self.get_true_sorting()
        self.labels = self.get_labels()
        np.save(self.npy_file_path, self.data)
        np.save(os.path.splitext(self.npy_file_path)[0] + "_spike_trains.npy", self.unit_spike_trains)
        np.save(os.path.splitext(self.npy_file_path)[0] + "_labels.npy", self.labels)
        np.save(os.path.splitext(self.npy_file_path)[0] + "_unit_ids.npy", self.unit_ids)
        np.save(os.path.splitext(self.npy_file_path)[0] + "_sampling_frequency.npy", np.array([self.sampling_frequency]))

    def _load_raw_data(self):
        raise NotImplementedError("Subclasses must implement _load_raw_data method.")

    def preprocess(self):
        if self.recording_extractor is None:
            raise ValueError("Recording data is not loaded.")

        if self.preprocess_with_si:
            rec = si.bandpass_filter(self.recording_extractor, freq_min=300, freq_max=5000)
            rec = si.common_reference(rec, reference="global", operator="median")
            rec = si.zscore(rec)
            return rec.get_traces()
        else:
            raw_data = self.recording_extractor.get_traces()
            return self._custom_preprocessing(raw_data)

    def _custom_preprocessing(self, raw_data):
        raise NotImplementedError("Subclasses must implement _custom_preprocessing method.")

    def get_trace(self):
        if self.no_preprocess:
            return self.recording_extractor.get_traces()
        rec = self.preprocess()
        return rec.get_traces() if self.preprocess_with_si else rec

    def get_true_sorting(self):
        sorting_true = self.raw_data.get_sorting_true_extractor()
        self.unit_ids = self._manage_unit_ids() if hasattr(self, '_manage_unit_ids') else sorting_true.get_unit_ids()
        self.unit_spike_trains = self._get_unit_spike_trains(sorting_true)
        return self.unit_spike_trains, self.unit_ids

    def _get_unit_spike_trains(self, sorting_true):
        unit_spike_trains = []
        for unit_id in self.unit_ids:
            spike_train = sorting_true.get_unit_spike_train(unit_id=unit_id)
            unit_spike_trains.extend([[spike, unit_id] for spike in spike_train])
        return np.array(unit_spike_trains)

    def get_labels(self):
        timestamps = self.get_timestamps()
        labels = np.zeros(self.recording_extractor.get_num_frames())

        for timestamp in timestamps:
            start = max(0, timestamp - self.label_range[0])
            end = min(self.data.shape[0], timestamp + self.label_range[1])
            labels[start:end] = 1
        return labels

    def get_data(self):
        if self.flag == 'train':
            self.standardize_data = True
            self.spatial_whitening = False
        else:
            self.standardize_data = False
            self.spatial_whitening = True

        if self.data is None:
            self.data = self.get_trace()
            self.unit_spike_trains, unit_ids = self.get_true_sorting()
            self.labels = self.get_labels()

        if self.spatial_whitening:
            self.data = spatial_whitening(self.data)
        
        print(f'standardize_data: {self.standardize_data}, spatial_whitening: {self.spatial_whitening}')
        return self.data, self.labels, self.unit_spike_trains

    def get_waveforms_and_labels(self, cut: list = [20, 40], waveform_num: int = 500):
        if self.data is None or self.unit_spike_trains is None:
            self.get_data()
        spike_waveforms = []
        for spike in self.unit_spike_trains[:, 0]:
            spike_idx = int(spike)
            spike_waveform = self.data[spike_idx - cut[0]: spike_idx + cut[1], :]
            spike_waveforms.append(spike_waveform)
        spike_waveforms = np.array(spike_waveforms)

        sorted_waveform = []
        sorted_label = []
        for unit_id in self.unit_ids:
            unit_waveform = []
            unit_label = []
            for i in range(self.unit_spike_trains.shape[0]):
                if len(unit_waveform) < waveform_num and self.unit_spike_trains[i, 1] == unit_id:
                    unit_waveform.append(spike_waveforms[i])
                    unit_label.append(unit_id)
            sorted_waveform.append(unit_waveform)
            sorted_label.append(unit_label)

        try:
            sorted_waveform = np.vstack(sorted_waveform)
            sorted_label = np.concatenate(sorted_label)
        except ValueError:
            print('The waveform numbers are not the same for all units')
        return sorted_waveform, sorted_label

    def get_timestamps(self):
        if self.unit_spike_trains is None:
            raise ValueError("Unit spike trains are not available.")
        return np.sort(self.unit_spike_trains[:, 0])

class HybridLoader(SFRecordingLoaderBase):
    def _determine_file_path(self, mounted_blob_dir, **kwargs):
        for key, value in kwargs.items():
            if key.startswith('hybrid') and key.endswith('path'):
                return os.path.join(mounted_blob_dir, value)
        return None

    def _load_raw_data(self):
        return sf.load_spikeforest_recording(study_name=self.study_name, 
                                             recording_name=self.recording_name, 
                                             uri='sha1://43298d72b2d0860ae45fc9b0864137a976cb76e8?hybrid-janelia-spikeforest-recordings.json')

    def _custom_preprocessing(self, raw_data):
        lowcut, highcut = 300, 10000
        rec = bandpass_filter(raw_data, lowcut, highcut, self.sampling_frequency)
        input_mean = np.mean(rec, axis=0)
        input_std = np.std(rec, axis=0)
        input_std[input_std == 0] = 1
        rec = (rec - input_mean) / input_std
        # if self.standardize_data:
        #     scaler = StandardScaler()
        #     rec = scaler.fit_transform(rec)
        return rec

    def _manage_unit_ids(self):
        ID_manager = UnitIDManager()
        return ID_manager(self.study_name, self.recording_name)

class CRCNSRecordingLoader(SFRecordingLoaderBase):
    def _determine_file_path(self, mounted_blob_dir, **kwargs):
        for key, value in kwargs.items():
            if key.startswith('crcns') and key.endswith('path'):
                return os.path.join(mounted_blob_dir, value)
        return None

    def _load_raw_data(self):
        return sf.load_spikeforest_recording(study_name=self.study_name, 
                                             recording_name=self.recording_name, 
                                             uri='sha1://1d343ed7e876ffd73bd8e0daf3b8a2c4265b783c?spikeforest-recordings.json')

    def _custom_preprocessing(self, raw_data):
        lowcut, highcut = 300, 2000
        rec = bandpass_filter(raw_data, lowcut, highcut, self.sampling_frequency)
        if self.standardize_data:
            scaler = StandardScaler()
            rec = scaler.fit_transform(rec)
        return rec

class HybridMultiLoader:
    def __init__(self,
                 file_paths:list=None,
                 label_range:list=[7, 8],
                 preprocess_with_si:bool=False,
                 flag:str='test',
                 **kwargs,
                 ):
        
        if file_paths is not None:
            self.file_paths = file_paths
        else:
            for key, value in kwargs.items():
                if key.startswith('hybrid') and key.endswith('paths'):
                    self.file_paths = value
                    break
            else:
                self.file_paths = []

        if not self.file_paths:
            raise ValueError("No file paths provided for HybridMultiLoader.")
        
        self.label_range = label_range
        self.preprocess_with_si = preprocess_with_si
        self.loaders = [HybridLoader(file_path, label_range, preprocess_with_si) for file_path in self.file_paths]
        self.flag = flag if flag is not None else kwargs.get('flag', 'test')
        self.sampling_frequency = None

    def get_data(self, max_segments:int=500000):
        lfp_list = []
        labels_list = []
        unit_spike_trains_list = []
        for loader in self.loaders:
            if self.flag == 'train':
                loader.flag = 'train'
            else:
                loader.flag = 'test'
            data, labels, unit_spike_trains = loader.get_data()
            data = data[:max_segments]
            labels = labels[:max_segments]
            unit_spike_trains = unit_spike_trains[np.argwhere(unit_spike_trains[:, 0] < max_segments).flatten()]
            self.sampling_frequency = loader.sampling_frequency
            lfp_list.append(data.tolist())
            labels_list.append(labels.tolist())
            unit_spike_trains_list.append(unit_spike_trains.tolist())

        data = np.array(lfp_list, dtype=object)
        labels = np.array(labels_list, dtype=object)
        unit_spike_trains = np.array(unit_spike_trains_list, dtype=object)
        return data, labels, unit_spike_trains

    def get_waveforms_and_labels(self, cut:list=[20, 40], waveform_num:int=500):
        waveform_list = []
        label_list = []
        for loader in self.loaders:
            waveforms, labels = loader.get_waveforms_and_labels(cut, waveform_num)
            waveform_list.append(waveforms)
            label_list.append(labels)
        return np.array(waveform_list), np.array(label_list)
    
class WaveClusLoader:
    def __init__(self, 
                 file_path: str = None, 
                 sampling_frequency: int = 24000, 
                 mounted_blob_dir: str = './', 
                 flag: str = 'test',
                 **kwargs):
        if file_path is not None:
            self.file_path = os.path.join(mounted_blob_dir, file_path)
        else:
            for key, value in kwargs.items():
                if key.startswith('waveclus') and key.endswith('path'):
                    self.file_path = os.path.join(mounted_blob_dir, value)
                    break
            else:
                self.file_path = None

        if self.file_path is None:
            raise ValueError("No valid file_path provided or found in kwargs.")

        self.sampling_frequency = sampling_frequency
        self.flag = flag if flag is not None else kwargs.get('flag', 'test')
        self.trial_num = None
        self.data = None
        self.labels = None
        self.unit_spike_trains = None
        self.standardize_data = None
        self.spatial_whitening = None

    @staticmethod
    def _generate_seed(file_path):
        """Generate a unique seed based on the file path."""
        hash_object = hashlib.md5(file_path.encode())
        return int(hash_object.hexdigest(), 16) % (2**32)  # Use a 32-bit integer seed

    def load_data(self):
        seed = self._generate_seed(self.file_path)
        np.random.seed(seed)

        mat_data = scipy.io.loadmat(self.file_path)
        spike_label = mat_data['spike_class'][0][0][0]
        spike_times = mat_data['spike_times'][0][0][0]
        raw_data = mat_data['data'][0]
        raw_data = np.expand_dims(raw_data, axis=0).T * 10
        raw_data = - raw_data  # Invert the data

        freq_min, freq_max = 300, 6000
        filtered_data = bandpass_filter(raw_data, freq_min, freq_max, self.sampling_frequency)

        # Expand to 4 channels with transformations
        data = np.zeros((filtered_data.shape[0], 4))
        for i in range(4):
            # Channel-specific transformations
            noise = np.random.normal(0, 0.01, size=filtered_data.shape)  # Small random noise
            scale = 1 + np.random.uniform(-0.5, 0)  # Small amplitude scaling
            shift = np.random.randint(-5, 6)  # Random time shift
            shifted_data = np.roll(filtered_data, shift, axis=0)  # Apply time shift
            data[:, i] = scale * shifted_data.squeeze() + noise.squeeze()
        print(f'Random aug the data with *{scale:.2f} scaling, {shift} shift, and 0.01 noise')

        # Generate labels
        labels = np.zeros(data.shape[0])
        for spike_time in spike_times:
            start = max(0, spike_time - 20)
            end = min(data.shape[0], spike_time + 20)
            labels[start:end] = 1

        # Generate unit spike trains
        offset = 20
        unit_spike_trains = np.array([[spike_time + offset, spike_label[j]] for j, spike_time in enumerate(spike_times)])

        self.trial_num = len(np.unique(spike_label))

        if self.standardize_data:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)

        if self.spatial_whitening:
            data = spatial_whitening(data)

        return data, labels, unit_spike_trains
    
    def get_data(self):
        if self.spatial_whitening is None:
            if self.flag == 'train':
                self.standardize_data = True
                self.spatial_whitening = False
            else:
                self.standardize_data = False
                self.spatial_whitening = True
        print(f'standardize_data: {self.standardize_data}, spatial_whitening: {self.spatial_whitening}')

        if self.data is None or self.labels is None or self.unit_spike_trains is None:
            self.data, self.labels, self.unit_spike_trains = self.load_data()
        return self.data, self.labels, self.unit_spike_trains

    def get_waveforms_and_labels(self, cut: list = [20, 40], waveform_num: int = 500):
        if self.spatial_whitening is None:
            if self.flag == 'train':
                self.standardize_data = True
                self.spatial_whitening = False
            else:
                self.standardize_data = False
                self.spatial_whitening = True
        if self.data is None or self.unit_spike_trains is None:
            self.data, self.labels, self.unit_spike_trains = self.load_data()

        waveforms = []
        for unit_id in np.unique(self.unit_spike_trains[:, 1]):
            unit_spike_indices = self.unit_spike_trains[self.unit_spike_trains[:, 1] == unit_id, 0]
            unit_waveforms = [
                self.data[spike_idx - cut[0]: spike_idx + cut[1], :]
                for spike_idx in unit_spike_indices
                if cut[0] <= spike_idx < self.data.shape[0] - cut[1]
            ]
            waveforms.extend(unit_waveforms[:waveform_num])

        sorted_waveform = np.stack(waveforms)
        sorted_label = np.concatenate([[unit_id] * len(waveforms[:waveform_num]) for unit_id in np.unique(self.unit_spike_trains[:, 1])])
        return sorted_waveform, sorted_label

class BbpRecordingLoader:
    """
    Loader for BBP recording data.

    Args:
        exp_name (str): Name of the experiment.
        demo_idx (int): Index of the demo data.
        data_path (str): Path to the BBP recording data.
        filter_data (bool): Whether to filter the data.
        standardize_data (bool): Whether to standardize the data.
        sampling_frequency (int): Sampling frequency of the data.
        label_range (list): Range of the labels.
        unit_spike_trains (bool): Whether to return the unit spike trains.
    """
    def __init__(self,
                 exp_name:str=None,
                 demo_idx:int=0,
                 data_path:str=None,
                 filter_data:bool=True,
                 standardize_data:bool=False,
                 sampling_frequency:int=10000,
                 label_range:list=[1, 4],
                 unit_spike_trains:bool=True,
                 convert_to_mv: bool = False,
                 flag:str='train',
                 **kwargs,
                 ):
        
        self.experiment = exp_name if exp_name is not None else kwargs.get('bbp_exp_name', None)
        self.demo_idx = demo_idx if demo_idx is not None else kwargs.get('bbp_demo_idx', 0)
        self.convert_to_mv = convert_to_mv

        if data_path is not None:
            self.data_path = data_path
        else:
            for key, value in kwargs.items():
                if key.startswith('bbp') and key.endswith('path'):
                    self.data_path = value
                    break
            else:
                self.data_path = None
        print(f"Loading BBP data from: {self.data_path}")
        self.filter_data = filter_data
        self.standardize_data = standardize_data
        self.sampling_frequency = sampling_frequency
        self.label_range = label_range
        self.unit_spike_trains = unit_spike_trains
        self.flag = flag if flag is not None else kwargs.get('flag', 'train')
        self.trial_num = None
        self.filter_data = None
        self.convert_to_mv = None
        self.spatial_whitening = None
        self.standardize_data = None

    def load_data(self):
        if self.data_path is not None:
            data = np.load(glob.glob(os.path.join(self.data_path, '*data.npy'))[0])
            labels = np.load(glob.glob(os.path.join(self.data_path, '*labels.npy'))[0])
            if self.unit_spike_trains:
                unit_spike_trains = np.load(glob.glob(os.path.join(self.data_path, '*spike_trains.npy'))[0], allow_pickle=True)

        elif self.experiment is not None and self.unit_spike_trains:
            data, labels, unit_spike_trains = self._load_data_by_exp()

        elif self.experiment is not None and not self.unit_spike_trains:
            data, labels = self._load_data_by_exp()
            
        else:
            raise ValueError("Either data_path or experiment must be provided.")
        
        if self.convert_to_mv:
            data = data * 1e3

        if len(data.shape) == 3 and len(labels.shape) == 2:
            self.trial_num = data.shape[0]
            data = data.reshape(-1, data.shape[-1]) # t * c
            labels = labels.reshape(-1)

        elif len(data.shape) == 4 and len(labels.shape) == 2:
            self.trial_num = data.shape[0]
            data = data.reshape(-1, 4, data.shape[-1])
            labels = labels.reshape(-1)

        if self.filter_data and self.sampling_frequency:
            fs = self.sampling_frequency
            lowcut = 300
            highcut = 3000
            data = bandpass_filter(data, lowcut, highcut, fs)
        
        if self.standardize_data:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            print('Standardizing data')
        if self.spatial_whitening:
            data = spatial_whitening(data)
            print('Spatial whitening applied')

        if self.unit_spike_trains:
            return data, labels, unit_spike_trains
        else:
            return data, labels
    
    def _load_data_by_exp(self):
        root_path = '/mnt/storage/simulate_data'
        file_path = glob.glob(os.path.join(root_path, self.experiment, 'demo_data', '*'))[self.demo_idx]
        lfp_dir = glob.glob(os.path.join(root_path, file_path, 'results') + '/*ep.npy')
        label_dir = glob.glob(os.path.join(root_path, file_path, 'label') + '/*v_label.npy')

        data = np.load(lfp_dir[0])
        labels = np.load(label_dir[0])

        lfp_sum = np.sum(data, axis=0) # c * t
        lfp_input = lfp_sum.T # t * c

        offset = 20
        spike_starts = np.where((labels[:, 1:] - labels[:, :-1]) > 0) # tuple, (neuron, spike)
        unit_ids = spike_starts[0]
        spike_starts = spike_starts[1]

        spike_peaks = spike_starts + offset
        unit_spike_trains = np.array([spike_peaks, unit_ids]).T
        #nit_spike_trains_sorted = sorted(unit_spike_trains, key=lambda x: x[0])
        label_sum = np.zeros(lfp_input.shape[0], dtype=int)

        for peak in spike_peaks:
            start = max(0, peak-self.label_range[0])
            end = min(lfp_input.shape[0], peak+self.label_range[1])  # [peak-range, peak+range]
            label_sum[start:end] = 1

        min_length = min(lfp_input.shape[0], label_sum.shape[0])
        lfp_input = lfp_input[:min_length]
        label_sum = label_sum[:min_length]

        if self.filter_data and self.sampling_frequency:
            fs = self.sampling_frequency
            lowcut = 300
            highcut = 3000
            lfp_input = bandpass_filter(lfp_input, lowcut, highcut, fs)
        
        if self.standardize_data:
            scaler = StandardScaler()
            lfp_input = scaler.fit_transform(lfp_input)
        
        if self.unit_spike_trains:
            return lfp_input, labels, unit_spike_trains
        else:
            return lfp_input, label_sum

    def get_data(self, **kwargs):
        self.unit_spike_trains = True
        self.convert_to_mv = False

        if self.spatial_whitening is None:
            if self.flag == 'train':
                self.standardize_data = True
                self.filter_data = True
                self.spatial_whitening = False
            elif self.flag == 'test':
                self.standardize_data = False
                self.filter_data = True
                self.spatial_whitening = True

        print(f'standardize_data: {self.standardize_data}, filter_data: {self.filter_data}, spatial_whitening: {self.spatial_whitening}')

        if self.data_path is not None:
            original_data_path = self.data_path
            self.data_path = os.path.join(self.data_path, 'ds_detection')

        data, labels, unit_spike_trains = self.load_data()
        self.data_path = original_data_path
        data = data.reshape(self.trial_num, -1, data.shape[-1])
        labels = labels.reshape(self.trial_num, -1)
        #print(data.shape, labels.shape, unit_spike_trains.shape)
        return data, labels, unit_spike_trains

    def get_waveforms_and_labels(self, **kwargs):
        self.unit_spike_trains = False
        self.filter_data = False
        self.standardize_data = False
        self.convert_to_mv = True

        if self.data_path is not None:
            self.data_path = os.path.join(self.data_path, 'ds_extraction')
        print(self.data_path)
        data, labels = self.load_data()
        data = data.reshape(self.trial_num, -1, 4, data.shape[-1]) # t * n * c * l
        labels = labels.reshape(self.trial_num, -1) # t * n
        #print(data.shape, labels.shape)
        return data, labels

class NexRecordingLoader: 
    def __init__(self, 
                 file_path:str=None,
                 raw_data_path:str=None,
                 sorted_data_path:str=None,
                 tetrode_idx:int=None,
                 label_range:list=[1, 4],
                 flag:str='test',
                 **kwargs):
        
        self.file_path = file_path if file_path is not None else kwargs.get('nex_file_path', None)
        self.sorted_data_path = sorted_data_path if sorted_data_path is not None else kwargs.get('nex_sorted_path', None)
        self.raw_data_path = raw_data_path if raw_data_path is not None else self.sorted_data_path.replace('-sorted.nex', '.nex')
        self.tetrode_idx = tetrode_idx if tetrode_idx is not None else kwargs.get('nex_tetrode_idx', None)
        self.flag = flag if flag is not None else kwargs.get('flag', 'test')

        self.label_range = label_range
        self.sampling_frequency = 40000

    def load_waveform_data(self, return_withlabel_dataset:bool=False):
        processor = DataProcessor(self.file_path)
        processor.read_data(read_spkc=False)
        processor.process_waves()
        dataset_with_labels = processor.get_data_with_labels()
        dataset_without_labels = processor.get_data_without_labels()
        labels = processor.get_labels()

        if return_withlabel_dataset:
            return dataset_with_labels
        
        return dataset_without_labels, labels
    
    def load_continous_data(self, 
                            continous_type='SPKC', 
                            cont_name=None, 
                            standardize_data:bool=False, 
                            print_cont_names:bool=False
                            ):

        processor = DataProcessor(self.file_path)
        processor.read_data(read_spkc=True)

        cont_names = [name[:int(len(continous_type))] == continous_type for name in processor.cont_names]
        cont_names = np.array(processor.cont_names)[cont_names]
        if print_cont_names:
            print(cont_names)

        if self.tetrode_idx is not None and cont_name is None:
            cont_data = np.array([
                processor.get_spkc(spkc_name=cont_names[0+4*self.tetrode_idx]),
                processor.get_spkc(spkc_name=cont_names[1+4*self.tetrode_idx]),
                processor.get_spkc(spkc_name=cont_names[2+4*self.tetrode_idx]),
                processor.get_spkc(spkc_name=cont_names[3+4*self.tetrode_idx]),
            ]).T
        elif cont_name is not None:
            cont_data = np.array([processor.get_spkc(spkc_name=cont_name)]).T # t * c
        else:
            raise ValueError("Either tetrode_idx or cont_name must be provided.")
        
        if standardize_data:
            # input_mean = np.mean(cont_data, axis=0)
            # input_std = np.std(cont_data, axis=0)
            # input_std[input_std == 0] = 1
            # cont_data = (cont_data - input_mean) / input_std
            scaler = StandardScaler()
            cont_data = scaler.fit_transform(cont_data)

        return cont_data

    def load_spike_timestamps(self, channel_name:str='SPKC01', print_neuron_names:bool=False):
        channel_name = 'SPKC' + str(self.tetrode_idx*4+1).zfill(2)

        processor = DataProcessor(self.file_path)
        processor.read_data(read_spkc=False)
        neuron_names = processor.neuron_names
        neurons_selected = [name[:int(len(channel_name))] == channel_name for name in neuron_names]
        neurons_selected = np.array(neuron_names)[neurons_selected]

        timestamps_all = []
        for neuron in neurons_selected:
            timestamps = processor.get_neuron_timestamps(neuron_name=neuron, 
                                                        print_neuron_names=print_neuron_names)
            timestamps_all.append(timestamps)

        timestamps_all = np.concatenate(timestamps_all)
        timestamps_all = np.sort(timestamps_all)

        return timestamps_all
    
    def get_unit_spike_trains(self, channel_name:str='SPKC01', 
                              print_neuron_names:bool=False):
        offset = self._calculate_timestamps_offset()
        channel_name = 'SPKC' + str(self.tetrode_idx*4+1).zfill(2)

        processor = DataProcessor(self.file_path)
        processor.read_data(read_spkc=False)
        neuron_names = processor.neuron_names
        neurons_selected = [name[:int(len(channel_name))] == channel_name for name in neuron_names]
        neurons_selected = np.array(neuron_names)[neurons_selected]

        unit_spike_trains = []
        for idx, neuron in enumerate(neurons_selected):
            timestamps = processor.get_neuron_timestamps(neuron_name=neuron, 
                                                        print_neuron_names=print_neuron_names)
            for timestamp in timestamps:
                timestamp = int(timestamp * 40000) + int(offset)
                unit_spike_trains.append([timestamp, idx])
        
        unit_spike_trains = np.array(unit_spike_trains)
        return unit_spike_trains
    
    def _calculate_timestamps_offset(self):
        self.file_path = self.raw_data_path
        demo_cont_data = self.load_continous_data(cont_name='SPKC01', standardize_data=True)
        demo_cont_data = demo_cont_data[:5000, 0]

        self.file_path = self.sorted_data_path
        timestamps = self.load_spike_timestamps()
        spike_indices = (timestamps * self.sampling_frequency).astype(int)
        demo_spikes = spike_indices[spike_indices < demo_cont_data.shape]

        # offset_list = []
        # offset_range = [-80, 20]

        # for spike in demo_spikes:
        #     start_idx = max(0, spike + offset_range[0])
        #     end_idx = min(demo_cont_data.shape[0], spike + offset_range[1])
        #     local_peak_idx = np.argmin(demo_cont_data[start_idx:end_idx])
        #     offset = local_peak_idx + offset_range[0]
        #     offset_list.append(offset)

        # offset = np.mean(offset_list)
        # select best offset by maximizing the sum of spike values

        best_offset = 0
        best_score = float('-inf')
        offset_range = np.arange(-100, 20, 1)

        for offset in offset_range:
            shifted_spikes = demo_spikes + offset
            valid_spikes = shifted_spikes[(shifted_spikes >= 0) & (shifted_spikes < demo_cont_data.shape[0])]
            spike_values = demo_cont_data[valid_spikes]
            score = np.sum(np.abs(spike_values))

            if score > best_score:
                best_score = score
                best_offset = offset

        offset = best_offset
        return offset

    def get_data(self):
        if self.flag == 'train':
            self.standardize_data = True
        else:
            self.standardize_data = False

        ts_offset = self._calculate_timestamps_offset()
        print(f"Timestamp offset: {ts_offset}")
        self.file_path = self.raw_data_path
        data = self.load_continous_data(standardize_data=True)

        self.file_path = self.sorted_data_path
        labels = np.zeros(data.shape[0])

        for spike in self.load_spike_timestamps():
            spike = int(spike * 40000) + int(ts_offset)
            start = max(0, spike-self.label_range[0])
            end = min(data.shape[0], spike+self.label_range[1])
            labels[start:end] = 1

        unit_spike_trains = self.get_unit_spike_trains()
        return data, labels, unit_spike_trains
    
    def get_waveforms_and_labels(self):
        self.file_path = self.sorted_data_path
        data, labels = self.load_waveform_data()
        channel_name = 'SPKC' + str(self.tetrode_idx*4+1).zfill(2)
        labels = labels.flatten()
        current_tetrode_idx = [i for i, label in enumerate(labels) if label[0:6] == channel_name]
        unit_id = np.unique(labels[current_tetrode_idx]).reshape(-1, 4)

        total_units_wf = []
        for i in range(len(unit_id)):
            unit_wf = []
            for j in range(4):
                unit_wf.append(data[np.where(labels == unit_id[i,j])])
            total_units_wf.append(unit_wf)

        sample_units_wf = []
        for i in range(len(unit_id)):
            sample_unit_wf = []
            for j in range(4):
                sample_unit_wf.append(total_units_wf[i][j][:500])
            sample_units_wf.append(sample_unit_wf)
        sample_units_wf = np.concatenate(sample_units_wf, axis=1)
        data = sample_units_wf.transpose(1, 2, 0) # (n, 64, 4)
        data = data[:, 2:-2, :] # (n, 60, 4)

        sample_units_labels = []
        for i in range(len(unit_id)):
            sample_units_labels.append(np.array([unit_id[i][i]]*500))
        labels = np.concatenate(sample_units_labels) # (n,)
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        #_, int_list = np.unique(labels, return_inverse=True)
        return data, labels

class NexRecordingMultiLoader:
    def __init__(self, 
                 raw_data_paths: list=None, 
                 sorted_data_paths: list=None,
                 **kwargs):
        self.sorted_data_paths = sorted_data_paths if sorted_data_paths is not None else kwargs.get('nex_sorted_paths', None)
        self.raw_data_paths = raw_data_paths if raw_data_paths is not None else [path.replace('-sorted.nex', '.nex') for path in self.sorted_data_paths]
        self.loaders = []

        for tetrode_idx in range(4):
            self.loaders.extend([NexRecordingLoader(raw_data_path=raw_data_path, 
                                                    sorted_data_path=sorted_data_path, 
                                                    tetrode_idx=tetrode_idx) 
                                 for raw_data_path, sorted_data_path in zip(self.raw_data_paths, self.sorted_data_paths)])

    def get_data(self, max_segments:int=500000):
        lfp_list = []
        labels_list = []
        unit_spike_trains_list = []
        for loader in self.loaders:
            data, labels, unit_spike_trains = loader.get_data()
            self.sampling_frequency = loader.sampling_frequency
            data = data[:max_segments]
            labels = labels[:max_segments]
            unit_spike_trains = unit_spike_trains[np.argwhere(unit_spike_trains[:, 0] < max_segments).flatten()]
            #print(data.shape, labels.shape, unit_spike_trains.shape)
            lfp_list.append(data.tolist())
            labels_list.append(labels.tolist())
            unit_spike_trains_list.append(unit_spike_trains.tolist())

        data = np.array(lfp_list, dtype=object)
        labels = np.array(labels_list, dtype=object)
        unit_spike_trains = np.array(unit_spike_trains_list, dtype=object)
        return data, labels, unit_spike_trains
    
    def get_waveforms_and_labels(self):
        waveform_list = []
        label_list = []
        for loader in self.loaders:
            data, labels = loader.get_waveforms_and_labels()
            waveform_list.append(data)
            label_list.append(labels)

        data = np.array(waveform_list)
        labels = np.array(label_list)
        return data, labels

class WFDataset(Dataset):
    """
    Dataset for waveform data.

    Args:
        data (np.ndarray): Waveform data, shape (n, nt, c).
        labels (np.ndarray): Labels for the waveform data, shape (n,).
        data_path (str): Path to the waveform data.
        transforms (list): List of transforms to apply to the data.
        transform_prob (float): Probability of applying the transforms.
    """
    def __init__(self, 
                 data:np.ndarray=None,
                 labels:np.ndarray=None,
                 data_path:str=None,
                 transforms:list=None,
                 transform_prob:float=0.5,
                 return_labels:bool=False,
                 preprocess:bool=True):

        self.data = data
        self.labels = labels
        self.transforms = transforms
        self.transform_prob = transform_prob
        self.num_samples = None
        self.return_labels = return_labels

        if data is None and data_path is None:
            raise ValueError("Either data or data_path must be provided.")
        
        if data is None and data_path is not None:
            self.data = np.load(data_path / "spikes_train.npy")
            self.labels = np.load(data_path / "labels_train.npy")

        if preprocess:
            self.preprocess()
        else:
            self.data = self.data.astype(np.float64).transpose(0, 2, 1).astype(np.float32) 
            print(f"Data shape: {self.data.shape}")
            self.labels = self.labels
            self.num_samples = self.data.shape[0]

        # Precompute the indices of each class for fast triplet sampling
        self.class_indices = {}
        for idx, label in enumerate(self.labels):
            if label.item() not in self.class_indices:
                self.class_indices[label.item()] = []
            self.class_indices[label.item()].append(idx)
    
    def preprocess(self):
        #train_set = self.data.astype(np.float64) * 1000 # convert to uV
        train_set = self.data.astype(np.float64)
        # apply lowpass filter
        fs = 10000
        cutoff = 1000
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(1, normal_cutoff, btype='low', analog=False)
        for i in range(train_set.shape[0]):
            train_set[i] = lfilter(b, a, train_set[i], axis=1)
        print('successfully smoothed data')

        self.data = train_set.transpose(0, 2, 1).astype(np.float32) 
        print(f"Data shape: {self.data.shape}")
        self.labels = self.labels
        self.num_samples = self.data.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the anchor sample.
        
        Returns:
            tuple: (anchor, positive, negative)
                - anchor (np.array): The anchor sample.
                - positive (np.array): A positive sample from the same class as the anchor.
                - negative (np.array): A negative sample from a different class as the anchor.
        """
        anchor = self.data[idx]
        #anchor_label = self.labels[idx].item()
        anchor_label = self.labels[idx].item() if np.isscalar(self.labels[idx]) else self.labels[idx][0]
        
        # Select a positive sample
        positive_idx = idx
        while positive_idx == idx:
            positive_idx = random.choice(self.class_indices[anchor_label])
        positive = self.data[positive_idx]

        # Select a negative sample
        negative_label = anchor_label
        while negative_label == anchor_label:
            negative_label = random.choice(list(self.class_indices.keys()))
        negative_idx = random.choice(self.class_indices[negative_label])
        negative = self.data[negative_idx]

        if self.transforms:
            if random.random() < self.transform_prob:
                anchor = self.transforms(deepcopy(anchor))
            if random.random() < self.transform_prob:
                positive = self.transforms(deepcopy(positive))
            if random.random() < self.transform_prob:
                negative = self.transforms(deepcopy(negative))
        if self.return_labels:
            return anchor, positive, negative, anchor_label
        return anchor, positive, negative

class WFDataset_test(Dataset):
    """
    Test dataset for waveform data.

    Args:
        data (np.ndarray): Waveform data, shape (n, nt, c).
        data_path (str): Path to the waveform data.
    """
    def __init__(self, 
                 data:np.ndarray=None,
                 data_path:str=None):

        if data is None and data_path is None:
            raise ValueError("Either data or data_path must be provided.")
        
        if data is None:
            data = np.load(data_path / "spikes_test.npy")

        self.data = data.astype(np.float32)
        self.samples = data.shape[0]
    
    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        return self.data[idx]

class LFPDataset(Dataset):
    """
    Dataset for LFP data.

    Args:
        lfp_input (np.ndarray): LFP data, shape (t, c).
        labels (np.ndarray): Labels for the LFP data, shape (t,).
        input_point_len (int): Length of the input data.
        window_size (int): Sliding window size.
        transforms (list): List of transforms to apply to the data.
        transform_prob (float): Probability of applying the transforms.
        random_access (bool): Whether to access data randomly or sequentially.
    """
    def __init__(self, 
                 lfp_input:np.ndarray, 
                 labels:np.ndarray, 
                 input_point_len:int=2000, 
                 window_size:int=None,
                 transforms:list=None,
                 transform_prob:float=0.5,
                 random_access:bool=False
                 ): 
        
        self.lfp_input = lfp_input.astype(np.float32)  # t * c
        self.labels = labels  # t
        self.input_point_len = input_point_len if input_point_len else lfp_input.shape[0]
        self.window_size = window_size if window_size else input_point_len
        self.transforms = transforms
        self.transform_prob = transform_prob
        self.random_access = random_access

    def __len__(self):
        return (self.labels.shape[0] - self.input_point_len) // self.window_size + 1

    def __getitem__(self, idx):
        if self.random_access:
            max_start_idx = self.labels.shape[0] - self.input_point_len
            start_idx = random.randint(0, max_start_idx)
        else:
            start_idx = idx * self.window_size
        
        lfp_input = self.lfp_input[start_idx: start_idx + self.input_point_len]  # t * c
        labels = self.labels[start_idx: start_idx + self.input_point_len].astype(np.int32)  # t
        
        if self.transforms:
            if random.random() < self.transform_prob:
                lfp_input = self.transforms(deepcopy(lfp_input))
        
        return lfp_input, labels

class LFPDataset_test(Dataset):
    """
    Test dataset for LFP data.

    Args:
        lfp_input (np.ndarray): LFP data, shape (t, c).
        input_point_len (int): Length of the input data.
        window_size (int): Sliding window size.
    """
    def __init__(self, 
                 lfp_input:np.ndarray, 
                 input_point_len:int=2000, 
                 window_size:int=None): # sliding window size
        
        self.lfp_input = lfp_input.astype(np.float32) # t * c
        self.input_point_len = input_point_len if input_point_len else lfp_input.shape
        self.window_size = window_size if window_size else input_point_len

    def __len__(self):
        return (self.lfp_input.shape[0] - self.input_point_len) // self.window_size + 1

    def __getitem__(self, idx):
        lfp_input = self.lfp_input[idx * self.window_size: (idx * self.window_size + self.input_point_len)] # t * c
        return lfp_input

from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import random
from collections import defaultdict

class ClassBatchSampler(Sampler):
    def __init__(self, dataset, num_classes_per_batch=4, batch_size=32):
        self.dataset = dataset
        self.num_classes_per_batch = num_classes_per_batch
        self.batch_size = batch_size
        self.class_indices = defaultdict(list)
        
        for idx, label in enumerate(dataset.labels):
            self.class_indices[label].append(idx)
        
        self.labels = list(self.class_indices.keys())
        self.num_samples_per_class = self.batch_size // self.num_classes_per_batch
        self.num_batches = len(dataset) // self.batch_size

    def __iter__(self):
        batch = []
        batch_count = 0
        while batch_count < self.num_batches:
            selected_labels = random.sample(self.labels, self.num_classes_per_batch)
            for label in selected_labels:
                indices = random.sample(self.class_indices[label], self.num_samples_per_class)
                batch.extend(indices)
                
                if len(batch) >= self.batch_size:
                    yield batch[:self.batch_size]
                    batch = []
                    batch_count += 1
                    if batch_count >= self.num_batches:
                        break
    def __len__(self):
            return self.num_batches





    