import os 
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spikeinterface as si
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw
import spikeinterface.sorters as ss
from spikeinterface.extractors import BinaryRecordingExtractor
from spikeinterface.core import NumpySorting, NumpyRecording
from data_factory.data_provider import LOODatasetManager, DataProvider

class SiCompareExperiment:
    def __init__(self,
                 root_path='./',
                 args=None,
                 data_info=None,
                 test_dataset_type=None,
                 sorting_method='kilosort2',
                 sorting_params=None,
                 folder='si_sorting_output', 
                 ):
        
        self.root_path = root_path
        self.args = args
        self.data_info = data_info
        self.test_dataset_type = test_dataset_type if test_dataset_type is not None else args.test_dataset_type
        self.sorting_method = sorting_method if sorting_method is not None else args.si_sorter
        self.folder = folder
        if os.path.exists(self.folder):
            shutil.rmtree(self.folder)

        if sorting_params is not None:
            self.sorter_params = sorting_params

        elif self.sorting_method == 'mountainsort4':
            self.sorter_params = {
                "freq_min": 300,
                "freq_max": 3000,
                "detect_threshold": 3,
                "filter": False if self.test_dataset_type == 'nex' else True,
            }
        
        elif self.sorting_method == 'mountainsort5':
            self.sorter_params = {
                "freq_min": 300,
                "freq_max": 3000,
                "detect_threshold": 3,
                "filter": False if self.test_dataset_type == 'nex' else True,
            }

        elif self.sorting_method == 'kilosort2':
            self.sorter_params = {
                "detect_threshold": 3,
            }
        
        elif self.sorting_method == 'kilosort4':
            self.sorter_params = {
                "nearest_chans": 4,
            }

        elif self.sorting_method == 'kilosort':
            self.sorter_params = {
                "freq_min": 300,
                "freq_max": 3000,
                "detect_threshold": 3,
            }
        elif self.sorting_method == 'herdingspikes':
            self.sorter_params = {
                #"filter": False if self.test_dataset_type == 'nex' else True,
            }
        elif self.sorting_method == 'ironclust':
            self.sorter_params = {
                #"filter": False if self.test_dataset_type == 'nex' else True,
            }

    def get_dataset(self):
        loader = DataProvider(args=self.args, data_info=self.data_info, flag='test')
        self.test_data, self.units_spike_train = loader.get_data(dataset_type=self.test_dataset_type)
        self.sampling_frequency = loader.sampling_frequency

    def load_data_to_si(self, data, gt_spike_times, gt_spike_labels):
        if self.sorting_method[:8] == 'kilosort':
            data.astype('int16').tofile('data.bin')
        else:
            data.astype('float32').tofile('data.bin')
        recording = BinaryRecordingExtractor(
            file_paths='data.bin',
            sampling_frequency=self.sampling_frequency,
            num_channels=4,
            dtype='int16' if self.sorting_method[:8] == 'kilosort' else 'float32'
        )
        recording.set_channel_locations([[0,0], [0,6.5], [-6.5,0], [-6.5,6.5]])
        gt_sorting = NumpySorting.from_times_labels(
            times_list=[gt_spike_times],
            labels_list=[gt_spike_labels.astype(int)],
            sampling_frequency=self.sampling_frequency
        )
        return recording, gt_sorting

    def run_sorting(self, recording):
        docker_images = {
            'mountainsort4': "spikeinterface/mountainsort4-fixed:latest",
            'mountainsort5': "spikeinterface/mountainsort5-base:latest",
            'kilosort': "spikeinterface/kilosort-compiled-base:latest",
            'kilosort2': "spikeinterface/kilosort2-compiled-base:latest",
            'kilosort4': "spikeinterface/kilosort4-base:latest",
            'herdingspikes': "spikeinterface/herdingspikes-base:latest", 
            'ironclust': "spikeinterface/ironclust-compiled-base:latest",
        }
        
        if self.sorting_method in docker_images:
            sorting = ss.run_sorter(
                sorter_name=self.sorting_method,
                recording=recording,
                docker_image=docker_images[self.sorting_method],
                remove_existing_folder=True,
                folder=self.folder,
                **self.sorter_params
            )
        else:
            raise ValueError(f"Unsupported sorting method: {self.sorting_method}")
        return sorting
    
    def show_sorting(self, recording, sorting):
        from spikeinterface.extractors import waveclustextractors
        unit_ids = sorting.unit_ids
        print(f'{len(unit_ids)} units found')
        outputs = sorting.count_num_spikes_per_unit()
        print(outputs)

    def evaluate_sorting(self, sorting, gt_sorting):
        compare = sc.compare_sorter_to_ground_truth(gt_sorting, sorting)
        metric = compare.get_performance()
        accuracy = metric['accuracy'].mean()
        recall = metric['recall'].mean()
        precision = metric['precision'].mean()
        print(metric)
        return accuracy, recall, precision

    def test(self, show_sorting=True):
        print(f'Running sorter: {self.sorting_method}')
        print(f'Default sorter params: {ss.get_default_sorter_params(sorter_name_or_class=self.sorting_method)}')
        print(f'Changed sorter params: {self.sorter_params}')
        self.get_dataset()
        if self.test_dataset_type in ['bbp', 'bbp_multi'] and len(self.test_data.shape) == 3:
            csv_path = os.path.join(self.root_path, f'{self.test_dataset_type}_{self.sorting_method}_sorting_results.csv')
            if os.path.exists(csv_path):
                os.remove(csv_path)

            for t in range(len(self.test_data)):
                if t < 20:
                    test_data = np.array([self.test_data[t]]).squeeze().astype(np.float64)
                    test_data = np.nan_to_num(test_data, nan=0.0, posinf=0.0, neginf=0.0)
                    units_spike_train = np.array([self.units_spike_train[t]]).squeeze().astype(np.float64)
                    recording, gt_sorting = self.load_data_to_si(test_data, units_spike_train[:, 0], units_spike_train[:, 1])
                    sorting = self.run_sorting(recording)
                    accuracy, recall, precision = self.evaluate_sorting(sorting, gt_sorting)

                    results = pd.DataFrame(columns=['accuracy', 'recall', 'precision'])
                    results['accuracy'] = [accuracy]
                    results['recall'] = [recall]
                    results['precision'] = [precision]
                else:
                    continue
                if os.path.exists(csv_path):
                    results.to_csv(csv_path, mode='a', header=False, index=False)
                else:
                    results.to_csv(csv_path, index=False)

        else:
            recording, gt_sorting = self.load_data_to_si(self.test_data, self.units_spike_train[:, 0], self.units_spike_train[:, 1])
            #print(np.mean(self.test_data), np.std(self.test_data))
            sorting = self.run_sorting(recording)
            if show_sorting:
                self.show_sorting(recording, sorting)
            accuracy, recall, precision = self.evaluate_sorting(sorting, gt_sorting)

            csv_path = os.path.join(self.root_path, f'{self.test_dataset_type}_{self.sorting_method}_sorting_results.csv')
            results = pd.DataFrame(columns=['accuracy', 'recall', 'precision'])
            results['accuracy'] = [accuracy]
            results['recall'] = [recall]
            results['precision'] = [precision]
            if os.path.exists(csv_path):
                    results.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                results.to_csv(csv_path, index=False)
            print(f'Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}')
