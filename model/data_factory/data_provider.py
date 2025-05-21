import os
import numpy as np
from data_factory.data_loader import (NexRecordingLoader, HybridLoader, BbpRecordingLoader, NexRecordingMultiLoader, 
                                      HybridMultiLoader, CRCNSRecordingLoader, WaveClusLoader)

data_dict = {
    'bbp': BbpRecordingLoader,
    'allen': BbpRecordingLoader,
    'nex': NexRecordingLoader,
    'hybrid': HybridLoader,
    'crcns': CRCNSRecordingLoader,
    'waveclus': WaveClusLoader,
}

multi_data_dict = {
    'bbp_multi': BbpRecordingLoader,
    'allen_multi': BbpRecordingLoader,
    'nex_multi': NexRecordingMultiLoader,
    'hybrid_multi': HybridMultiLoader,
}

class DataProvider:
    """
    Data provider class for single dataset.

    Args:
        args (argparse.Namespace): Arguments.
        data_info (dict): Data information.
        flag (str): Flag for train or test.
        max_segments (int): Maximum length of recording.

    """
    def __init__(self, 
                 args=None, 
                 data_info:dict=None, 
                 flag:str = 'test',
                 max_segments:int =100000,):
        
        self.args = args
        self.data_info = data_info
        self.flag = flag
        self.max_segments = max_segments
        self.sampling_frequency = None
    
    def get_data(self,dataset_type=None):
        if dataset_type is not None:
            Data = data_dict[dataset_type]
            self.dataset_type = dataset_type
        else:
            Data = data_dict[self.args.dataset_type]
            self.dataset_type = self.args.dataset_type

        if self.args.task_name == 'detection':
            loader = Data(**self.data_info)
            data, labels, units_spike_train = loader.get_data()
            self.sampling_frequency = loader.sampling_frequency
            
            if self.flag == 'test':
                loader.flag = 'test'
                return data, units_spike_train
            elif self.flag == 'train':
                loader.flag = 'train'
                return data, labels
            
        elif self.args.task_name == 'extraction':
            loader = Data(**self.data_info)
            wf_data, wf_labels = loader.get_waveforms_and_labels()
            return wf_data, wf_labels
        
        elif self.args.task_name == 'sorting' or self.args.task_name == 'si':
            loader = Data(**self.data_info)
            loader.flag = 'test'
            if self.args.task_name == 'si' and self.dataset_type == 'bbp':
                loader.filter_data = False
                loader.spatial_whitening = False 
                loader.standardize_data = False
            elif self.args.task_name == 'si' and self.dataset_type == 'hybrid':
                loader.no_preprocess = True
            elif self.args.task_name == 'si' and self.dataset_type == 'waveclus':
                loader.spatial_whitening = False
                
            data, labels, units_spike_train = loader.get_data()
            self.sampling_frequency = loader.sampling_frequency
            return data, units_spike_train
    
    def get_multi_data(self, dataset_type):
        Data = multi_data_dict[dataset_type]
        loader = Data(**self.data_info)
        data, labels, units_spike_train = loader.get_data(max_segments=self.max_segments)
        self.sampling_frequency = loader.sampling_frequency
        self.dataset_type = dataset_type

        if self.args.task_name == 'detection':
            if self.flag == 'test':
                loader.flag = 'test'
                return data, units_spike_train
            elif self.flag == 'train':
                loader.flag = 'train'
                return data, labels

        elif self.args.task_name == 'extraction':
            wf_data, wf_labels = loader.get_waveforms_and_labels()
            return wf_data, wf_labels
        
        elif self.args.task_name == 'sorting' or self.args.task_name == 'si':
            loader.flag = 'test'
            return data, units_spike_train
        
class LOODatasetManager:
    """
    Leave-one-out experiment dataset manager.

    Args:
     loader (List): List of data loaders object (e.g. HybridLoader, BbpRecordingLoader, NexRecordingLoader).

    """
    def __init__(self,
                 args=None,
                 data_info: dict = None,
                 max_segments: int = 100000,):
        
        self.args = args
        self.data_info = data_info
        self.dataset_type = [key.split('_')[0]+'_multi' for key in data_info]
        self.dataset_type = [item for item in self.dataset_type if item in ['bbp_multi', 'nex_multi', 'hybrid_multi', 'allen_multi']]
        data_info_items = list(data_info.items())
        self.loaders = [
            DataProvider(args=self.args, data_info={data_info_items[i][0]: data_info_items[i][1]}, max_segments=max_segments) 
            for i in range(len(self.dataset_type)) if i < len(data_info_items)
        ]
        #self.loaders = [DataProvider(data_info=data_info, max_segments=max_segments) for _ in range(len(self.dataset_type))]
        self.sampling_frequency = None
        
    def get_dataset(self, exclude_index):
        if self.args.task_name == 'detection':
            return self._get_detection_dataset(exclude_index)
        elif self.args.task_name == 'extraction':
            return self._get_wf_dataset(exclude_index)
        elif self.args.task_name == 'sorting' or self.args.task_name == 'si':
            return self._get_detection_dataset(exclude_index)

    def _get_detection_dataset(self, exclude_index):
        train_data = []
        train_labels = []
        test_data, test_spike_trains = None, None
        
        for idx, loader in enumerate(self.loaders):
            if idx == exclude_index:
                loader.flag = 'test'
                print('test:', self.dataset_type[idx])
                lfp, unit_spike_trains = loader.get_multi_data(dataset_type=self.dataset_type[idx])
                print(lfp.shape, unit_spike_trains.shape)
                self.sampling_frequency = loader.sampling_frequency
                test_data = lfp
                test_spike_trains = unit_spike_trains
                self.test_dataset_type = self.dataset_type[idx]
            else:
                loader.flag = 'train'
                print('train:', self.dataset_type[idx])
                lfp, labels = loader.get_multi_data(dataset_type=self.dataset_type[idx])
                print(lfp.shape, labels.shape)
                train_data.append(lfp)
                train_labels.append(labels)

        train_data = np.vstack(train_data)
        train_labels = np.concatenate(train_labels, axis=0)

        return train_data, train_labels, test_data, test_spike_trains

    def _get_wf_dataset(self, exclude_index):
        train_data = []
        train_labels = []
        test_data, test_labels = None, None
        
        for idx, loader in enumerate(self.loaders):
            if idx == exclude_index:
                loader.flag = 'test'
                print('test:', self.dataset_type[idx])
                wf_data, wf_labels = loader.get_multi_data(dataset_type=self.dataset_type[idx])
                print(wf_data.shape, wf_labels.shape)
                test_data = wf_data
                test_labels = wf_labels
            else:
                loader.flag = 'train'
                print('train:', self.dataset_type[idx])
                wf_data, wf_labels = loader.get_multi_data(dataset_type=self.dataset_type[idx])
                print(wf_data.shape, wf_labels.shape)
                train_data.append(wf_data)
                train_labels.append(wf_labels)

        train_data = np.vstack(train_data)
        train_labels = np.concatenate(train_labels, axis=0)

        return train_data, train_labels, test_data, test_labels



