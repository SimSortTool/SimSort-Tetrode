from nex5file.reader import Reader
from nex5file.filedata import FileData
import numpy as np
from pathlib import Path
import os

class DataProcessor:
    def __init__(self, filename):
        self.filename = filename
        self.file_data = FileData(tsFrequency=40000, comment='plexon data')
        self.reader = Reader()
        self.data = None
        self.labels = None
        self.dataset = None
        self.dataset_no_label = None
        self.cont_names = None
        self.neuron_names = None
        self.data_cont = None

    def read_data(self, read_spkc=False):
        if read_spkc:
            self.data = self.reader.ReadNexHeadersOnly(self.filename)
            self.cont_names = self.data.ContinuousNames()
            self.data_cont = self.reader.ReadNexFileVariables(self.filename, self.cont_names)
        else:
            self.data = self.reader.ReadNexFile(self.filename)
            self.neuron_names = self.data.NeuronNames()
            Wavename = self.data.WaveNames()
            lens1 = self.data[Wavename[0]].NumPointsInWave()
            self.labels = np.empty((1,1))
            self.dataset = np.empty((0, lens1 + 1))
            self.dataset_no_label = np.empty((0, lens1))

    def process_waves(self):
        n = 0
        for name in self.data.WaveNames():
            values = self.data[name].WaveformValues()
            
            string_array = np.full((values.shape[0], 1), name)
            labels_column = np.full((values.shape[0], 1), n)
            self.labels = np.vstack((self.labels, string_array))
            collected_data_array = np.hstack((values, labels_column))
            self.dataset = np.vstack((self.dataset, collected_data_array))
            self.dataset_no_label = np.vstack((self.dataset_no_label, values))
            n += 1
            
        self.labels = np.delete(self.labels, np.s_[0:1], axis=0) # 删除初始化时创建的空行
    
    def get_event_timestamps(self, event_name, print_event_names=False):
        Event = self.data.EventNames()
        if print_event_names:
            print(Event)
        if event_name not in Event:
            print(f'event_name must be ont of {Event}')
        timestamps = self.data[event_name].Timestamps()
        return timestamps
    
    def get_neuron_timestamps(self, neuron_name, print_neuron_names=False):
        start_time = self.get_event_timestamps(event_name='Start')
        if print_neuron_names:
            print(self.neuron_names)
        if neuron_name not in self.neuron_names:
            print(f'neuron_name must be ont of {self.neuron_names}')
        timestamps = self.data[neuron_name].Timestamps() - start_time
        return timestamps
    
    def get_spkc(self, spkc_name, print_spkc_names=False):
        if print_spkc_names:
            print(self.cont_names)
        spkc = self.data_cont[spkc_name].ContinuousValues()
        return spkc

    def get_data_with_labels(self):
        return self.dataset

    def get_data_without_labels(self):
        return self.dataset_no_label

    def get_labels(self):
        return self.labels

if __name__ == '__main__':
    
    base_dir = Path.home() / 'Spikesort_Dev'
    filename = base_dir / 'sample_data_zym/20220802_IS_chr2_1-0.nex'
    data_processor = DataProcessor(filename)
    data_processor.read_data()
    data_processor.process_waves()

    dataset_with_labels = data_processor.get_data_with_labels()
    dataset_without_labels = data_processor.get_data_without_labels()
    labels = data_processor.get_labels()
    print(dataset_without_labels.shape)
