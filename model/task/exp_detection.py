import os
import yaml
import numpy as np
import argparse
from torch.utils.data import random_split
from imblearn.over_sampling import RandomOverSampler

from model import SpikeDetector
from data_factory.data_aug import (
    RandomChannelAugmentation,
    AddWithNoise,
    RandomAmplitudeScaling,
    RandomTimeJitter,
)
from data_factory.data_provider import LOODatasetManager, DataProvider
from data_factory.data_loader import LFPDataset

class DetectionExperiment:
    def __init__(
        self,
        root_path: str = './',
        args: argparse.Namespace = None,
        data_info: dict = None,
        train_dataset_type: str = None,
        test_dataset_type: str = None,
        loo_data_info: dict = None,
        detect_method: str = None,
        chunk_size: int = None,
        dmodel_save_path: str = None,
    ):

        self.root_path = root_path
        self.args = args
        self.data_info = data_info
        self.train_dataset_type = (
            train_dataset_type
            if train_dataset_type is not None
            else args.train_dataset_type
        )
        self.test_dataset_type = (
            test_dataset_type
            if test_dataset_type is not None
            else args.test_dataset_type
        )
        self.detect_method = (
            detect_method if detect_method is not None else args.detect_method
        )
        self.chunk_size = (
            chunk_size if chunk_size is not None else args.detect_chunk_size
        )
        self.save_folder = (
            dmodel_save_path if dmodel_save_path is not None else args.dmodel_save_path
        )
        os.makedirs(self.save_folder, exist_ok=True)

        if args.loo_exp == 1:
            self.train_dataset_type = None
            self.test_dataset_type = None

        self.loo_data_info = loo_data_info
        self.dmodel_arch = args.dmodel_arch
        self.dmodel_args = {
            'input_size': args.input_size,
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'output_size': args.output_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'pos_weight': args.pos_weight,
            'dropout': args.dropout,
            'nhead': args.nhead,
        }

        scale_range = [float(x) for x in args.amplitude_scale_range.split(',')]
        probabilities = [float(p) for p in args.probabilities.split(',')]

        self.transform = RandomChannelAugmentation(
            transforms=[
                RandomAmplitudeScaling(scale_range=scale_range),
                AddWithNoise(noise_level=args.noise_level),
                RandomTimeJitter(shift_range=args.time_jitter_range),
            ],
            max_channels=4,
            probabilities=probabilities,
        )

        self.test_data = None

    def _preprocess_train_data(self, train_data, train_labels):
        if len(train_data.shape) == 3:
            train_data = train_data.reshape(-1, 4)
            train_labels = train_labels.reshape(-1)
        pos_weight = np.sum(train_labels == 0) / np.sum(train_labels == 1)
        self.dmodel_args['pos_weight'] = pos_weight

        train_dataset = LFPDataset(
            train_data,
            train_labels,
            input_point_len=self.args.lfp_input_points,
            transforms=self.transform,
            transform_prob=self.args.transform_prob,
            random_access=True,
        )

        total_length = len(train_dataset)
        train_length = int(total_length * 0.8)
        val_length = total_length - train_length

        train_dataset, val_dataset = random_split(
            train_dataset, [train_length, val_length]
        )
        val_dataset = val_dataset
        return train_dataset, val_dataset

    def _load_loo_dataset(self, exclude_index=0):
        loader = LOODatasetManager(
            args=self.args, data_info=self.loo_data_info, max_segments=100000
        )
        train_data, train_labels, test_data, test_spike_trains = loader.get_dataset(
            exclude_index=exclude_index
        )
        self.test_dataset_type = loader.test_dataset_type
        self.train_dataset, self.val_dataset = self._preprocess_train_data(
            train_data, train_labels
        )
        self.test_data = test_data
        self.units_spike_train = test_spike_trains
        self.test_sampling_frequency = loader.sampling_frequency

    def _load_dataset(self):
        if self.args.only_test != 1:
            loader = DataProvider(
            args=self.args, data_info=self.data_info, max_segments=100000, flag='train'
            )
            data, labels = loader.get_data(dataset_type=self.train_dataset_type)
            self.train_dataset, self.val_dataset = self._preprocess_train_data(data, labels)
            print(f'train data {self.train_dataset_type} {data.shape}')
        else:
            self.train_dataset = None
            self.val_dataset = None

        loader = DataProvider(
            args=self.args, data_info=self.data_info, max_segments=100000, flag='test'
        )
        self.test_data, self.units_spike_train = loader.get_data(
            dataset_type=self.test_dataset_type
        )
        print(f'test data {self.test_dataset_type} {self.test_data.shape}')
        self.test_sampling_frequency = loader.sampling_frequency

    def get_dataset(self):
        if self.args.loo_exp == 1:
            self._load_loo_dataset(exclude_index=self.args.exclude_index)
        else:
            self._load_dataset()
        return (
            self.train_dataset,
            self.val_dataset,
            self.test_data,
            self.units_spike_train,
        )

    def train(self, save_config=True):
        self.get_dataset()
        if save_config:
            model_params = [
                'dmodel_arch',
                'input_size',
                'hidden_size',
                'num_layers',
                'output_size',
                'learning_rate',
                'weight_decay',
                'dropout',
                'nhead',
            ]
            config = {param: getattr(self.args, param) for param in model_params}
            config_path = os.path.join(self.root_path, 'detection_config.yaml')
            with open(config_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)

            aug_params = [
                'amplitude_scale_range',
                'noise_level',
                'time_jitter_range',
                'probabilities',
                'transform_prob',
                'model_threshold',
            ]
            aug_config = {param: getattr(self.args, param) for param in aug_params}
            aug_config_path = os.path.join(self.root_path, 'detection_aug_config.yaml')
            with open(aug_config_path, 'w') as file:
                yaml.dump(aug_config, file, default_flow_style=False)

        self.model = SpikeDetector(
            root_path=self.root_path,
            model_arch=self.dmodel_arch,
            model_args=self.dmodel_args,
            mode='train',
            gpu=0,
        )

        train_loader = self.model.load_data(
            dataset=self.train_dataset, batch_size=self.args.batch_size
        )
        val_loader = self.model.load_data(
            dataset=self.val_dataset, batch_size=self.args.batch_size
        )

        def tester():
            return self.test()

        self.model.train(
            train_loader=train_loader,
            tester=tester,
            val_loader=val_loader,
            num_epochs=self.args.num_epochs,
            predict_every=self.args.predict_cycle,
            resume=self.args.resume,
            seed=self.args.seed,
            save_model=True,
            save_path=self.save_folder,
        )

    def test(self, model_save_path=None, test_demo=False):
        if self.test_data is None:
            self.get_dataset()

        if model_save_path is not None:
            self.save_folder = model_save_path

        self.model = SpikeDetector(
            root_path=self.root_path,
            model_arch=self.dmodel_arch,
            model_args=self.dmodel_args,
            mode='test',
            gpu=0,
        )
        self.model.print_model()

        if self.args.loo_exp == 1 or self.test_dataset_type == 'bbp':
            for i in range(len(self.test_data)):
                if i < 20:
                    test_data = (
                        np.array([self.test_data[i]]).squeeze().astype(np.float64)
                    )
                    units_spike_train = (
                        np.array([self.units_spike_train[i]])
                        .squeeze()
                        .astype(np.float64)
                    )
                    self.model.predict(
                        data=test_data,
                        units_spike_train=units_spike_train,
                        dataset_type=self.test_dataset_type,
                        sampling_frequency=self.test_sampling_frequency,
                        detect_chunk_size=self.chunk_size,
                        detect_method=self.detect_method,
                        model_save_path=self.save_folder,
                        model_th=self.args.model_threshold,
                        threshold=self.args.threshold,
                        save_csv=False,
                        save_name=f'loo_{self.test_dataset_type}_data{i}',
                    )
        else:
            if test_demo:
                self.test_data = self.test_data[: self.chunk_size * 10, :]
                self.units_spike_train = self.units_spike_train[
                    self.units_spike_train[:, 0] < self.chunk_size * 10
                ]
            
            accuracy, recall, precision = self.model.predict(
                data=self.test_data,
                units_spike_train=self.units_spike_train,
                dataset_type=self.test_dataset_type,
                sampling_frequency=self.test_sampling_frequency,
                detect_chunk_size=self.chunk_size,
                detect_method=self.detect_method,
                model_save_path=self.save_folder,
                model_th=self.args.model_threshold,
                threshold=self.args.threshold,
                only_test=True if self.args.only_test == 1 else False,
                save_csv=True,
                save_name=f'{self.test_dataset_type}_data',
            )
            return accuracy, recall, precision
