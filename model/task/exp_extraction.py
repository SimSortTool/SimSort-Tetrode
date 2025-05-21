import os
from tqdm import tqdm
import yaml
import numpy as np
import pandas as pd

import torch
from data_factory.data_provider import LOODatasetManager, DataProvider
from model import RepresentationExtractor

class ExtractionExperiment:
    def __init__(self,
                 root_path='./',
                 args=None,
                 data_info=None,
                 train_dataset_type=None,
                 test_dataset_type=None,
                 loo_data_info=None,
                 ):
        
        self.root_path = root_path
        self.args = args
        self.data_info = data_info
        self.train_dataset_type = train_dataset_type if train_dataset_type is not None else args.train_dataset_type
        self.test_dataset_type = test_dataset_type if test_dataset_type is not None else args.test_dataset_type
        self.loo_data_info = loo_data_info
        self.emodel_arch = args.emodel_arch

        if self.emodel_arch == 'gru' or self.emodel_arch == 'GRU':
            self.emodel_args = {
                'input_size': args.e_input_size,
                'hidden_size': args.e_hidden_size,
                'num_layers': args.e_num_layers, 
                'learning_rate': args.e_learning_rate,
                'weight_decay': 1e-5,
            }
            
        elif self.emodel_arch == 'transformer' or self.emodel_arch == 'Transformer':
            self.emodel_args = {
                'input_size': args.e_input_size,
                'hidden_size': args.e_hidden_size,
                'num_layers': args.e_num_layers, 
                'learning_rate': args.e_learning_rate,
                'nhead': args.e_nhead,
                'dropout': args.e_dropout,
                'weight_decay': 1e-5,
            }

        self.aug_args = {
            'shift_max': args.e_shift_max,
            'stretch_range': args.e_strech_range,
            'scale_range': args.e_scale_range,
            'noise_level': args.e_noise_level,
            'max_channels': args.e_max_channels,
            'prob': args.e_probabilities
        }

        #self.save_folder = os.path.join(self.root_path, 'saved_models')
        self.save_folder = args.emodel_save_path
        print(f'model save folder: {self.save_folder}')
        os.makedirs(self.save_folder, exist_ok=True)

        self.train_data, self.train_label = None, None
        self.test_data, self.test_label = None, None

    def _load_dataset(self):
        loader = DataProvider(args=self.args, data_info=self.data_info)
        if self.args.only_test != 1:
            self.train_data, self.train_label = loader.get_data(dataset_type=self.train_dataset_type)
            self.train_data = self.train_data.reshape(-1, 4, self.train_data.shape[-1])
            self.train_label = self.train_label.reshape(-1)
            print(f'train data shape: {self.train_data.shape}, train label shape: {self.train_label.shape}')
        else:
            self.train_data = None
            self.train_label = None
        self.test_data, self.test_label = loader.get_data(dataset_type=self.test_dataset_type)
        print(f'test data shape: {self.test_data.shape}, test label shape: {self.test_label.shape}')

    def _load_loo_dataset(self, exclude_idx=0):
        loader = LOODatasetManager(args=self.args, data_info=self.loo_data_info)
        self.train_data, self.train_label, self.test_data, self.test_label = loader.get_dataset(exclude_idx)
        print(f'train data shape: {self.train_data.shape}, train label shape: {self.train_label.shape}')
        print(f'test data shape: {self.test_data.shape}, test label shape: {self.test_label.shape}')

    def get_dataset(self):
        if self.args.loo_exp:
            self._load_loo_dataset()
        else:
            self._load_dataset()
        return self.train_data, self.train_label, self.test_data, self.test_label

    def train(self, save_config=True):
        self.get_dataset()

        if save_config:
            model_params = ['e_input_size', 'e_hidden_size', 'e_num_layers', 
                              'e_learning_rate', 'e_batch_size', 'e_num_epochs',
                              'e_dropout', 'e_nhead', 'e_rl', 'e_rl_batches', 'e_rl_weight']
            config = {param: getattr(self.args, param) for param in model_params}
            config_path = os.path.join(self.root_path, 'config.yaml')
            with open(config_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)

            aug_params = ['e_noise_level', 'e_scale_range', 'e_transform_prob', 'e_probabilities',
                          'e_max_channels', 'e_shift_max', 'e_strech_range']
            aug_config = {param: getattr(self.args, param) for param in aug_params}
            aug_config_path = os.path.join(self.root_path, 'aug_config.yaml')
            with open(aug_config_path, 'w') as file:
                yaml.dump(aug_config, file, default_flow_style=False)

        model = RepresentationExtractor(
                        root_path=os.getenv('AMLT_OUTPUT_DIR', './'),
                        model_arch=self.emodel_arch,
                        model_args=self.emodel_args,
                        gpu=0
                    )
        
        model.train(
            data=self.train_data,
            labels=self.train_label,
            test_data=self.test_data,
            test_labels=self.test_label,
            optimizer='adam',
            aug_args=self.aug_args,
            aug_prob=self.args.e_transform_prob,
            batch_size=self.args.e_batch_size,
            num_epochs=self.args.e_num_epochs,
            save_path=self.save_folder,
            resume=self.args.resume,
            seed=self.args.seed,
            predict=True,
            predict_every=self.args.predict_cycle,
            down_dim_method=self.args.down_dim_method,
            cal_rl_loss=True if self.args.e_rl == 1 else False, 
            rl_batches=self.args.e_rl_batches,
            rl_weight=self.args.e_rl_weight,
            cluster_method='GMM',
            batch_class=self.args.e_batch_class,
        )

    def test(self, model_save_path=None):
        if self.test_data is None:
            self.get_dataset()
        
        if model_save_path is not None:
            self.save_folder = model_save_path

        model = RepresentationExtractor(
                        root_path=os.getenv('AMLT_OUTPUT_DIR', './'),
                        model_arch=self.emodel_arch,
                        model_args=self.emodel_args,
                        mode='test',
                        gpu=0,
                    )
        
        model.load_model(checkpoint_path=self.save_folder)

        if self.args.loo_exp == 1:
            test_data = np.array([self.test_data[i]]).squeeze().astype(np.float64)
            test_label = self.test_label[i]

            for i in range(len(self.test_data)):
                model.predict(
                    data=test_data,
                    sorted_labels=test_label,
                    model_save_path=self.save_folder,
                    down_dim_method=self.args.down_dim_method,
                    plot=True,
                )
        else:
            ARI_list = []
            for i in tqdm(range(20), desc='Running 20 times for ARI'):
                ARI = model.predict(
                    data=self.test_data,
                    sorted_labels=self.test_label,
                    model_save_path=self.save_folder,
                    down_dim_method=self.args.down_dim_method,
                    use_model=True,
                    plot=False,
                )
                ARI_list.append(ARI)
            mean_ARI = np.mean(ARI_list)
            std_ARI = np.std(ARI_list)
            print(f'mean ARI: {mean_ARI}, std ARI: {std_ARI}')
            csv_path = os.path.join(self.root_path, 'Identification_ARI.csv')

            df = pd.DataFrame(columns=['mean_ARI', 'std_ARI'])
            df['mean_ARI'] = [mean_ARI]
            df['std_ARI'] = [std_ARI]
            df.to_csv(
                csv_path,
                mode='a' if os.path.exists(csv_path) else 'w',
                header=not os.path.exists(csv_path),
                index=False,
            )




            








            




