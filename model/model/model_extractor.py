import os
import time
import umap
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import BatchSampler

from model import GRU, Transformer
from data_factory.data_loader import WFDataset, WFDataset_test, ClassBatchSampler
from data_factory.data_aug import RandomChannelAugmentation, AddWithNoise, RandomAmplitudeScaling, RandomTimeJitter, RandomTimeStretch
from utils.tool import reconstruct


class RepresentationExtractor:
    def __init__(self, 
                 root_path: str = '/',
                 model_arch: str = 'gru', 
                 model_args: dict = None,
                 mode: str = 'train',
                 gpu: int = 0):
        
        self.root_path = root_path
        self.model_arch = model_arch
        self.model_args = model_args
        self.mode = mode
        self.aug_args = None
        self.optimizer = None

        if gpu is None or gpu < 0 or not torch.cuda.is_available():
            self.device = "cpu"
            print("Using device: cpu")
        else:
            self.device = f"cuda:{gpu}"
            print(f"Using device: cuda:{gpu}")
            
        if model_arch == 'gru' or model_arch == 'GRU':
            self.model = GRU(input_size=model_args.get('input_size', 4),
                             hidden_size=model_args.get('hidden_size', 512),
                             num_layers=model_args.get('num_layers', 1),
                             ).to(self.device)

        elif model_arch == 'transformer' or model_arch == 'Transformer':
            self.model = Transformer(input_size=model_args.get('input_size', 4),
                                     hidden_size=model_args.get('hidden_size', 128),
                                     num_layers=model_args.get('num_layers', 2),
                                     nhead=model_args.get('nhead', 4),
                                     dropout=model_args.get('dropout', 0.1),
                                     mode='feature_extraction',
                                     ).to(self.device)

    def load_model(self, checkpoint_path: str = 'saved_models'):
        checkpoint_full_path = os.path.join(checkpoint_path, 'checkpoint.pth')

        if not os.path.exists(checkpoint_full_path):
            print(f"Checkpoint not found at {checkpoint_full_path}.")
            return 0, []

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            checkpoint = torch.load(checkpoint_full_path, map_location=device, weights_only=True)

            if self.mode == 'train':
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                losses = checkpoint['losses']
                self.model.train()  
                print(f"Resuming from epoch {start_epoch}.")
            elif self.mode == 'test':
                self.model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = None
                losses = None
                self.model.eval()

        except Exception as e:
            print(f"Error loading model: {e}")
            return 0, []

        return start_epoch, losses

    def print_model(self):
        if self.model_arch == 'gru' or self.model_arch == 'GRU':
            print("Extraction Model Configuration:")
            print(f"{'-'*30}")
            print(f"{'Parameter':<20} | {'Value':<10}")
            print(f"{'-'*30}")
            print(f"{'Model Architecture':<20} | {self.model_arch}")
            print(f"{'Input Size':<20} | {self.model_args['input_size']}")
            print(f"{'Hidden Size':<20} | {self.model_args['hidden_size']}")
            print(f"{'Number of Layers':<20} | {self.model_args['num_layers']}")
            print(f"{'-'*30}")
        elif self.model_arch == 'transformer' or self.model_arch == 'Transformer':
            print("Extraction Model Configuration:")
            print(f"{'-'*30}")
            print(f"{'Parameter':<20} | {'Value':<10}")
            print(f"{'-'*30}")
            print(f"{'Model Architecture':<20} | {self.model_arch}")
            print(f"{'Input Size':<20} | {self.model_args['input_size']}")
            print(f"{'Hidden Size':<20} | {self.model_args['hidden_size']}")
            print(f"{'Number of Layers':<20} | {self.model_args['num_layers']}")
            print(f"{'Number of Heads':<20} | {self.model_args['nhead']}")
            print(f"{'Dropout':<20} | {self.model_args['dropout']}")
            print(f"{'-'*30}")
        
    def train(self,
              data: str = None,
              labels: str = None,
              data_path: str = None,
              test_data: str = None,
              test_data_path: str = None,
              test_labels: str = None,
              optimizer: str = 'adam', 
              aug_args: dict = {'shift_ws': 35,
                                'shift_max': 5,
                                'stretch_range': 0.5,
                                'scale_range': '0,1', # from 0 to 1
                                'noise_level': 4,
                                'max_channels': 4, # maximum number of channels to be transformed
                                'prob': '0,0,1,1', # transform probability for each augmentation
              },
                aug_prob: float = 0.5,
                batch_size: int = 1024,
                num_epochs: int = 30,
                save_path: str = None,
                resume: bool = False,
                seed: int = 42,
                predict : bool = True,
                down_dim_method: str = 'TSNE',
                predict_every: int = 20,
                cal_rl_loss: bool = False,
                rl_batches: int = 5,
                rl_weight: float = 0.1,
                cluster_method: str = 'GMM',
                batch_class: int = 10,
              ):
        
        scale_range = [float(x) for x in aug_args['scale_range'].split(',')]
        shift_range = np.random.randint(1, aug_args['shift_max'])
        prob = [float(x) for x in aug_args['prob'].split(',')]

        transform = RandomChannelAugmentation(
            transforms=[
                RandomTimeJitter(shift_range=shift_range),
                RandomTimeStretch(stretch_range=aug_args['stretch_range']),
                RandomAmplitudeScaling(scale_range=scale_range),
                AddWithNoise(noise_level=aug_args['noise_level']),
            ],
            max_channels=aug_args['max_channels'],
            probabilities=prob,
        )   
        train_dataset = WFDataset(data=data, 
                                  labels=labels, 
                                  data_path=data_path, 
                                  transforms=transform, 
                                  transform_prob=aug_prob,
                                  return_labels=True)
        
        
        if cal_rl_loss:
            print('Training model with reinforcement learning')
            batch_sampler = ClassBatchSampler(train_dataset, num_classes_per_batch=batch_class, batch_size=batch_size)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        print('successfully prepared dataset')

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        criterion = nn.TripletMarginLoss(margin=1.0, p=2)

        if optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.model_args['learning_rate'], 
                                        weight_decay=self.model_args['weight_decay'])
            
        scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)

        save_path = os.path.join(self.root_path, save_path)
        checkpoint_path = os.path.join(save_path, 'checkpoint.pth')

        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        if resume and os.path.exists(checkpoint_path):
            start_epoch, losses = self.load_model(checkpoint_path=save_path)
        else:
            start_epoch = 0
            losses = []

        self.print_model()

        for epoch in range(start_epoch, num_epochs):
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            self.model.train()

            running_loss = 0.0
            batch_idx = 0
            rl_indices = random.sample(range(len(train_loader)), rl_batches)
            
            for anchor, positive, negative, anchor_label in train_loader:

                cal_this_batch = batch_idx in rl_indices
                rl_loss = 0.0
                anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)

                anchor = reconstruct(anchor, k=8)
                positive = reconstruct(positive, k=8)
                negative = reconstruct(negative, k=8)

                self.optimizer.zero_grad()

                anchor_output = self.model(anchor)
                positive_output = self.model(positive)
                negative_output = self.model(negative)

                triplet_loss = criterion(anchor_output, positive_output, negative_output)
                mix_loss = triplet_loss

                if cal_rl_loss and cal_this_batch:
                    distribution = torch.distributions.Normal(anchor_output, 0.25 * torch.ones_like(anchor_output))
                    samples = distribution.sample()
                    reward = self._calculate_reward(samples.cpu().numpy(), anchor_label, cluster_method=cluster_method, down_dim_method=down_dim_method)
                    #print(f"Reward: {reward}")
                    reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
                    log_prob = distribution.log_prob(samples)
                    rl_loss = rl_weight * (-reward.detach() * log_prob).mean()
                    mix_loss += rl_loss

                mix_loss.backward() 
                self.optimizer.step() 

                running_loss += triplet_loss.item()
                batch_idx += 1

            scheduler.step()
            losses.append(running_loss / batch_idx)

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")
            
            if predict:
                if (epoch+1) % predict_every == 0 and epoch > 0:

                    ari_score = self.predict(data_path=test_data_path,
                                             data=test_data,
                                             sorted_labels=test_labels,
                                             plot=True, 
                                             down_dim_method=down_dim_method,
                                             model_save_path=save_path)
                    
                    print(f"Test ARI after Epoch {epoch + 1}: {ari_score * 100:.2f}%")

            # save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'losses': losses
            }, checkpoint_path)

        print("Training Finished!")

    def _calculate_reward(self, 
                          features: np.ndarray, 
                          true_labels: np.ndarray, 
                          cluster_method: str = 'GMM', 
                          down_dim_method: str = 'TSNE', 
                          ):

        if down_dim_method == 'PCA':
            pca = PCA(n_components=5)
            features = pca.fit_transform(features)
            reduced_features = features

        elif down_dim_method == 'TSNE2':
            tsne = TSNE(n_components=2)
            reduced_features = tsne.fit_transform(features)
        
        elif down_dim_method == 'TSNE3':
            tsne = TSNE(n_components=3)
            reduced_features = tsne.fit_transform(features)
        
        elif down_dim_method == 'UMAP':
            umap_model = umap.UMAP(n_components=2)
            reduced_features = umap_model.fit_transform(features)
        
        elif down_dim_method == 'None':
            reduced_features = features

        if cluster_method == 'DBSCAN':
            scaler = StandardScaler()
            reduced_features = scaler.fit_transform(reduced_features)
            labels = DBSCAN(eps=0.15, min_samples=20).fit_predict(reduced_features)

        elif cluster_method == 'GMM':
            gmm = GaussianMixture(n_components=5, random_state=42)
            gmm.fit(reduced_features)
            labels = gmm.predict(reduced_features)

        ARI = adjusted_rand_score(labels, true_labels)
        return ARI
    
    def test(self, 
             data: str = None,
             data_path: str = None, 
             batch_size: int = 256,
             saved_model_path: str = 'saved_models'):
    
        if saved_model_path is not None:
            saved_model_path = os.path.join(self.root_path, saved_model_path)
            self.load_model(checkpoint_path=saved_model_path)

        test_dataset = WFDataset_test(data=data, data_path=data_path)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        sample_num = len(test_loader.dataset)
        output_size = self.model.hidden_size

        features = np.zeros((sample_num, output_size), dtype=np.float32)

        start_idx = 0

        with torch.no_grad():
            for batch_idx, anchor in enumerate(test_loader):

                anchor = anchor.to(self.device, non_blocking=True)
                anchor = reconstruct(anchor, k=8)
                anchor_output = self.model(anchor).cpu().numpy()

                batch_size = anchor.size(0)
                features[start_idx:start_idx + batch_size] = anchor_output

                start_idx += batch_size
        return features

    def predict(self, 
                data: str = None,
                sorted_labels: str = None,
                data_path: str = None, 
                num_cluster: int = None, 
                down_dim_method: str = 'PCA',
                use_model: bool = True,
                model_save_path: str = 'saved_models',
                plot=False
                ):
        
        if use_model:
            features = self.test(data=data, data_path=data_path, saved_model_path=model_save_path)
        else:
            #print(f'use {down_dim_method} for feature extraction, no model is used')
            features = data.transpose(0, 2, 1).reshape(data.shape[0], -1)

        if sorted_labels is not None:
            sorted_label = sorted_labels
        else:
            sorted_label = np.load(data_path / "labels_test.npy")

        if num_cluster is None:
            n_components = np.unique(sorted_label).shape[0]
        else:
            n_components = num_cluster

        if down_dim_method == 'PCA':
            pca = PCA(n_components=5)
            reduced_features = pca.fit_transform(features)
        elif down_dim_method == 'TSNE2':
            tsne = TSNE(n_components=2)
            reduced_features = tsne.fit_transform(features)
        elif down_dim_method == 'TSNE3':
            tsne = TSNE(n_components=3)
            reduced_features = tsne.fit_transform(features)
        elif down_dim_method == 'UMAP':
            umap_model = umap.UMAP(n_components=2)
            reduced_features = umap_model.fit_transform(features)
        elif down_dim_method == 'None':
            reduced_features = reduced_features

        # scaler = StandardScaler()
        # reduced_features = scaler.fit_transform(reduced_features)

        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(reduced_features)
        labels = gmm.predict(reduced_features)
        
        ARI = adjusted_rand_score(labels, sorted_label)

        if plot:
            if down_dim_method == 'None':
                reduced_features = TSNE(n_components=2, random_state=42).fit_transform(features)
            else:
                reduced_features = reduced_features

            fig, axes = plt.subplots(2, 1, figsize=(5, 10), constrained_layout=True)

            ax = axes[0]
            centers = np.array([reduced_features[labels == i].mean(axis=0) for i in range(n_components)])
            num_label = pd.factorize(labels)[0]
            ax.scatter(reduced_features[:, 0], reduced_features[:, 1], c=num_label, cmap='tab20', s=15, alpha=0.4)

            for i, center in enumerate(centers):
                ax.scatter(center[0], center[1], s=10, marker='o', color='black')
                if down_dim_method == 'PCA':
                    ax.text(center[0] + np.mean(np.abs(centers) / 2), center[1], f'{i + 1}', color='black', fontsize=16, ha='center', va='center')
                else:
                    ax.text(center[0] + np.mean(np.abs(centers) / 5), center[1], f'{i + 1}', color='black', fontsize=16, ha='center', va='center')

            # if down_dim_method == 'PCA':
            #     ax.set_xlabel('PC 1', fontsize=15)
            #     ax.set_ylabel('PC 2', fontsize=15)
            # elif down_dim_method in ['TSNE2', 'TSNE3', 'None']:
            #     ax.set_xlabel('TSNE 1', fontsize=15)
            #     ax.set_ylabel('TSNE 2', fontsize=15)
            # elif down_dim_method == 'UMAP':
            #     ax.set_xlabel('UMAP 1', fontsize=15)
            #     ax.set_ylabel('UMAP 2', fontsize=15)

            if use_model:
                ax.set_title(f'SimSort embedding ({down_dim_method[:4]}), \nGMM Pred Labels {ARI:.2f}', fontsize=18, ha='center')
            else:
                ax.set_title(f'Raw feature ({down_dim_method[:4]}), \nGMM Pred Labels {ARI:.2f}', fontsize=18, ha='center')

            ax = axes[1]
            num_label = pd.factorize(sorted_label)[0]
            centers = np.array([reduced_features[num_label == i].mean(axis=0) for i in range(n_components)])
            ax.scatter(reduced_features[:, 0], reduced_features[:, 1], c=sorted_label, cmap='tab20', s=16, alpha=0.4)

            for i, center in enumerate(centers):
                ax.scatter(center[0], center[1], s=10, marker='o', color='black')
                if down_dim_method == 'PCA':
                    ax.text(center[0] + np.mean(np.abs(centers) / 2), center[1], f'{i + 1}', color='black', fontsize=16, ha='center', va='center')
                else:
                    ax.text(center[0] + np.mean(np.abs(centers) / 5), center[1], f'{i + 1}', color='black', fontsize=16, ha='center', va='center')

            # if down_dim_method == 'PCA':
            #     ax.set_xlabel('PC 1', fontsize=15)
            #     ax.set_ylabel('PC 2', fontsize=15)
            # elif down_dim_method in ['TSNE2', 'TSNE3', 'None']:
            #     ax.set_xlabel('TSNE 1', fontsize=15)
            #     ax.set_ylabel('TSNE 2', fontsize=15)
            # elif down_dim_method == 'UMAP':
            #     ax.set_xlabel('UMAP 1', fontsize=15)
            #     ax.set_ylabel('UMAP 2', fontsize=15)

            if use_model:
                ax.set_title(f'SimSort embedding ({down_dim_method[:4]}), True Labels', fontsize=18)
            else:
                ax.set_title(f'Raw feature ({down_dim_method[:4]}), True Labels', fontsize=18)

            plt.savefig(os.path.join(self.root_path, f'predict_{ARI * 100:.2f}.pdf'), bbox_inches='tight', dpi=300)
            print(f"ARI: {ARI * 100:.2f}%")
            plt.close()

            fig_height = max(6, 1 * n_components)
            fig = plt.figure(figsize=(8, fig_height))

            grid_rows = max(7, n_components + 2)
            main_plot_rows = min(5, grid_rows // 2)

            gs = fig.add_gridspec(grid_rows, 5, width_ratios=[1, 0.3, 4.5, 0.3, 1])

            axes_left = [fig.add_subplot(gs[i, 0]) for i in range((n_components + 1) // 2)]
            axes_right = [fig.add_subplot(gs[i, 4]) for i in range(n_components // 2)]

            ax_center = fig.add_subplot(gs[:main_plot_rows, 2])

            left_corners = []
            right_corners = []

            if use_model:
                num_label, unique_labels = pd.factorize(labels)
                n_components = len(unique_labels)
                colormap = plt.cm.get_cmap('tab20', n_components)
                norm = plt.Normalize(vmin=0, vmax=n_components - 1)

                for i in range(n_components):
                    unit_data = data[labels == unique_labels[i]]
                    mean_waveform = unit_data.mean(axis=0)
                    max_channel = np.argmax(np.max(np.abs(mean_waveform), axis=0))
                    
                    if i < (n_components + 1) // 2:
                        ax = axes_left[i]
                    else:
                        ax = axes_right[i - (n_components + 1) // 2]
                    
                    cluster_color = colormap(norm(i))
                    
                    for sample in unit_data[:30]:
                        ax.plot(sample[:, max_channel], linewidth=0.5, alpha=0.5, color=cluster_color)

                    ax.tick_params(axis='y', length=1, labelsize=5)
                    ax.tick_params(axis='x', length=1, labelsize=5, pad=0.5)
                    ax.set_ylabel('z-scored \n Amplitude', fontsize=5)

                    for spine in ax.spines.values():
                        spine.set_linewidth(0.5)

                    bbox = ax.get_position()
                    if i < (n_components + 1) // 2:
                        right_upper = (bbox.x1, bbox.y1)
                        left_corners.append(right_upper)
                    else:
                        left_upper = (bbox.x0, bbox.y1)
                        right_corners.append(left_upper)

                centers = np.array([reduced_features[labels == unique_labels[i]].mean(axis=0) for i in range(n_components)])
                scatter = ax_center.scatter(reduced_features[:, 0], reduced_features[:, 1], c=num_label, cmap='tab20', s=15, alpha=0.4)

                for i, center in enumerate(centers):
                    ax_center.scatter(center[0], center[1], s=5, marker='o', color='black')
                    ax_center.text(center[0] + np.mean(np.abs(centers) / 7), center[1], f'{i + 1}', color='black', fontsize=10, ha='center', va='center')

                    center_canvas = fig.transFigure.inverted().transform(ax_center.transData.transform(center))
                    
                    if i < (n_components + 1) // 2:
                        point = left_corners[i]
                    else:
                        point = right_corners[i - (n_components + 1) // 2]
                    
                    line = Line2D([center_canvas[0], point[0]], [center_canvas[1], point[1]],
                                transform=fig.transFigure, color="gray", linestyle="--", linewidth=0.5)
                    fig.lines.append(line)

                # if down_dim_method == 'PCA':
                #     ax_center.set_xlabel('PC 1', fontsize=12)
                #     ax_center.set_ylabel('PC 2', fontsize=12)
                # elif down_dim_method in ['TSNE2', 'TSNE3', 'None']:
                #     ax_center.set_xlabel('TSNE 1', fontsize=12)
                #     ax_center.set_ylabel('TSNE 2', fontsize=12)
                # elif down_dim_method == 'UMAP':
                #     ax_center.set_xlabel('UMAP 1', fontsize=12)
                #     ax_center.set_ylabel('UMAP 2', fontsize=12)

                ax_center.set_title(f'SimSort embedding ({down_dim_method[:4]}), \nGMM Pred Labels {ARI:.2f}', fontsize=14, ha='center')

                plt.savefig(f'cluster_waveforms_predict_{ARI * 100:.2f}.pdf', bbox_inches='tight', dpi=300)
                plt.show()

        return ARI