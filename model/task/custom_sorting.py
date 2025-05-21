import os
import time
import yaml
import umap
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp

from utils.tool import bandpass_filter, highpass_filter, spatial_whitening, zscore
from model import SpikeDetector, RepresentationExtractor


class SortingConfig:
    """
    Note that the following attributes can be adjusted via the YAML configuration file.
    And for most of the time, you should not change the default values here.

    """
    def __init__(self, **kwargs):
        self.task_name = "sorting"
        self.emodel_arch = "gru"
        self.dmodel_arch = "transformer"
        self.detect_method = "model"
        self.down_dim_method = "UMAP"
        self.cluster_method = "MS"
        self.detect_chunk_size = 6000000
        self.sorting_chunk_size = 6000000
        self.quantile = 0.12
        self.only_test = 1
        self.model_threshold = 0.97
        self.e_input_size = 4
        self.e_hidden_size = 512
        self.e_num_layers = 1
        self.input_size = 4
        self.hidden_size = 256
        self.num_layers = 4
        self.nhead = 4
        self.dropout = 0.2
        self.emodel_save_path = "simsort_pretrained/extractor_bbp_L1-L5-8192/saved_models"
        self.dmodel_save_path = "simsort_pretrained/detector_bbp_L1-L5-8192/saved_models"

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

class SortingTask:
    """
    SimSort class for custom spike sorting task.

    Attributes:
    ----------
    root_path : str
        Directory for saving or loading files.
    yaml_path : str
        Path to the YAML configuration file.
    recording : object
        SpikeInterface RecordingExtractor object for neural data.
    dmodel_save_path : str
        Path to the pre-trained spike detection model (overrides YAML if set).
    emodel_save_path : str
        Path to the pre-trained spike identification model (overrides YAML if set).
    cluster_method : str
        Clustering method ('MeanShift', 'KMeans', 'GMM', etc.), defaults to 'MeanShift', adjustable via YAML.
    n_clusters : int
        Number of clusters for applicable methods (e.g., KMeans, GMM).
    ms_quantile : float
        Quantile for MeanShift clustering, adjustable via YAML.
    verbose : bool
        Enables detailed logging if True.
    """

    def __init__(self, 
                 root_path='./', 
                 yaml_path='SimSort.yaml', 
                 recording=None, 
                 dmodel_save_path=None, 
                 emodel_save_path=None, 
                 cluster_method=None,
                 n_clusters=None,
                 ms_quantile=None,
                 tetrode_idx=None,
                 verbose=False
                 ):
        
        self.root_path = root_path
        self.recording = recording
        self.verbose = verbose
        self.config = self._load_config(yaml_path)
        self.segment_size = self.config.sorting_chunk_size

        self.emodel_args = {
            'input_size': self.config.e_input_size,
            'hidden_size': self.config.e_hidden_size,
            'num_layers': self.config.e_num_layers,
        }

        self.dmodel_args = {
            'input_size': self.config.input_size,
            'hidden_size': self.config.hidden_size,
            'num_layers': self.config.num_layers,
            'dropout': self.config.dropout,
            'nhead': self.config.nhead,
        }

        self.emodel_arch = self.config.emodel_arch
        self.dmodel_arch = self.config.dmodel_arch
        self.emodel_save_path = emodel_save_path if emodel_save_path else self.config.emodel_save_path
        self.dmodel_save_path = dmodel_save_path if dmodel_save_path else self.config.dmodel_save_path
        self.detect_method = self.config.detect_method
        self.down_dim_method = self.config.down_dim_method
        self.cluster_method = cluster_method if cluster_method else self.config.cluster_method
        self.MS_quantile = ms_quantile if ms_quantile else self.config.quantile
        self.n_components = n_clusters
        self.tetrode_idx = tetrode_idx

    def _load_config(self, yaml_path):
        try:
            with open(yaml_path, 'r') as f:
                config_data = yaml.safe_load(f)
                if self.verbose:
                    print(f"Loaded configuration from {yaml_path}")
                return SortingConfig(**config_data)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file '{yaml_path}' not found.")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")

    def get_data(self):
        self.test_data = self.recording.get_traces()
        self.sampling_frequency = self.recording.get_sampling_frequency()

        if self.test_data.shape[1] > 4 and self.tetrode_idx is None:
            if self.verbose:
                print("More than 4 channels detected. Using only the first 4 channels.")
            self.test_data = self.test_data[:, :4]

        elif self.tetrode_idx is not None:
            if self.verbose:
                print(f"Using tetrode index {self.tetrode_idx}.")
            self.test_data = self.test_data[:, self.tetrode_idx*4:(self.tetrode_idx+1)*4]
            print(f'Using tetrode channel {self.tetrode_idx*4}-{(self.tetrode_idx+1)*4-1}')

        if len(self.test_data) < self.segment_size:
            if self.verbose:
                print(f"Adjusting segment_size from {self.segment_size} to {len(self.test_data)} due to insufficient data.")
            self.segment_size = len(self.test_data)

        self.test_data = self.test_data[:self.segment_size * 1, :]

        if self.verbose:
            print(f"After adjusting, test data shape: {self.test_data.shape}")
        print('Data shape:', self.test_data.shape)
        return self._data_preprocessing()

    def _data_preprocessing(self):
        filtered_data = bandpass_filter(self.test_data, 250, 10000, self.sampling_frequency)
        data = zscore(filtered_data)
        self.test_data = spatial_whitening(data)
        print('Data preprocessing done')

    def detect_spike(self, test_data):
        if len(test_data) < self.segment_size:
            if self.verbose:
                print(f"Adjusting segment_size from {self.segment_size} to {len(test_data)} during spike detection.")
            self.segment_size = len(test_data)

        detector = SpikeDetector(
            root_path=self.root_path,
            model_arch=self.dmodel_arch,
            model_args=self.dmodel_args,
            mode='test',
            gpu=0,
        )
        if self.verbose:
            detector.print_model()

        detector.load_model(self.dmodel_save_path)
        self.peak_positions, self.snippets = (
            detector.detect(
                data=test_data,
                sampling_frequency=self.sampling_frequency,
                detect_method=self.detect_method,
                detect_chunk_size=self.config.detect_chunk_size,
                model_save_path=self.dmodel_save_path,
                model_th=self.config.model_threshold,
                only_detect = True,
                sorting_chunk_size=self.segment_size,
            )
        )
        return (
            self.peak_positions,
            self.snippets,
        )

    def cluster(self):
        extractor = RepresentationExtractor(
            root_path=self.root_path,
            model_arch=self.emodel_arch,
            model_args=self.emodel_args,
            mode='test',
            gpu=0,
        )
        if self.verbose:
            extractor.print_model()

        labels_list = []
        reduced_features_list = []
        for i in tqdm(range(len(self.snippets)), desc="Processing Segments"):
            test_data = self.snippets[i]
            features = extractor.test(
                data=test_data, batch_size=256, saved_model_path=self.emodel_save_path
            )
            reduced_features = features

            if self.down_dim_method == 'PCA':
                pca = PCA(n_components=5)
                features = pca.fit_transform(features)
                reduced_features = features

            elif self.down_dim_method == 'TSNE2':
                tsne = TSNE(n_components=2)
                reduced_features = tsne.fit_transform(features)

            elif self.down_dim_method == 'TSNE3':
                tsne = TSNE(n_components=3)
                reduced_features = tsne.fit_transform(features)

            elif self.down_dim_method == 'UMAP':
                reducer = umap.UMAP()
                reduced_features = reducer.fit_transform(features)

            elif self.down_dim_method == 'None':
                reduced_features = reduced_features

            else:
                raise ValueError('Invalid dimensionality reduction method')
            
            scaler = StandardScaler()
            reduced_features = scaler.fit_transform(reduced_features)

            if self.cluster_method == 'KMeans':
                kmeans = KMeans(n_clusters=self.n_components, random_state=42)
                labels = kmeans.fit_predict(reduced_features)

            elif self.cluster_method == 'GMM':
                gmm = GaussianMixture(n_components=self.n_components, random_state=42)
                gmm.fit(reduced_features)
                labels = gmm.predict(reduced_features)

            elif self.cluster_method == 'MeanShift' or self.cluster_method == 'MS':
                bandwidth = estimate_bandwidth(reduced_features, quantile=self.MS_quantile, n_samples=500)
                ms = MeanShift(n_jobs=-1, bandwidth=bandwidth)
                labels = ms.fit_predict(reduced_features)
                
            elif self.cluster_method == 'SpectralClustering' or self.cluster_method == 'SC':
                connectivity = kneighbors_graph(reduced_features, n_neighbors=10, include_self=False)
                connectivity = 0.5 * (connectivity + connectivity.T)
                sc = SpectralClustering(n_clusters=self.n_components, affinity='precomputed', random_state=42)
                labels = sc.fit_predict(connectivity.toarray())


            else:
                raise ValueError('Invalid clustering method')

            labels_list.append(labels)
            reduced_features_list.append(reduced_features)
            
            if self.verbose:
                unique_labels, counts = np.unique(labels, return_counts=True)
                for i in range(len(unique_labels)):
                    print(f"Cluster {unique_labels[i]+1}: {counts[i]} spikes")

        return labels_list, reduced_features_list

    def visualize(self, labels, reduced_features, snippets):
        if self.down_dim_method == 'None':
            reduced_features = PCA(n_components=5).fit_transform(reduced_features)
        n_components = len(np.unique(labels))
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

        num_label, unique_labels = pd.factorize(labels)
        colormap = plt.cm.get_cmap('tab20', n_components)
        norm = plt.Normalize(vmin=0, vmax=n_components - 1)

        for i in range(n_components):
            unit_data = snippets[labels == unique_labels[i]]
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

        if self.down_dim_method == 'PCA':
            ax_center.set_xlabel('PC 1', fontsize=8)
            ax_center.set_ylabel('PC 2', fontsize=8)

        elif self.down_dim_method in ['TSNE2', 'TSNE3']:
            ax_center.set_xlabel('TSNE 1', fontsize=8)
            ax_center.set_ylabel('TSNE 2', fontsize=8)

        elif self.down_dim_method == 'UMAP':
            ax_center.set_xlabel('UMAP 1', fontsize=8)
            ax_center.set_ylabel('UMAP 2', fontsize=8)
        
        elif self.down_dim_method == 'None':
            ax_center.set_xlabel('SimSort PC1', fontsize=8)
            ax_center.set_ylabel('SimSort PC2', fontsize=8)
        
        else:
            raise ValueError('Invalid dimensionality reduction method')
        
        plt.show()
        
    def run(self):
        self.get_data()
        self.peak_positions, self.snippets = self.detect_spike(self.test_data)
        self.labels_list, self.reduced_features_list = self.cluster()
        for i in range(len(self.labels_list)):
            self.visualize(self.labels_list[i], self.reduced_features_list[i], self.snippets[i])

    def get_results(self, save=False, output_dir='./simsort_results', overwrite=False):
        if not overwrite and os.path.exists(output_dir):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_dir = f"{output_dir}_{timestamp}"
            print(f"Output directory exists. Saving to: {output_dir}")
        if save:
            os.makedirs(output_dir, exist_ok=True)

        if not self.labels_list or not self.reduced_features_list:
            raise ValueError("No clustering results found. Ensure clustering is completed before saving.")

        for seg_idx, (labels, features) in enumerate(zip(self.labels_list, self.reduced_features_list)):
            num_label, unique_labels = pd.factorize(labels)
            
            if save:
                segment_dir = os.path.join(output_dir, f'segment_{seg_idx}')
                os.makedirs(segment_dir, exist_ok=True)
                
                labels_path = os.path.join(segment_dir, 'labels.csv')
                if not overwrite and os.path.exists(labels_path):
                    print(f"Labels file already exists: {labels_path}. Skipping.")
                else:
                    pd.DataFrame({'Cluster': num_label}).to_csv(labels_path, index=False)

            self.neurons_waveforms = []
            self.neurons_timestamps = []
            self.neurons_ids = []
            
            for i, cluster_id in enumerate(unique_labels):
                cluster_mask = labels == cluster_id
                
                cluster_waveforms = self.snippets[seg_idx][cluster_mask]
                cluster_timestamps = self.peak_positions[seg_idx][cluster_mask]
                
                if save:
                    cluster_waveform_path = os.path.join(segment_dir, f'waveforms_cluster_{i}.npy')
                    if not overwrite and os.path.exists(cluster_waveform_path):
                        print(f"Waveforms file already exists: {cluster_waveform_path}. Skipping.")
                    else:
                        np.save(cluster_waveform_path, cluster_waveforms)

                    cluster_timestamp_path = os.path.join(segment_dir, f'timestamps_cluster_{i}.csv')
                    if not overwrite and os.path.exists(cluster_timestamp_path):
                        print(f"Timestamps file already exists: {cluster_timestamp_path}. Skipping.")
                    else:
                        pd.DataFrame({'Timestamp': cluster_timestamps}).to_csv(cluster_timestamp_path, index=False)
                    if self.verbose:
                        print(f"Saved waveforms and timestamps for segment {seg_idx}, cluster {i}.")

                self.neurons_waveforms.append(cluster_waveforms)
                self.neurons_timestamps.append(cluster_timestamps)
                self.neurons_ids.append(i)

            self.metadata = {
                'sampling_frequency': self.sampling_frequency,
                'segment_size': self.segment_size,
                'cluster_method': self.cluster_method,
                'down_dim_method': self.down_dim_method,
            }
            metadata_path = os.path.join(output_dir, 'metadata.yaml')
            if save:
                if not overwrite and os.path.exists(metadata_path):
                    print(f"Metadata file already exists: {metadata_path}. Skipping.")
                else:
                    with open(metadata_path, 'w') as f:
                        yaml.dump(self.metadata, f)

                print(f"Results saved to: {output_dir}")

    def save_results(self, output_dir='./simsort_results', overwrite=False):
        if not overwrite and os.path.exists(output_dir):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_dir = f"{output_dir}_{timestamp}"
            print(f"Output directory exists. Saving to: {output_dir}")

        self.get_results(save=True, output_dir=output_dir, overwrite=overwrite)
            
