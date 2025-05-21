import os
import time
import umap
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from matplotlib.lines import Line2D

from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils.score import calculate_n_match, calculate_metrics
from data_factory.data_provider import LOODatasetManager, DataProvider
from model import SpikeDetector, RepresentationExtractor

class SortingExperiment:
    def __init__(
        self,
        root_path='./',
        args=None,
        data_info=None,
        test_dataset_type=None,
        sorting_chunk_size=800000,
        dmodel_save_path=None,
        emodel_save_path=None,
        use_true_snippets=False,
        use_true_clusters=False,
    ):

        self.root_path = root_path
        self.args = args
        self.data_info = data_info
        self.use_true_snippets = use_true_snippets
        self.use_true_clusters = use_true_clusters
        self.test_dataset_type = (
            test_dataset_type
            if test_dataset_type is not None
            else args.test_dataset_type
        )
        self.segment_size = args.sorting_chunk_size

        self.emodel_args = {
            'input_size': args.e_input_size,
            'hidden_size': args.e_hidden_size,
            'num_layers': args.e_num_layers,
            'learning_rate': args.e_learning_rate,
            'weight_decay': 1e-5,
            'nhead': args.e_nhead,
            'dropout': args.e_dropout,
        }

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

        self.emodel_arch = args.emodel_arch
        self.dmodel_arch = args.dmodel_arch

        self.emodel_save_path = (
            emodel_save_path if emodel_save_path is not None else args.emodel_save_path
        )
        self.dmodel_save_path = (
            dmodel_save_path if dmodel_save_path is not None else args.dmodel_save_path
        )

        self.detect_method = args.detect_method
        self.down_dim_method = args.down_dim_method
        self.cluster_method = args.cluster_method

    def get_dataset(self):
        loader = DataProvider(args=self.args, data_info=self.data_info, flag='test')
        self.test_data, self.units_spike_train = loader.get_data(
            dataset_type=self.test_dataset_type
        )
        if len(self.test_data.shape) == 3:
            if self.test_data.shape[0] == 1:
                self.test_data = self.test_data.squeeze()

            elif self.test_data.shape[1] < self.segment_size:
                print(f"Adjusting segment_size from {self.segment_size} to {self.test_data.shape[1]} due to insufficient data.")
                self.segment_size = self.test_data.shape[1]

        elif len(self.test_data.shape) == 2:
            if len(self.test_data) < self.segment_size:
                print(f"Adjusting segment_size from {self.segment_size} to {len(self.test_data)} due to insufficient data.")
                self.segment_size = len(self.test_data)

        # self.test_data = self.test_data[:self.segment_size * 1, :]
        # self.units_spike_train = self.units_spike_train[self.units_spike_train[:, 0] < self.segment_size * 1]
        
        print(f'test data {self.test_dataset_type} {self.test_data.shape}')
        self.sampling_frequency = loader.sampling_frequency
        return self.test_data, self.units_spike_train

    def detect_spike(self, test_data, units_spike_train):
        if len(test_data) < self.segment_size:
            print(f"Adjusting segment_size from {self.segment_size} to {len(test_data)} during spike detection.")
            self.segment_size = len(test_data)

        detector = SpikeDetector(
            root_path=self.root_path,
            model_arch=self.dmodel_arch,
            model_args=self.dmodel_args,
            mode='test',
            gpu=0,
        )

        detector.print_model()
        detector.load_model(self.dmodel_save_path)
        self.peak_positions, self.snippets, self.gt_spike_train, self.true_snippets = (
            detector.detect(
                data=test_data,
                units_spike_train=units_spike_train,
                sampling_frequency=self.sampling_frequency,
                detect_method=self.detect_method,
                detect_chunk_size=self.args.detect_chunk_size,
                model_save_path=self.dmodel_save_path,
                model_th=self.args.model_threshold,
                threshold=self.args.threshold,
                only_detect = True,
                return_true_snippets=True,
                return_gt_spike_train=True,
                sorting_chunk_size=self.segment_size,
            )
        )
        return (
            self.peak_positions,
            self.snippets,
            self.gt_spike_train,
            self.true_snippets,
        )

    def _true_embedding_plot(self, gt_spike_train, data_idx):
        extractor = RepresentationExtractor(
            root_path=self.root_path,
            model_arch=self.emodel_arch,
            model_args=self.emodel_args,
            mode='test',
            gpu=0,
        )

        test_data = np.array(self.true_snippets[data_idx])
        features = extractor.test(
            data=test_data, batch_size=256, saved_model_path=self.emodel_save_path
        )

        if self.down_dim_method == 'PCA':
            pca = PCA(n_components=5)
            reduced_features = pca.fit_transform(features)

        elif self.down_dim_method == 'TSNE2':
            tsne = TSNE(n_components=2, random_state=self.args.tsne_seed)
            reduced_features = tsne.fit_transform(features)
        
        elif self.down_dim_method == 'TSNE3':
            tsne = TSNE(n_components=3, random_state=self.args.tsne_seed)
            reduced_features = tsne.fit_transform(features)

        elif self.down_dim_method == 'UMAP':
            reducer = umap.UMAP()
            reduced_features = reducer.fit_transform(features)

        elif self.down_dim_method == 'None':
            reduced_features = features

        else:
            raise ValueError('Invalid dimensionality reduction method')

        scaler = StandardScaler()
        reduced_features = scaler.fit_transform(reduced_features)

        gt_spike_train = gt_spike_train[np.argsort(gt_spike_train[:, 0])]
        labels = gt_spike_train[:, 1].astype(int)
        unit_ids = np.unique(labels)
        num_label = pd.factorize(labels)[0]

        # plot clustering
        plt.figure(figsize=(5, 4))
        if self.down_dim_method == 'None':
            reduced_features = PCA(n_components=5).fit_transform(reduced_features)

        centers = np.array(
            [
                reduced_features[labels == unit_ids[i]].mean(axis=0)
                for i in range(len(unit_ids))
            ]
        )
        plt.scatter(
            reduced_features[:, 0],
            reduced_features[:, 1],
            c=num_label,
            cmap='tab20',
            s=15,
            alpha=0.5,
        )

        for i, center in enumerate(centers):
            plt.scatter(center[0], center[1], s=10, marker='o', color='black')
            plt.text(
                center[0] + np.mean(np.abs(centers) / 5),
                center[1],
                f'{unit_ids[i]}',
                color='black',
                fontsize=12,
                ha='center',
                va='center',
            )

        if self.down_dim_method == 'PCA':
            plt.xlabel('PC 1', fontsize=8)
            plt.ylabel('PC 2', fontsize=8)

        elif self.down_dim_method in ['TSNE2', 'TSNE3']:
            plt.xlabel('TSNE 1', fontsize=8)
            plt.ylabel('TSNE 2', fontsize=8)

        elif self.down_dim_method == 'UMAP':
            plt.xlabel('UMAP 1', fontsize=8)
            plt.ylabel('UMAP 2', fontsize=8)
        
        elif self.down_dim_method == 'None':
            plt.xlabel('SimSort PC1', fontsize=8)
            plt.ylabel('SimSort PC2', fontsize=8)

        else:
            raise ValueError('Invalid dimensionality reduction method')
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.root_path, f'{self.test_dataset_type}_true_snippets_embedding.png'
            ),
            dpi=300,
        )
        plt.close()

    def cluster(self, gt_spike_train, snippets_to_use):
        extractor = RepresentationExtractor(
            root_path=self.root_path,
            model_arch=self.emodel_arch,
            model_args=self.emodel_args,
            mode='test',
            gpu=0,
        )

        labels_list = []
        reduced_features_list = []
        extractor.print_model()
        for i in tqdm(range(len(snippets_to_use)), desc="Extracting Segments"):
            test_data = snippets_to_use[i]
            features = extractor.test(
                data=test_data, batch_size=256, saved_model_path=self.emodel_save_path
            )
            reduced_features = features

            if self.down_dim_method == 'PCA':
                pca = PCA(n_components=5)
                features = pca.fit_transform(features)
                reduced_features = features

            elif self.down_dim_method == 'TSNE2':
                tsne = TSNE(n_components=2, random_state=self.args.tsne_seed)
                reduced_features = tsne.fit_transform(features)

            elif self.down_dim_method == 'TSNE3':
                tsne = TSNE(n_components=3, random_state=self.args.tsne_seed)
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

            # calculate the number of components for clustering methods that require cluster number
            units_spike_train = gt_spike_train[i]
            unit_ids = np.unique(units_spike_train[:, 1])
            if self.test_dataset_type == 'crcns':
                n_components = len(unit_ids) + 1
            else:
                n_components = len(unit_ids)

            if self.cluster_method == 'DBSCAN':
                labels = DBSCAN(
                    eps=self.args.eps, min_samples=self.args.min_samples, n_jobs=-1
                ).fit_predict(reduced_features)

            elif self.cluster_method == 'KMeans':
                kmeans = KMeans(n_clusters=n_components, random_state=42)
                labels = kmeans.fit_predict(reduced_features)

            elif self.cluster_method == 'GMM':
                gmm = GaussianMixture(n_components=n_components, random_state=42)
                gmm.fit(reduced_features)
                labels = gmm.predict(reduced_features)

            elif self.cluster_method == 'MeanShift' or self.cluster_method == 'MS':
                from sklearn.cluster import estimate_bandwidth
                bandwidth = estimate_bandwidth(reduced_features, quantile=self.args.quantile, n_samples=500)
                ms = MeanShift(n_jobs=-1, bandwidth=bandwidth)
                labels = ms.fit_predict(reduced_features)
                
            elif self.cluster_method == 'SpectralClustering' or self.cluster_method == 'SC':
                connectivity = kneighbors_graph(reduced_features, n_neighbors=10, include_self=False)
                connectivity = 0.5 * (connectivity + connectivity.T)
                sc = SpectralClustering(n_clusters=n_components, affinity='precomputed', random_state=42)
                labels = sc.fit_predict(connectivity.toarray())


            else:
                raise ValueError('Invalid clustering method')

            labels_list.append(labels)
            reduced_features_list.append(reduced_features)

        return labels_list, reduced_features_list

    def evaluate_with_ground_truth(
        self, gt_spike_trains, detected_spikes, reduced_features, plot=False
    ):

        gt_times = gt_spike_trains[:, 0] / self.sampling_frequency
        gt_labels = gt_spike_trains[:, 1]
        detected_spikes = np.array(detected_spikes) / self.sampling_frequency

        #time_window = 0.003 if self.test_dataset_type in ['bbp', 'bbp_multi'] else 0.001
        time_window = 0.001
        units = np.unique(gt_labels)

        total_tp, total_fp, total_fn = 0, 0, 0
        matched_units = np.full(len(detected_spikes), -1, dtype=int)
        unit_matched = {
            unit: np.zeros(sum(gt_labels == unit), dtype=bool) for unit in units
        }

        for i, spike in enumerate(detected_spikes):
            best_match_unit = None
            best_match_idx = None
            best_match_distance = float('inf')

            for unit in units:
                unit_gt_times = gt_times[gt_labels == unit]
                matches = np.where(np.abs(unit_gt_times - spike) <= time_window)[0]

                if matches.size > 0:
                    closest_idx = matches[
                        np.argmin(np.abs(unit_gt_times[matches] - spike))
                    ]
                    distance = abs(unit_gt_times[closest_idx] - spike)

                    if (
                        not unit_matched[unit][closest_idx]
                        and distance < best_match_distance
                    ):
                        best_match_unit = unit
                        best_match_idx = closest_idx
                        best_match_distance = distance

            if best_match_unit is not None:
                unit_matched[best_match_unit][best_match_idx] = True
                matched_units[i] = best_match_unit
                total_tp += 1
            else:
                total_fp += 1

        for unit in units:
            total_fn += len(unit_matched[unit]) - np.sum(unit_matched[unit])

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        accuracy = (
            total_tp / (total_tp + total_fp + total_fn)
            if (total_tp + total_fp + total_fn) > 0
            else 0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        if plot:
            if self.down_dim_method == 'None':
                reduced_features = PCA(n_components=5).fit_transform(reduced_features)
            unit_color_map = {unit: idx for idx, unit in enumerate(units)}
            colors = []
            for unit in matched_units:
                if unit == -1:
                    colors.append('gray')
                else:
                    colors.append(plt.cm.tab20(unit_color_map[unit] % 20))

            plt.figure(figsize=(7, 3.5))
            plt.scatter(
                reduced_features[:, 0],
                reduced_features[:, 1],
                c=colors,
                s=20,
                alpha=0.5,
            )
            plt.grid(True, linestyle='--', alpha=0.5)

            for unit, color_idx in unit_color_map.items():
                plt.scatter(
                    [], [], color=plt.cm.tab20(color_idx % 20), label=f'Unit {unit}'
                )
            plt.scatter([], [], color='gray', label='Unmatched')

            plt.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))

            if self.down_dim_method == 'None':
                plt.xlabel('SimSort PC1', fontsize=8)
                plt.ylabel('SimSort PC2', fontsize=8)
            else:
                plt.xlabel('TSNE 1', fontsize=8)
                plt.ylabel('TSNE 2', fontsize=8)

            plt.tight_layout(rect=[0, 0, 0.8, 1])

            plt.savefig(
                os.path.join(
                    self.root_path,
                    f'{self.test_dataset_type}_{self.detect_method}_{self.down_dim_method}_embedding.png',
                ),
                dpi=300,
                bbox_inches="tight",
            )

        return accuracy, recall, precision, f1

    def evalue(
        self,
        test_data: np.ndarray = None,
        peak_positions: np.ndarray = None,
        snippets: np.ndarray = None,
        true_snippets: np.ndarray = None,
        segment_spike_train: np.ndarray = None,
        labels: np.ndarray = None,
        reduced_features: np.ndarray = None,
        save_csv=False,
        plot=True,
    ):

        unit_ids = np.unique(segment_spike_train[:, 1])

        if self.cluster_method == 'DBSCAN':
            n_components = len(np.unique(labels)) - 1
        else:
            n_components = len(np.unique(labels))

        #time_window = 0.003 if self.test_dataset_type in ['bbp', 'bbp_multi'] else 0.001
        time_window = 0.001

        spike_trains = []
        for i in range(n_components):
            spike_train = (
                peak_positions[np.argwhere(labels == i).flatten()]
                / self.sampling_frequency
            )
            spike_trains.append(spike_train)

        total_accuracy = [0] * len(unit_ids)
        total_recall = [0] * len(unit_ids)
        total_precision = [0] * len(unit_ids)
        total_f1 = [0] * len(unit_ids)

        match_matrix = np.zeros((len(unit_ids), len(spike_trains)))

        for ids in range(len(unit_ids)):
            gt_spike_train = (
                segment_spike_train[
                    np.argwhere(segment_spike_train[:, 1] == unit_ids[ids]).flatten(), 0
                ]
                / self.sampling_frequency
            )
            for i in range(len(spike_trains)):
                n_matches, _, _ = calculate_n_match(
                    spike_trains[i], gt_spike_train, time_window
                )
                match_matrix[ids, i] = n_matches

        row_ind, col_ind = linear_sum_assignment(-match_matrix)

        matched_units = set(row_ind)
        all_units = set(range(len(unit_ids)))
        unmatched_units = all_units - matched_units

        match_idxs = [None] * len(
            unit_ids
        )  # Initialize match_idxs to keep track of matched clusters

        # Update metrics for matched units
        for row_idx, col_idx in zip(row_ind, col_ind):
            gt_spike_train = (
                segment_spike_train[
                    np.argwhere(
                        segment_spike_train[:, 1] == unit_ids[row_idx]
                    ).flatten(),
                    0,
                ]
                / self.sampling_frequency
            )
            accuracy, recall, precision = calculate_metrics(
                gt_spike_train, spike_trains[col_idx], time_window
            )
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

            total_accuracy[row_idx] = accuracy
            total_recall[row_idx] = recall
            total_precision[row_idx] = precision
            total_f1[row_idx] = f1
            match_idxs[row_idx] = col_idx  # Store the matched cluster index

            if save_csv:
                save_folder = os.path.join(self.root_path, 'sorting_results')
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                results = pd.DataFrame(columns=['accuracy', 'recall', 'precision', 'f1'])
                results['accuracy'] = [accuracy]
                results['recall'] = [recall]
                results['precision'] = [precision]
                results['f1'] = [f1]

                csv_path = os.path.join(
                    save_folder,
                    f'{self.test_dataset_type}_unit{unit_ids[row_idx]}_{self.detect_method}_{self.down_dim_method}_results.csv',
                )
                if os.path.exists(csv_path):
                    results.to_csv(csv_path, mode='a', header=False, index=False)
                else:
                    results.to_csv(csv_path)

        # Handle unmatched units
        for unmatched_id in unmatched_units:
            total_accuracy[unmatched_id] = 0
            total_recall[unmatched_id] = 0
            total_precision[unmatched_id] = 0
            total_f1[unmatched_id] = 0
            match_idxs[unmatched_id] = None  # No match for unmatched units

        # Plotting results
        if plot:
            ## Plot detection
            # trace_seg = test_data[200:1200] # change to plot all channels
            # peak_positions_ = peak_positions[(peak_positions < 1200) & (peak_positions > 200)] - 200
            # segment_spike_train_ = segment_spike_train[
            #     (segment_spike_train[:, 0] < 1200) & (segment_spike_train[:, 0] > 200) 
            # ] - 200

            # detected_spike_train = peak_positions_.astype(int)
            # gt_spike_train = segment_spike_train_[
            #     np.argwhere(segment_spike_train_[:, 0]).flatten(), 0
            # ].astype(int)

            # _, _, _, detected_matched_boolean, gt_matched_boolean = calculate_n_match(
            #     detected_spike_train / self.sampling_frequency,
            #     gt_spike_train / self.sampling_frequency,
            #     time_window,
            #     return_match_boolean=True,
            # )

            # positive_detected_spike_train = detected_spike_train[
            #     detected_matched_boolean
            # ]
            # np_detected_spike_train = detected_spike_train[~detected_matched_boolean]
            # fn_gt_spike_train = gt_spike_train[~gt_matched_boolean]

            # fig, axes = plt.subplots(4, 1, figsize=(15, 10))
            # for i in range(4):
            #     axes[i].plot(np.arange(len(trace_seg)), trace_seg[:, i], c='gray', alpha=0.5)
            #     axes[i].scatter(
            #         positive_detected_spike_train,
            #         trace_seg[positive_detected_spike_train, i],
            #         c='blue',
            #         s=25,
            #         marker='x',
            #         label='True Positive Detected',
            #     )
            #     axes[i].scatter(
            #         np_detected_spike_train,
            #         trace_seg[np_detected_spike_train, i],
            #         c='red',
            #         s=25,
            #         marker='x',
            #         label='False Positive Detected',
            #     )
            #     axes[i].scatter(
            #         fn_gt_spike_train,
            #         trace_seg[fn_gt_spike_train, i],
            #         c='k',
            #         s=10,
            #         label='Missed Ground Truth',
            #     )
            #     #axes[i].legend()
            #     axes[i].set_xlabel('Time (samples)', fontsize=10)
            #     axes[i].set_ylabel('z-scored \n Amplitude')
            #     axes[i].set_title(f'Channel {i + 1}')
            
            # plt.legend()
            # plt.tight_layout()
            # plt.savefig(
            #     os.path.join(
            #         self.root_path,
            #         f'{self.test_dataset_type}_{self.detect_method}_{self.down_dim_method}_detection_{time.time()}.png',
            #     )
            # )
            
            # plt.close()

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

            plt.savefig(f'cluster_waveforms_{time.time()}.png', bbox_inches='tight', dpi=300)
            plt.show()

        if (
            len(total_accuracy) > 0
            and len(total_recall) > 0
            and len(total_precision) > 0
            and len(total_f1) > 0
        ):
            return (
                np.mean(total_accuracy),
                np.mean(total_recall),
                np.mean(total_precision),
                np.mean(total_f1),
            )
        else:
            return None, None, None, None

    def run(self, test_data):
        if self.use_true_snippets:
            snippets_to_use = self.true_snippets
        else:
            snippets_to_use = self.snippets

        labels_list, reduced_features_list = self.cluster(
            self.gt_spike_train, snippets_to_use
        )
        accuracy_list, recall_list, precision_list, f1_list = [], [], [], []
        segment_length = test_data.shape[0] // len(labels_list)

        for i, (label, reduced_feature) in tqdm(
            enumerate(zip(labels_list, reduced_features_list)), desc="Evaluating Scores"
        ):
            segment_data = test_data[segment_length * i : segment_length * (i + 1)]

            if self.use_true_snippets:
                segment_peak_positions = self.gt_spike_train[i][:, 0]
            else:
                segment_peak_positions = self.peak_positions[i]

            if i == 0:  # Only plot embedding for the first segment
                if self.gt_spike_train[i].shape[0] > 30:
                    self._true_embedding_plot(self.gt_spike_train[i], i)
                else:
                    print('True embedding plot is not available for this dataset')
                plot = True
            else:
                plot = False

            if self.use_true_clusters:
                accuracy, recall, precision, f1 = self.evaluate_with_ground_truth(
                    self.gt_spike_train[i],
                    segment_peak_positions,
                    reduced_feature,
                    plot=plot,
                )
            else:
                accuracy, recall, precision, f1 = self.evalue(
                    segment_data,
                    segment_peak_positions,
                    snippets_to_use[i],
                    self.true_snippets[i],
                    self.gt_spike_train[i],
                    label,
                    reduced_feature,
                    save_csv=True,
                    plot=plot,
                )
            accuracy_list.append(accuracy)
            recall_list.append(recall)
            precision_list.append(precision)
            f1_list.append(f1)

        accuracy_list = [x for x in accuracy_list if x is not None]
        recall_list = [x for x in recall_list if x is not None]
        precision_list = [x for x in precision_list if x is not None]
        f1_list = [x for x in f1_list if x is not None]

        print(
            f'Average Accuracy: {np.mean(accuracy_list):.2f}, '
            f'Average Recall: {np.mean(recall_list):.2f}, '
            f'Average Precision: {np.mean(precision_list):.2f}, '
            f'Average F1: {np.mean(f1_list):.2f}'

        )

        return np.mean(accuracy_list), np.mean(recall_list), np.mean(precision_list), np.mean(f1_list)

    def test(self):
        self.get_dataset()
        if self.use_true_snippets:
            print("Using true_snippets directly for clustering and evaluation...")
            self.detect_spike(self.test_data, self.units_spike_train)
            accuracy, recall, precision = self.run(self.test_data)

        else:
            print("Using detected spikes for evaluation...")

            if (
                self.test_dataset_type in ['bbp', 'bbp_multi']
                and len(self.test_data.shape) == 3
            ):
                # Handle multi-segment datasets
                csv_path = os.path.join(
                    self.root_path,
                    f'{self.test_dataset_type}_{self.detect_method}_sorting_results.csv',
                )
                if os.path.exists(csv_path):
                    os.remove(csv_path)

                for t in range(len(self.test_data)):
                    if t < 20:  # Process the first 20 trials
                        test_data = (
                            np.array([self.test_data[t]]).squeeze().astype(np.float64)
                        )
                        print(f'test_data shape {test_data.shape}')
                        units_spike_train = (
                            np.array([self.units_spike_train[t]])
                            .squeeze()
                            .astype(np.float64)
                        )

                        self.detect_spike(test_data, units_spike_train)
                        accuracy, recall, precision, f1 = self.run(test_data)

                        results = pd.DataFrame(
                            columns=['accuracy', 'recall', 'precision', 'f1']
                        )
                        results['accuracy'] = [accuracy]
                        results['recall'] = [recall]
                        results['precision'] = [precision]
                        results['f1'] = [f1]

                        results.to_csv(
                            csv_path,
                            mode='a' if os.path.exists(csv_path) else 'w',
                            header=not os.path.exists(csv_path),
                            index=False,
                        )
                    else:
                        continue
            else:
                # Handle single-segment datasets
                self.detect_spike(self.test_data, self.units_spike_train)
                accuracy, recall, precision, f1 = self.run(self.test_data)

                csv_path = os.path.join(
                    self.root_path,
                    f'{self.test_dataset_type}_{self.detect_method}_sorting_results.csv',
                )
                results = pd.DataFrame(columns=['accuracy', 'recall', 'precision', 'f1'])
                results['accuracy'] = [accuracy]
                results['recall'] = [recall]
                results['precision'] = [precision]
                results['f1'] = [f1]

                results.to_csv(
                    csv_path,
                    mode='a' if os.path.exists(csv_path) else 'w',
                    header=not os.path.exists(csv_path),
                    index=False,
                )

