import os
import math
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import Union
from torch import nn, optim
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from model import GRU2, Transformer
from data_factory.data_loader import LFPDataset_test
from utils.score import calculate_n_match, calculate_metrics
from utils.detect import merge_predicted_labels, extract_snippets, detect_spikes

class SpikeDetector:   
    def __init__(
        self,
        root_path: str = '/',
        model_arch: str = 'gru',
        model_args: dict = None,
        mode: str = 'train',
        gpu: int = 0,
        **kwargs,
    ):

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
            self.model = GRU2(
                input_size=model_args.get('input_size', 4),
                hidden_size=model_args.get('hidden_size', 256),
                num_layers=model_args.get('num_layers', 1),
                output_size=model_args.get('output_size', 1),
            ).to(self.device)

        elif model_arch == 'transformer' or model_arch == 'Transformer':
            self.model = Transformer(
                input_size=model_args.get('input_size', 4),
                hidden_size=model_args.get('hidden_size', 256),
                num_layers=model_args.get('num_layers', 4),
                output_size=model_args.get('output_size', 1),
                nhead=model_args.get('nhead', 8),
                dropout=model_args.get('dropout', 0.1),
            ).to(self.device)
        else:
            raise ValueError('Model architecture not supported')

        if mode == 'train':
            pos_weight = torch.tensor([model_args.get('pos_weight', 1.0)]).to(
                self.device
            )
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            # self.criterion = nn.BCEWithLogitsLoss()
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=model_args.get('learning_rate', 1e-3),
                weight_decay=model_args.get('weight_decay', 0),
            )

    def load_data(self, dataset, batch_size, shuffle=True):
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4
        )
        return dataloader

    def load_model(self, checkpoint_path: str = 'saved_models'):
        checkpoint_full_path = os.path.join(checkpoint_path, 'checkpoint.pth')

        if not os.path.exists(checkpoint_full_path):
            print(f"Checkpoint not found at {checkpoint_full_path}.")
            return 0, []

        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(self.device)
        try:
            checkpoint = torch.load(
                checkpoint_full_path, map_location=device, weights_only=True
            )

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

        except FileNotFoundError:
            print(f"Checkpoint not found at {checkpoint_full_path}.")
            return 0, []
        except RuntimeError as e:
            print(f"Error loading model: {e}")
            return 0, []

        return start_epoch, losses

    def print_model(self):
        if self.model_arch == 'gru' or self.model_arch == 'GRU':
            print("Detection Model Configuration:")
            print(f"{'-'*30}")
            print(f"{'Parameter':<20} | {'Value':<10}")
            print(f"{'-'*30}")
            print(f"{'Model Architecture':<20} | {self.model_arch}")
            print(f"{'Input Size':<20} | {self.model_args['input_size']}")
            print(f"{'Hidden Size':<20} | {self.model_args['hidden_size']}")
            print(f"{'Number of Layers':<20} | {self.model_args['num_layers']}")
            print(f"{'-'*30}")
        elif self.model_arch == 'transformer' or self.model_arch == 'Transformer':
            print("Detection Model Configuration:")
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

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        tester: Union[None, callable] = None,
        num_epochs: int = 100,
        val_loader: torch.utils.data.DataLoader = None,
        save_model: bool = True,
        resume: bool = False,
        seed: int = 42,
        save_path='saved_models',
        predict: bool = True,
        predict_every: int = 50,  # trigger prediction every n epochs
        plot_loss: bool = True,
        warmup_steps: int = 4000,  # number of warmup steps
    ):

        checkpoint_path = os.path.join(save_path, 'checkpoint.pth')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if resume and os.path.exists(checkpoint_path):
            start_epoch, losses = self.load_model(checkpoint_path=save_path)
        else:
            start_epoch = 0
            losses = []

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        epoch = start_epoch

        # Define the Warmup + Cosine Annealing scheduler
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / warmup_steps  # Linear warmup
            else:
                progress_within_cycle = (current_step - warmup_steps) % (
                    num_epochs * len(train_loader)
                )  # Cycle step
                cycle_length = num_epochs * len(
                    train_loader
                )  # Define a full cycle length
                return 0.5 * (
                    1 + math.cos(progress_within_cycle / cycle_length * math.pi)
                )  # Cosine annealing

        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        global_step = 0  # Initialize global step for tracking across epochs

        for epoch in range(start_epoch, num_epochs):
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            self.model.train()
            total_loss = 0

            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).float()

                # (batch_size, seq_len)
                outputs = self.model(inputs).squeeze(-1)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update the learning rate scheduler
                scheduler.step()

                total_loss += loss.item()
                global_step += 1  # Increment global step

            avg_loss = total_loss / len(train_loader)
            losses.append(avg_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                print(f"Validation Loss: {val_loss:.4f}")

            if save_model:
                self.save_model(save_path, epoch, losses)

            if predict and tester is not None:
                if (epoch + 1) % predict_every == 0 and epoch > 0:
                    tester()

        if plot_loss:
            plt.plot(losses)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.savefig(os.path.join(save_path, 'loss_curve.png'), dpi=300)
            plt.close()

    def evaluate(self, val_loader: torch.utils.data.DataLoader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).float()

                outputs = self.model(inputs).squeeze(-1)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        return avg_loss

    def test(
        self,
        test_loader: torch.utils.data.DataLoader,
        model_th: float = 0.5,
        cal_matrix: bool = True,
        saved_model_path: str = None,
    ):

        if saved_model_path is not None and self.mode == 'test':
            self.load_model(saved_model_path)

        self.model.eval()
        all_outputs = []
        all_labels = []
        all_inputs = []

        with torch.no_grad():
            # detect whether the test_loader has labels
            for batch in test_loader:
                if len(batch) == 2:
                    inputs, labels = batch
                    labels = labels.to(self.device).float()
                    has_labels = True
                else:
                    inputs = batch
                    has_labels = False

                inputs = inputs.to(self.device)

                outputs = self.model(inputs).squeeze(-1)
                probabilities = torch.sigmoid(outputs)
                predicted = (
                    probabilities > model_th
                ).float()  # model_th is the threshold for binary classification

                all_outputs.extend(predicted.cpu().numpy())
                all_inputs.extend(inputs.cpu().numpy())

                if has_labels:
                    all_labels.extend(labels.cpu().numpy())

        if cal_matrix and has_labels:
            return self.calculate_metrics(all_labels, all_outputs)
        else:
            predicted_labels = np.asarray(all_outputs).flatten()
            input_data = np.asarray(all_inputs).reshape(-1, 4)

            if has_labels:
                true_labels = np.asarray(all_labels).flatten()
                return true_labels, predicted_labels, input_data
            else:
                return predicted_labels, input_data

    def calculate_metrics(self, true_labels: list, predicted_labels: list):
        true_labels = np.asarray(true_labels).flatten()
        predicted_labels = np.asarray(predicted_labels).flatten()
        print(f"Unique predicted labels: {np.unique(predicted_labels)}")

        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels, average='binary')
        f1 = f1_score(true_labels, predicted_labels, average='binary')
        accuracy = accuracy_score(true_labels, predicted_labels)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return accuracy, precision, recall, f1

    def _plot_results(
        self,
        data,
        dataset_type,
        segment_idx: int,
        segment_size,
        spike_train,
        peak_positions,
        sampling_frequency,
        model_save_path,
        fig_name: str = None,
    ):

        trace_seg = data[
            segment_idx * segment_size : (segment_idx + 1) * segment_size, 0
        ]
        trace_seg = trace_seg[:60000]
        detected_spike_train = peak_positions[peak_positions < 60000].astype(int)
        gt_spike_train = spike_train[np.argwhere(spike_train).flatten()].astype(int)
        gt_spike_train = gt_spike_train[gt_spike_train < 60000]
        time_window = 0.003 if dataset_type in ['bbp', 'bbp_multi'] else 0.001
        _, _, _, detected_matched_boolean, gt_matched_boolean = calculate_n_match(
            detected_spike_train / sampling_frequency,
            gt_spike_train / sampling_frequency,
            time_window,
            return_match_boolean=True,
        )

        positive_detected_spike_train = detected_spike_train[detected_matched_boolean]
        np_detected_spike_train = detected_spike_train[~detected_matched_boolean]
        fn_gt_spike_train = gt_spike_train[~gt_matched_boolean]

        plt.figure(figsize=(20, 6))
        plt.plot(np.arange(len(trace_seg)), trace_seg, c='gray', alpha=0.5)
        plt.scatter(
            positive_detected_spike_train,
            trace_seg[positive_detected_spike_train],
            c='blue',
            s=35,
            marker='x',
            label='True Positive Detected',
        )
        plt.scatter(
            np_detected_spike_train,
            trace_seg[np_detected_spike_train],
            c='red',
            s=35,
            marker='x',
            label='False Positive Detected',
        )
        plt.scatter(
            fn_gt_spike_train,
            trace_seg[fn_gt_spike_train],
            c='k',
            s=20,
            label='Missed Ground Truth',
        )
        plt.legend()

        figure_folder = os.path.join(model_save_path, 'figures')
        if not os.path.exists(figure_folder):
            os.makedirs(figure_folder)

        if fig_name is not None:
            plt.savefig(
                os.path.join(
                    figure_folder, f'{fig_name}_segment{segment_idx}_detection.png'
                )
            )
        else:
            plt.savefig(
                os.path.join(
                    figure_folder,
                    f'{self.detect_method}_segment{segment_idx}_detection.png',
                )
            )
        plt.close()

    def _slice_detect(
        self,
        data: np.ndarray,
        segment_idx: int,
        detect_chunk_size: int = 6000,
        detect_method: str = 'model',
        model_save_path: str = None,
        model_th: float = 0.5,
        threshold: float = 6.0,
    ):

        # Extract the relevant segment
        X = torch.tensor(data.T, dtype=torch.float32).to(self.device)
        X = X[:, segment_idx * detect_chunk_size: (segment_idx + 1) * detect_chunk_size]
        X_cpu = X.cpu().numpy().T

        # Perform spike detection
        if detect_method == 'model':
            test_dataset = LFPDataset_test(lfp_input=X_cpu, input_point_len=2000)
            test_loader = self.load_data(test_dataset, batch_size=512, shuffle=False)
            predicted_labels, input_data = self.test(
                test_loader,
                cal_matrix=False,
                saved_model_path=model_save_path,
                model_th=model_th,
            )

            max_channel = np.argmax(np.abs(input_data).sum(axis=0))
            channel_data = input_data[:, max_channel]
            merged_labels = merge_predicted_labels(predicted_labels, channel_data)
            peak_positions = np.argwhere(merged_labels == 1).flatten()
            
        elif detect_method == 'threshold':
            peak_positions = detect_spikes(
                X,
                nt=61,
                Th_single_ch=threshold,
                loc_range=5,
                long_range=10,
                merge_window=15,
                #device=torch.device("cuda"),
                device=self.device,
            )

        global_peak_positions = peak_positions + segment_idx * detect_chunk_size
        return global_peak_positions

    def detect(
        self,
        data: np.ndarray,
        units_spike_train: np.ndarray = None,
        sampling_frequency: int = 30000,
        detect_chunk_size: int = 60000,
        detect_method: str = 'model',
        model_save_path: str = None,
        model_th: float = 0.5,
        threshold: float = 6.0,
        only_detect: bool = False,
        return_true_snippets: bool = False,
        return_gt_spike_train: bool = False,
        dataset_type: str = 'nex',
        sorting_chunk_size: int = 600000,
    ):

        total_peak_positions = []
        # Process each segment
        if len(data) <= detect_chunk_size:
            for segment_idx in tqdm(range(len(data) // len(data)), desc="Detecting Segments"): # To keep the same foramt
                segment_peak_positions = self._slice_detect(
                    data=data,
                    segment_idx=0,
                    detect_chunk_size=len(data),  # Use the full data length
                    detect_method=detect_method,
                    model_save_path=model_save_path,
                    model_th=model_th,
                    threshold=threshold,
                )
                total_peak_positions.append(segment_peak_positions)
        else:
            # Process each segment
            for segment_idx in tqdm(range(len(data) // detect_chunk_size), desc="Detecting Segments"):
                segment_peak_positions = self._slice_detect(
                    data=data,
                    segment_idx=segment_idx,
                    detect_chunk_size=detect_chunk_size,
                    detect_method=detect_method,
                    model_save_path=model_save_path,
                    model_th=model_th,
                    threshold=threshold,
                )
                total_peak_positions.append(segment_peak_positions)

        # Flatten results
        total_peak_positions = np.sort(np.concatenate(total_peak_positions))
        if units_spike_train is not None:
            units_spike_train = units_spike_train[units_spike_train[:, 0].argsort()]
            total_gt_spike_train = units_spike_train[:, 0]
            total_gt_unit = units_spike_train[:, 1]
        else:
            total_gt_spike_train = None
            total_gt_unit = None

        if only_detect:
        # Separate detection results into sorting chunks
            sorted_chunks = []
            snippets_chunks = []
            true_snippets_chunks = []
            units_spike_train_chunks = []

            num_chunks = len(data) // sorting_chunk_size

            for i in range(num_chunks):
                start_idx = i * sorting_chunk_size
                end_idx = (i + 1) * sorting_chunk_size

                # Chunk peak positions and make them relative to the chunk
                chunk_peak_positions = total_peak_positions[
                    (total_peak_positions >= start_idx) & (total_peak_positions < end_idx)
                ]
                relative_peak_positions = chunk_peak_positions - start_idx
                sorted_chunks.append(relative_peak_positions)

                # Chunk snippets
                snippets = extract_snippets(
                    data.T, nt=61, positions=chunk_peak_positions, cuts=[1, 2]
                )
                snippets_chunks.append(snippets)
                
                # Chunk spike train
                if total_gt_spike_train is not None:
                    spike_train_in_chunk = total_gt_spike_train[
                        (total_gt_spike_train >= start_idx) & (total_gt_spike_train < end_idx)
                    ]
                    relative_spike_train_in_chunk = spike_train_in_chunk - start_idx

                    gt_unit_chunks = total_gt_unit[
                        (total_gt_spike_train >= start_idx) & (total_gt_spike_train < end_idx)
                    ]

                    #print(f"Chunk {i}: detected {len(np.unique(chunk_peak_positions))} spikes")

                    us_train = np.vstack((relative_spike_train_in_chunk, gt_unit_chunks)).T
                    units_spike_train_chunks.append(us_train)

                    if return_true_snippets:
                        true_snippets = extract_snippets(
                            data.T,
                            nt=61,
                            positions=spike_train_in_chunk,
                            cuts=[1, 2],
                        )
                        true_snippets_chunks.append(true_snippets)

            if return_true_snippets and return_gt_spike_train:
                return sorted_chunks, snippets_chunks, units_spike_train_chunks, true_snippets_chunks
            elif return_true_snippets:
                return sorted_chunks, snippets_chunks, true_snippets_chunks
            elif return_gt_spike_train:
                return sorted_chunks, snippets_chunks, units_spike_train_chunks
            else:
                return sorted_chunks, snippets_chunks
            
        # Calculate global metrics
        if units_spike_train is not None:
            # time_window = 0.003 if dataset_type in ['bbp', 'bbp_multi'] else 0.001
            time_window = 0.001
            accuracy, recall, precision = calculate_metrics(
                total_gt_spike_train / sampling_frequency,
                total_peak_positions / sampling_frequency,
                time_window,
            )
            print(
                f'Global Detection - Accuracy: {accuracy:.2f}, Recall: {recall:.2f}, Precision: {precision:.2f}'
            )

        return accuracy, recall, precision

    def predict(
        self,
        data: np.ndarray = None,
        units_spike_train: np.ndarray = None,
        dataset_type: str = 'nex',
        sampling_frequency: int = 30000,
        detect_chunk_size: int = 6000,
        detect_method: str = 'model',
        model_save_path: str = None,
        model_th: float = 0.5,
        threshold: float = 6.0,
        only_test: bool = False,
        save_csv: bool = False,
        save_name: str = None,
    ):

        accuracy, recall, precision = self.detect(
            data=data,
            units_spike_train=units_spike_train,
            sampling_frequency=sampling_frequency,
            detect_chunk_size=detect_chunk_size,
            detect_method=detect_method,
            model_save_path=model_save_path,
            model_th=model_th,
            threshold=threshold,
            only_detect=False,
            return_true_snippets=False,
            dataset_type=dataset_type,
        )

        if save_csv:
            if only_test:
                result_folder = os.path.join(self.root_path, 'results')
            else:
                result_folder = os.path.join(model_save_path, 'results')
                
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)

            result_data = pd.DataFrame(
                {'accuracy': [accuracy], 'recall': [recall], 'precision': [precision]}
            )

            save_path = os.path.join(
                result_folder,
                (
                    f'{save_name}_detection_results.csv'
                    if save_name
                    else 'detection_results.csv'
                ),
            )
            if os.path.exists(save_path):
                result_data.to_csv(save_path, mode='a', header=False, index=False)
            else:
                result_data.to_csv(save_path, index=False)

        return accuracy, recall, precision

    def save_model(
        self, save_path: str = 'saved_models', epoch: int = 0, losses: list = []
    ):

        checkpoint_path = os.path.join(save_path, 'checkpoint.pth')
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'losses': losses,
            },
            checkpoint_path,
        )
