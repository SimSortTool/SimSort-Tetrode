import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
import umap
from types import SimpleNamespace

from torch.utils.data import Dataset, DataLoader
from benchmark.ceed.model_simclr import ModelSimCLR
from benchmark.ceed.utils.utils import get_torch_reps
from benchmark.ceed.data_aug.wf_data_augs import Crop
from benchmark.ceed.data_aug.contrastive_learning_dataset import WFDataset_lab
from data_factory.data_loader import WFDataset
from model import GRU
from utils.tool import reconstruct


# Helper Functions
def load_model_CEED(ckpt_dir, device, num_extra_chans):
    model_CEED = ModelSimCLR(
        base_model="fc_encoder",
        out_dim=5,
        proj_dim=5,
        fc_depth=2,
        expand_dim=False,
        multichan=True,
        input_size=(2 * num_extra_chans + 1) * 121,
        old_ckpt=False,
    )
    ckpt = torch.load(os.path.join(ckpt_dir, "checkpoint.pth"), map_location="cpu")
    model_CEED.load_state_dict(ckpt["state_dict"])
    model_CEED.to(device)
    return model_CEED


def load_gru_model(model_save_folder, device):
    checkpoint = torch.load(os.path.join(model_save_folder, 'checkpoint.pth'), weights_only=True)
    model = GRU(input_size=4, hidden_size=512, num_layers=1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def prepare_data(seed, data_dir, num_units, num_waveforms, num_extra_chans):
    np.random.seed(seed)
    test_data = np.load(os.path.join(data_dir, 'spikes_test.npy'))
    test_labels = np.load(os.path.join(data_dir, 'labels_test.npy'))

    selected_units = np.random.choice(np.unique(test_labels), num_units, replace=False)
    cutted_test_data = test_data[:, :, 20:70]
    max_channel = 11 // 2
    surround_channels = [2, 2]
    cutted_test_data = cutted_test_data[:, (max_channel-surround_channels[0]):(max_channel+surround_channels[1]), :]

    selected_spikes = []
    selected_labels = []
    for unit in selected_units:
        idx = np.where(test_labels == unit)[0]
        idx = np.random.choice(idx, num_waveforms, replace=False)
        selected_spikes.append(cutted_test_data[idx])
        selected_labels.append(test_labels[idx])

    selected_spikes = np.concatenate(selected_spikes, axis=0)
    selected_labels = np.concatenate(selected_labels, axis=0)
    return selected_spikes, selected_labels, selected_units


def calculate_ari(features, labels_test, num_units, num_trials=20):
    ari_list = []
    for _ in range(num_trials):
        gmm = GaussianMixture(n_components=num_units)
        predict_labels = gmm.fit_predict(features)
        ARI = adjusted_rand_score(labels_test, predict_labels)
        ari_list.append(ARI)
    return np.mean(ari_list), np.std(ari_list)


def run_umap(features):
    umap_reducer = umap.UMAP()
    return umap_reducer.fit_transform(features)


def compute_features(model, test_loader, device):
    model.eval()
    sample_num = len(test_loader.dataset)
    features = np.zeros((sample_num, model.hidden_size), dtype=np.float32)

    start_idx = 0
    with torch.no_grad():
        for batch_idx, (anchor, _, _) in enumerate(test_loader):
            anchor = anchor.to(device, non_blocking=True)
            anchor = reconstruct(anchor, k=5)
            anchor_output = model(anchor).cpu().numpy()

            batch_size = anchor.size(0)
            features[start_idx:start_idx + batch_size] = anchor_output
            start_idx += batch_size

    return features


# Main Execution Logic
def run_experiment(seed, data_dir, num_units, num_waveforms, num_extra_chans, model_CEED, model, device, batch_size):
    # Prepare Data
    selected_spikes, selected_labels, selected_units = prepare_data(seed, data_dir, num_units, num_waveforms, num_extra_chans)
    
    # Prepare Dataloaders
    test_dataset = WFDataset_lab(
        data_dir,
        split="test",
        multi_chan=True, 
        transform=Crop(prob=0.0, num_extra_chans=num_extra_chans, ignore_chan_num=True),
        n_units=num_units,
        units_list=selected_units
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    # CEED Model Evaluation
    args = SimpleNamespace(ddp=False, rank=0, multi_chan=True, use_gpt=False, num_extra_chans=num_extra_chans, arch="fc_encoder")
    reps_test, labels_test = get_torch_reps(model_CEED, test_loader, device, args)
    ari_mean, ari_std = calculate_ari(reps_test, labels_test, num_units)
    
    # CEED with UMAP
    reduced_features = run_umap(reps_test)
    ari_umap_mean, ari_umap_std = calculate_ari(reduced_features, labels_test, num_units)

    # SimSort (GRU Model Evaluation)
    test_set = WFDataset(selected_spikes, selected_labels, preprocess=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    features = compute_features(model, test_loader, device)

    # SimSort with UMAP
    reduced_features_simsort = run_umap(features)
    ari_simsort_mean, ari_simsort_std = calculate_ari(reduced_features_simsort, labels_test, num_units)

    return [
        ('CEED', ari_mean, ari_std, seed),
        ('CEED_UMAP', ari_umap_mean, ari_umap_std, seed),
        ('SimSort', ari_simsort_mean, ari_simsort_std, seed)
    ]


def main():
    # Setup paths and parameters
    container = '/mnt/storage'
    base_dir = os.path.join(container, 'backup/IBL_data_CEED')
    ckpt_dir = '/benchmark/ceed/400neur_11c_200s_new'
    model_save_folder = '/simsort_pretrained/extractor_bbp_L1-L5-8192/saved_models'

    data_dir = os.path.join(base_dir, '1a276285-8b0e-4cc9-9f0a-a3a002978724', 'ds')
    
    # Initialize Parameters
    num_units = 10
    num_waveforms = 100
    num_extra_chans = 5  # 11 channels total
    batch_size = 128
    
    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_CEED = load_model_CEED(ckpt_dir, device, num_extra_chans)
    model = load_gru_model(model_save_folder, device)

    # Initialize DataFrame for results
    df = pd.DataFrame(columns=['method', 'mean_ARI', 'std_ARI', 'seed'])
    
    # Run experiment for each seed
    for seed in tqdm(range(100), desc='Running Experiments'):
        results = run_experiment(seed, data_dir, num_units, num_waveforms, num_extra_chans, model_CEED, model, device, batch_size)
        for result in results:
            df.loc[len(df)] = result
    
    # Save results to CSV
    df.to_csv('IBL_identification_compare_ARI.csv', index=False)


# Entry point
if __name__ == "__main__":
    main()
