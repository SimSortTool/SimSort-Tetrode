# SimSort: A Data-Driven Framework for Spike Sorting by Large-Scale Electrophysiology Simulation


<p align="center">
·
Paper
·
<a href="#">Code</a>
·
<a href="https://simsorttool.github.io/">Webpage</a>
·
</p>

This repository contains the PyTorch implementation of "*SimSort: A Data-Driven Framework for Spike Sorting by Large-Scale Electrophysiology Simulation*".

## Overview
SimSort is a data-driven spike sorting framework. It provides:  

✨ A large-scale electrophysiology simulator for generating biologically realistic extracellular data<br>
✨ Pre-trained models for spike detection and identification<br>
✨ Evaluation on public benchmark datasets<br>
✨ A tool for custom spike sorting on tetrode data

## 🏃Quick Start

### Installation
```bash
# Create and activate conda environment
conda create -n SimSort python=3.10 -y
conda activate SimSort

# Install dependencies
cd SimSort-Tetrode
pip install -r model/requirements.txt

# Install and setup kachery-cloud (required for benchmark datasets)
pip install kachery-cloud
kachery-cloud-init
```

### Using Pre-trained Models
```bash
# Run spike sorting with pre-trained models
bash scripts/run_sorting.sh
```

### Training Models
```bash
# Train detection model
bash scripts/train_detector.sh

# Train identification model
bash scripts/train_extractor.sh
```

### Benchmark Comparison
```bash
# Compare with other spike sorting methods (e.g., Kilosort)
bash scripts/run_sorting_with_si.sh --si_sorter kilosort
```

## Custom Data

For applying SimSort to your own recordings, see `model/SimSort_Tool_Demo.ipynb`.



<!-- ## Part 1: Simulator - Generate Custom Datasets

**Simulator** is designed to generate custom neural signal datasets for use in spike detection, spike identification, and spike sorting tasks.

> **Content Under Development**: The details for the Simulator will be added in a future update.


## Part 2: Model - Train, Evaluate, and Deploy Pre-trained Models

### Overview
The **Model** component includes:
1. **Pre-trained Models**: Ready-to-use models for spike detection and spike identification.
2. **Scripts**: For training and evaluating `detection` and `extraction` models or directly running pre-trained models for spike sorting.
3. **Datasets**: You can use pre-generated datasets from Zenodo or custom datasets to train your models.
4. **Benchmark**: Includes publicly available recordings such as Hybird and WaveClus, with ground-truth.


### Environment Setup

1. Create and activate the environment:
   ```bash
   conda create -n SimSort python=3.10 -y
   conda activate SimSort
   ```
2. Install dependencies:
   ```bash
   pip install -r model/requirements.txt
   ```


### Dataset Preparation

#### 1️⃣ **Pre-generated Datasets (Coming Soon)**
You can download pre-generated datasets from `Zenodo`:
```bash
wget "https://zenodo.org/record/xxxxxx/files/datasets.zip?download=1" -O datasets.zip
unzip datasets.zip -d model/datasets/
```

#### 2️⃣ **Custom Datasets (Optional)**
If you generate your own datasets using the **simulator**, place them in the `model/datasets/` directory.

#### 3️⃣ **Benchmark Datasets with `kachery-cloud`**
Some benchmark datasets (e.g., `HYBRID`) require `kachery-cloud` for access. Follow these steps to configure and use `kachery-cloud`:

1. Install `kachery-cloud`:
   ```bash
   pip install kachery-cloud
   ```
2. Initialize the configuration:
   ```bash
   kachery-cloud-init
   ```
   This will create necessary client keys in the default location: `~/.kachery-cloud/`.

3. Run the relevant script to access benchmark datasets:
   ```bash
   bash scripts/run_sorting.sh --test_dataset_type 'hybrid'
   ```

> **Troubleshooting**:  
> If you encounter `Client keys not found`, re-run `kachery-cloud-init` to regenerate keys, or ensure they exist in `~/.kachery-cloud/`.  
> Ensure you have the correct permissions to access the datasets.

## 📝 Script Usage Guide

The repository includes several pre-defined scripts to simplify the training and evaluation process. Below is a guide on how to use these scripts:

### 1️⃣ **Run Sorting with Pre-trained Models**
To perform spike sorting using the pre-trained models, run the `run_sorting.sh` script:
```bash
bash scripts/run_sorting.sh
```

### 2️⃣ **Train Detection Model**
To train the spike detection model from scratch, use the `train_detector.sh` script:
```bash
bash scripts/train_detector.sh
```

### 3️⃣ **Train Identification Model**
To train the spike identification model, use the `train_extractor.sh` script:
```bash
bash scripts/train_extractor.sh
```

### 4️⃣ **Run Sorting with SpikeInterface**
The `run_sorting_with_si.sh` script integrates **SpikeInterface** to use external spike sorting algorithms (e.g., Kilosort, Kilosort2, MountainSort4, MountainSort5) for comparison.
```bash
bash scripts/run_sorting_with_si.sh --si_sorter kilosort 
```

### 🔧 **Spike Sorting for Custom Data**  
If you want to apply spike sorting to your own data, please refer to the `model/SimSort_Tool_Demo.ipynb` script. -->