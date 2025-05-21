import os
import yaml
import argparse
from task.exp_detection import DetectionExperiment
from task.exp_extraction import ExtractionExperiment
from task.exp_sorting import SortingExperiment
from task.exp_spikeinterface import SiCompareExperiment

def parse_args():
    parser = argparse.ArgumentParser(description='Spike detection training script')
    root_path = os.getenv('AMLT_OUTPUT_DIR', './')
    default_save_folder = os.path.join(root_path, 'saved_models')

    # Experiment parameters
    parser.add_argument('--task_name', type=str, default='detection', help='Task name for the experiment')
    parser.add_argument('--train_dataset_type', type=str, default='bbp', help='Dataset type for training')
    parser.add_argument('--test_dataset_type', type=str, default='hybrid', help='Dataset type for testing')
    parser.add_argument('--loo_exp', type=int, default=0, help='Leave-one-out experiment')
    parser.add_argument('--exclude_index', type=int, default=0, help='Index to exclude for the leave-one-out experiment')
    parser.add_argument('--only_test', type=int, default=0, help='Only test the model')
    parser.add_argument('--dmodel_save_path', type=str, default=default_save_folder, help='Path to load the detection model for testing')
    parser.add_argument('--emodel_save_path', type=str, default=default_save_folder, help='Path to load the extraction model for testing')
    parser.add_argument('--use_true_snippets', type=int, default=0, help='Use true snippets for sorting')
    parser.add_argument('--use_true_clusters', type=int, default=0, help='Use true clusters for sorting')
    parser.add_argument('--si_sorter', type=str, default='kilosort', help='Sorting method for spikeinterface') # 'mountainsort4', 'kilosort2', 'kilosort'

    # Training parameters
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint')

    # Model parameters for detection
    parser.add_argument('--dmodel_arch', type=str, default='GRU', help='Model architecture for detection')
    parser.add_argument('--input_size', type=int, default=4, help='Input size of the model')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size of the GRU model')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of GRU layers')
    parser.add_argument('--output_size', type=int, default=1, help='Output size of the model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for the optimizer')
    parser.add_argument('--pos_weight', type=float, default=1.0, help='Positive class weight for handling class imbalance')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for the model')
    parser.add_argument('--nhead', type=int, default=4, help='Number of heads for the transformer model')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for data loaders')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--detect_chunk_size', type=int, default=600000, help='Chunk size for detection')
    parser.add_argument('--lfp_input_points', type=int, default=2000, help='Number of seqence points for LFP input')

    # Model parameters for extraction
    parser.add_argument('--emodel_arch', type=str, default='GRU', help='Model architecture for extraction')
    parser.add_argument('--e_num_layers', type=int, default=1, help='Number of GRU layers')
    parser.add_argument('--e_input_size', type=int, default=4, help='Input size of the model')
    parser.add_argument('--e_learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--e_batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--e_hidden_size', type=int, default=512, help='hidden size')
    parser.add_argument('--e_num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--e_dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--e_nhead', type=int, default=4, help='number of heads')

    # Data augmentation parameters for detection
    parser.add_argument('--amplitude_scale_range', type=str, default='0.5,1.5', help='Range for random amplitude scaling')
    parser.add_argument('--noise_level', type=float, default=0.6, help='Noise level for adding noise augmentation')
    parser.add_argument('--time_jitter_range', type=int, default=5, help='Shift range for time jitter augmentation')
    parser.add_argument('--transform_prob', type=float, default=0.5, help='transform probability')
    parser.add_argument('--probabilities', type=str, default='1,1,1', help='probabilities of each transform')
    parser.add_argument('--model_threshold', type=float, default=0.5, help='Threshold for the model output')
    parser.add_argument('--predict_cycle', type=int, default=20, help='Predict every n epochs')

    # Data augmentation parameters for extraction
    parser.add_argument('--e_noise_level', type=float, default=4., help='noise level')
    parser.add_argument('--e_scale_range', type=str, default='0,1', help='scale range')
    parser.add_argument('--e_transform_prob', type=float, default=0.5, help='transform probability')
    parser.add_argument('--e_probabilities', type=str, default='1,0,1,1', help='probabilities of each transform')
    parser.add_argument('--e_max_channels', type=int, default=4, help='maximum number of channels to apply the transforms')
    parser.add_argument('--e_shift_max', type=int, default=50, help='max shift for time jitter')
    parser.add_argument('--e_strech_range', type=float, default=0.5, help='stretch range')

    # Reinforcement learning parameters
    parser.add_argument('--e_batch_class', type=int, default=10, help='number of classes per batch')
    parser.add_argument('--e_rl', type=int, default=0, help='Use reinforcement learning for training')
    parser.add_argument('--e_rl_batches', type=int, default=5, help='number of batches for RL per epoch')
    parser.add_argument('--e_rl_weight', type=float, default=0.1, help='weight for RL loss')

    # Sorting parameters
    parser.add_argument('--down_dim_method', type=str, default='PCA', help='Dimensionality reduction method')
    parser.add_argument('--cluster_method', type=str, default='DBSCAN', help='Clustering method')
    parser.add_argument('--eps', type=float, default=0.15, help='DBSCAN eps parameter')
    parser.add_argument('--min_samples', type=int, default=20, help='DBSCAN min_samples parameter')
    parser.add_argument('--quantile', type=float, default=0.05, help='Quantile for mean shift bandwidth')
    parser.add_argument('--detect_method', type=str, default='model', help='Detection method')
    parser.add_argument('--threshold', type=float, default=6., help='Detection threshold')
    parser.add_argument('--sorting_chunk_size', type=int, default=600000, help='Recording chunk size for sorting')
    parser.add_argument('--tsne_seed', type=int, default=42, help='Random seed for t-SNE')

    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def save_config(data, save_path, filename):
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, filename), 'w') as file:
        yaml.dump(data, file)

def create_experiment(args, root_path, data_info, loo_data_info):
    if args.only_test == 1:
        e_parent_path = os.path.dirname(args.emodel_save_path)
        config_path = os.path.join(e_parent_path, 'config.yaml')
        
        if os.path.exists(config_path):
            print(f'Loading extraction model config from {config_path}')
            additional_config = load_config(config_path)
            args.e_input_size = additional_config.get('e_input_size', args.e_input_size)
            args.e_hidden_size = additional_config.get('e_hidden_size', args.e_hidden_size)
            args.e_num_layers = additional_config.get('e_num_layers', args.e_num_layers)

        d_parent_path = os.path.dirname(args.dmodel_save_path)
        d_config_path = os.path.join(d_parent_path, 'detection_config.yaml')
        
        if os.path.exists(d_config_path):
            print(f'Loading detection model config from {d_config_path}')
            detection_config = load_config(d_config_path)
            if args.dmodel_arch == 'gru' or args.dmodel_arch == 'GRU':
                args.hidden_size = detection_config.get('hidden_size')
                args.num_layers = detection_config.get('num_layers')
                args.weight_decay = detection_config.get('weight_decay')
            if args.dmodel_arch == 'transformer' or args.dmodel_arch == 'Transformer':
                args.num_layers = detection_config.get('num_layers')
                args.hidden_size = detection_config.get('hidden_size')
                args.dropout = detection_config.get('dropout')
                args.nhead = detection_config.get('nhead')
                args.weight_decay = detection_config.get('weight_decay')

    if args.task_name == 'detection':
        return DetectionExperiment(root_path=root_path, 
                                   args=args, 
                                   data_info=data_info, 
                                   loo_data_info=loo_data_info)

    elif args.task_name == 'extraction':
        return ExtractionExperiment(root_path=root_path, 
                                    args=args, 
                                    data_info=data_info, 
                                    loo_data_info=loo_data_info)

    elif args.task_name == 'sorting':
        return SortingExperiment(root_path=root_path, 
                                 args=args, data_info=data_info,
                                 emodel_save_path=args.emodel_save_path,
                                 dmodel_save_path=args.dmodel_save_path,
                                 use_true_snippets=args.use_true_snippets,
                                 use_true_clusters=args.use_true_clusters,
                                 sorting_chunk_size=args.sorting_chunk_size,
                                 )

    elif args.task_name == 'si':
        return SiCompareExperiment(root_path=root_path, 
                                   args=args, 
                                   data_info=data_info, 
                                   test_dataset_type=args.test_dataset_type,
                                   sorting_method=args.si_sorter)
    
    else:
        raise ValueError(f"Unknown task name: {args.task_name}")

def run_experiment(experiment, args):
    if args.task_name == 'detection':
        if (args.only_test == 1) or (args.detect_method == 'threshold'):
            experiment.test(args.dmodel_save_path)
        else:
            experiment.train()

    elif args.task_name == 'extraction':
        if args.only_test == 1:
            experiment.test(args.emodel_save_path)
        else:
            experiment.train()

    elif args.task_name == 'sorting':

        if args.only_test == 1:
            experiment.test()
        else:
            experiment.train()

    elif args.task_name == 'si':
        experiment.test()