import os
import glob
import yaml
import argparse


def collect_paths(base_path, relative_path):
    full_path = os.path.join(base_path, relative_path)
    paths = glob.glob(full_path)
    return sorted(paths) if paths else []

def parse_arguments():
    parser = argparse.ArgumentParser(description="Spikesorting dataset info script")
    parser.add_argument('--task_name', type=str, default='detection', help='Task name')
    parser.add_argument('--hybrid_file_idx', type=int, default=0, help='Index of hybrid file to use')
    parser.add_argument('--crcns_file_idx', type=int, default=2, help='Index of CRCNS file to use')
    return parser.parse_args()

def safe_get(paths, idx, default=None):
    return paths[idx] if idx < len(paths) else default

def main():
    base_path = './'
    if not os.path.exists(base_path):
        base_path = os.path.abspath('./')

    args = parse_arguments()

    hybrid_static_paths = collect_paths(base_path, 'benchmark/HYBRID_JANELIA/hybrid_static_tetrode/*.json')
    hybrid_drift_paths = collect_paths(base_path, 'benchmark/HYBRID_JANELIA/hybrid_drift_tetrode/*.json')
    crcns_data_paths = collect_paths(base_path, 'benchmark/PAIRED_CRCNS_HC1/paired_crcns/*.json')

    if args.task_name == 'detection' or args.task_name == 'extraction':
        bbp_L1_paths = os.path.join(base_path, 'datasets/6e5_dt0.1_n5-long-L1/datasets')
        bbp_L23_paths = os.path.join(base_path, 'datasets/6e5_dt0.1_n5-long-L23/datasets')
        bbp_L4_paths = os.path.join(base_path, 'datasets/6e5_dt0.1_n5-long-L4/datasets')
        bbp_L5_paths = os.path.join(base_path, 'datasets/6e5_dt0.1_n5-long-L5/datasets')
        bbp_L1_L5_paths = os.path.join(base_path, 'datasets/6e5_dt0.1_n5-long-L1-L5/datasets')
        bbp_L6_paths = os.path.join(base_path, 'datasets/6e5_dt0.1_n5-long-L6/datasets')
        bbp_allen_paths = os.path.join(base_path, 'datasets/6e5_dt0.1_n5_allen-long-1/datasets')

    elif args.task_name == 'sorting' or args.task_name == 'si':
        bbp_L1_paths = os.path.join(base_path, 'datasets/6e5_dt0.1_n5-long-L1-sorting/datasets')
        bbp_L23_paths = os.path.join(base_path, 'datasets/6e5_dt0.1_n5-long-L23-sorting/datasets')
        bbp_L4_paths = os.path.join(base_path, 'datasets/6e5_dt0.1_n5-long-L4-sorting/datasets')
        bbp_L5_paths = os.path.join(base_path, 'datasets/6e5_dt0.1_n5-long-L5-sorting/datasets')
        bbp_L6_paths = os.path.join(base_path, 'datasets/6e5_dt0.1_n5-long-L6-sorting/datasets')
        bbp_L1_L5_paths = []
        bbp_allen_paths = os.path.join(base_path, 'datasets/6e5_dt0.1_n5_allen-long-1-sorting/datasets')


    yaml_content = {
        'loo_data_info': {
            'bbp_L1_path': bbp_L1_paths,
            'bbp_L23_path': bbp_L23_paths,
            'bbp_L4_path': bbp_L4_paths,
            'bbp_L5_path': bbp_L5_paths,
            'bbp_L6_path': bbp_L6_paths,
            'bbp_allen_path': bbp_allen_paths,
            'hybrid_drift_paths': hybrid_drift_paths,
            'hybrid_static_paths': hybrid_static_paths,
        },
        'data_info': {
            'crcns_data_path': safe_get(crcns_data_paths, args.crcns_file_idx, default=None),
            'hybrid_file_path': safe_get(hybrid_static_paths, args.hybrid_file_idx, default=None),
            'bbp_data_path': bbp_L1_L5_paths,
        }
    }

    os.makedirs('data_factory', exist_ok=True)
    with open('data_factory/data_info.yaml', 'w') as file:
        yaml.dump(yaml_content, file, default_flow_style=False)

    print('YAML file created')

if __name__ == '__main__':
    main()