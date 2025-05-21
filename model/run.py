import os
from task.task_manager import parse_args, load_config, save_config, create_experiment, run_experiment

if __name__ == '__main__':
    args = parse_args()

    root_path = os.getenv('AMLT_OUTPUT_DIR', './')
    save_folder = os.path.join(root_path, 'saved_models')

    save_config(vars(args), save_folder, 'args.yaml')
    
    yaml_file = 'data_factory/data_info.yaml'
    config = load_config(yaml_file)
    save_config(config, save_folder, 'data_info.yaml')

    loo_data_info = config['loo_data_info']
    data_info = config['data_info']

    experiment = create_experiment(args, root_path, data_info, loo_data_info)
    run_experiment(experiment, args)