python -u data_info.py \
    --task_name 'extraction' \
    --hybrid_file_idx 0

python -u run.py \
    --resume \
    --task_name 'extraction' \
    --test_dataset_type 'hybrid' \
    --down_dim_method 'UMAP' \
    --only_test 0 \
    --emodel_arch 'gru' \
    --e_num_epochs 10 \
    --e_num_layers 1 \
    --e_batch_size 128 \
    --e_hidden_size 512 \
    --e_learning_rate 0.0001 \
    --e_shift_max 50 \
    --e_noise_level 4 \
    --e_scale_range '0.5,1' \
    --e_transform_prob 0.5 \
    --predict_cycle 1