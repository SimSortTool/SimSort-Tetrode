python -u data_info.py \
    --task_name 'detection' \
    --hybrid_file_idx 0

python -u run.py \
    --resume \
    --task_name 'detection' \
    --test_dataset_type 'hybrid' \
    --only_test 0 \
    --dmodel_arch 'transformer' \
    --num_epochs 20 \
    --num_layers 1 \
    --batch_size 12 \
    --hidden_size 256 \
    --learning_rate 0.0005 \
    --amplitude_scale_range '0,1' \
    --noise_level 0.9 \
    --transform_prob 0.8 \
    --time_jitter_range 5 \
    --predict_cycle 1