python -u data_info.py \
    --task_name 'si' \
    --hybrid_file_idx 0 \
    --waveclus_file_idx 0

python -u run.py \
    --task_name 'si' \
    --test_dataset_type 'hybrid' \
    --si_sorter 'kilosort' 