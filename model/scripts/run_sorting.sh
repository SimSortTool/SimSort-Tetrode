python -u data_info.py \
    --task_name 'sorting' \
    --hybrid_file_idx 8

python -u run.py \
    --task_name 'sorting' \
    --emodel_arch 'gru' \
    --dmodel_arch 'transformer' \
    --test_dataset_type 'hybrid' \
    --down_dim_method 'UMAP' \
    --detect_method 'model' \
    --model_threshold 0.97 \
    --cluster_method 'MS' \
    --detect_chunk_size 3000000 \
    --sorting_chunk_size 3000000 \
    --quantile 0.04 \
    --only_test 1 \
    --use_true_snippets 0 \
    --use_true_clusters 0 \
    --emodel_save_path 'simsort_pretrained/extractor_bbp_L1-L5-8192/saved_models' \
    --dmodel_save_path 'simsort_pretrained/detector_bbp_L1-L5-8192/saved_models'