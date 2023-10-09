#!/bin/bash

# 0.0001, 0.0005, 0.001, 0.005
# 128, 256, 512

lr='0.005'
bz='512'
CUDA_VISIBLE_DEVICES=1 python3 train.py --lr ${lr} --batch_size ${bz} --train_data1 dataset4dasl/fix_Movies_and_TV_train.csv --train_data2 dataset4dasl/fix_CDs_and_Vinyl_train.csv --test_data1 processed_data_all/Movies_and_TV_valid.csv --test_data2 processed_data_all/CDs_and_Vinyl_valid.csv --neg_data1 processed_data_all/Movies_and_TV_negative.csv --neg_data2 processed_data_all/CDs_and_Vinyl_negative.csv --all_metric_name ./valid_metric/mc-lr-${lr}-bz-${bz} >./valid_results/mc-lr-${lr}-bz-${bz}.txt &
CUDA_VISIBLE_DEVICES=1 python3 train.py --lr ${lr} --batch_size ${bz} --de false --da false --train_data1 dataset4dasl/fix_Movies_and_TV_train.csv --train_data2 dataset4dasl/fix_CDs_and_Vinyl_train.csv --test_data1 processed_data_all/Movies_and_TV_valid.csv --test_data2 processed_data_all/CDs_and_Vinyl_valid.csv --neg_data1 processed_data_all/Movies_and_TV_negative.csv --neg_data2 processed_data_all/CDs_and_Vinyl_negative.csv --all_metric_name ./valid_metric/mc-single-lr-${lr}-bz-${bz} >./valid_results/mc-single-lr-${lr}-bz-${bz}.txt &
