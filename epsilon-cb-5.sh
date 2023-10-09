#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python3 train.py --train_data1 dataset4dasl/fix_CDs_and_Vinyl_train_and_valid.csv --train_data2 dataset4dasl/noisy_Books_5_train_and_valid.csv --test_data1 processed_data_all/CDs_and_Vinyl_test.csv --test_data2 processed_data_all/Books_test.csv --neg_data1 processed_data_all/CDs_and_Vinyl_negative.csv --neg_data2 processed_data_all/Books_negative.csv --all_metric_name ./epsilon_metric/cb_5_1 >./epsilon_results/cb_5_1.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python3 train.py --train_data1 dataset4dasl/fix_CDs_and_Vinyl_train_and_valid.csv --train_data2 dataset4dasl/noisy_Books_5_train_and_valid.csv --test_data1 processed_data_all/CDs_and_Vinyl_test.csv --test_data2 processed_data_all/Books_test.csv --neg_data1 processed_data_all/CDs_and_Vinyl_negative.csv --neg_data2 processed_data_all/Books_negative.csv --all_metric_name ./epsilon_metric/cb_5_2 >./epsilon_results/cb_5_2.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python3 train.py --train_data1 dataset4dasl/fix_CDs_and_Vinyl_train_and_valid.csv --train_data2 dataset4dasl/noisy_Books_5_train_and_valid.csv --test_data1 processed_data_all/CDs_and_Vinyl_test.csv --test_data2 processed_data_all/Books_test.csv --neg_data1 processed_data_all/CDs_and_Vinyl_negative.csv --neg_data2 processed_data_all/Books_negative.csv --all_metric_name ./epsilon_metric/cb_5_3 >./epsilon_results/cb_5_3.txt 2>&1 &
