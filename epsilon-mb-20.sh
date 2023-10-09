#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 train.py --train_data1 dataset4dasl/fix_Movies_and_TV_train_and_valid.csv --train_data2 dataset4dasl/noisy_Books_20_train_and_valid.csv --test_data1 processed_data_all/Movies_and_TV_test.csv --test_data2 processed_data_all/Books_test.csv --neg_data1 processed_data_all/Movies_and_TV_negative.csv --neg_data2 processed_data_all/Books_negative.csv --all_metric_name ./epsilon_metric/mb_20_1 >./epsilon_results/mb_20_1.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python3 train.py --train_data1 dataset4dasl/fix_Movies_and_TV_train_and_valid.csv --train_data2 dataset4dasl/noisy_Books_20_train_and_valid.csv --test_data1 processed_data_all/Movies_and_TV_test.csv --test_data2 processed_data_all/Books_test.csv --neg_data1 processed_data_all/Movies_and_TV_negative.csv --neg_data2 processed_data_all/Books_negative.csv --all_metric_name ./epsilon_metric/mb_20_2 >./epsilon_results/mb_20_2.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python3 train.py --train_data1 dataset4dasl/fix_Movies_and_TV_train_and_valid.csv --train_data2 dataset4dasl/noisy_Books_20_train_and_valid.csv --test_data1 processed_data_all/Movies_and_TV_test.csv --test_data2 processed_data_all/Books_test.csv --neg_data1 processed_data_all/Movies_and_TV_negative.csv --neg_data2 processed_data_all/Books_negative.csv --all_metric_name ./epsilon_metric/mb_20_3 >./epsilon_results/mb_20_3.txt 2>&1 &
