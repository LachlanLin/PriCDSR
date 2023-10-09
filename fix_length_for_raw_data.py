import time
import argparse
import numpy as np
import pandas as pd


def fix_length(dataset, max_len):
    user_list = dataset.user_id.unique()
    res_list = list()
    for user in user_list:
        user_seq_pd = dataset.loc[dataset['user_id'] == user]
        user_seq_pd = user_seq_pd.sort_values(['timestamp'])
        user_seq_np = user_seq_pd.item_id.values
        if len(user_seq_np) > max_len:
            user_seq_np = user_seq_np[-max_len:]
        elif len(user_seq_np) < max_len:
            user_seq_np = np.pad(user_seq_np, (max_len - len(user_seq_np), 0), 'constant', constant_values=(0, 0))
        user_id_seq_np = np.full((max_len,), user, dtype=np.int32)
        timestamp_seq_np = np.arange(1, max_len + 1, dtype=np.int64)
        user_dataset = {'user_id': user_id_seq_np, 'item_id': user_seq_np, 'timestamp': timestamp_seq_np}
        user_pd = pd.DataFrame.from_dict(user_dataset)
        res_list.append(user_pd)
    res_pd = pd.concat(res_list)
    res_pd.reset_index(drop=True, inplace=True)
    return res_pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Books')
    param = parser.parse_args()
    dataset = param.dataset
    data_train = pd.read_csv(f'processed_data_all/{dataset}_train.csv',
                             header=None, names=['user_id', 'item_id', 'timestamp'],
                             dtype={'user_id': 'int32', 'item_id': 'int32', 'timestamp': 'int64'})
    data_valid = pd.read_csv(f'processed_data_all/{dataset}_valid.csv',
                             header=None, names=['user_id', 'item_id', 'timestamp'],
                             dtype={'user_id': 'int32', 'item_id': 'int32', 'timestamp': 'int64'})
    data = pd.concat([data_train, data_valid])
    data.reset_index(drop=True, inplace=True)
    start_time = time.time()
    data_noisy = fix_length(data, 101)
    df_valid = data_noisy.groupby(['user_id']).tail(1)
    df_train = data_noisy.drop(df_valid.index, axis='index', inplace=False)
    df_train.to_csv(f'dataset4dasl/fix_{dataset}_train.csv', header=False, index=False)
    df_nouse = data_noisy.groupby(['user_id']).head(1)
    df_train_and_valid = data_noisy.drop(df_nouse.index, axis='index', inplace=False)
    df_train_and_valid.to_csv(f'dataset4dasl/fix_{dataset}_train_and_valid.csv', header=False, index=False)
