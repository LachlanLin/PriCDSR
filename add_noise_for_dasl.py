import pandas as pd
import time
import argparse
import numpy as np
import math


def add_noise_to_dataset(dataset, max_len, epsilon):
    user_list = dataset.user_id.unique()
    item_list = dataset.item_id.unique()
    item_max_id = item_list.max()
    res_list = list()
    for user in user_list:
        user_seq_pd = dataset.loc[dataset['user_id'] == user]
        user_seq_pd = user_seq_pd.sort_values(['timestamp'])
        user_seq_np = user_seq_pd.item_id.values
        if len(user_seq_np) > max_len:
            user_seq_np = user_seq_np[-max_len:]
        elif len(user_seq_np) < max_len:
            user_seq_np = np.pad(user_seq_np, (max_len - len(user_seq_np), 0), 'constant', constant_values=(0, 0))
        prob = np.zeros(item_max_id + 1, dtype=np.float32)
        prob[0] = 1
        prob[item_list] = 1
        items = np.arange(item_max_id + 1, dtype=np.int32)
        for i in range(len(user_seq_np)):
            prob[user_seq_np[i]] = math.exp(epsilon)
            perturbed_item = int(np.random.choice(items, p=prob / prob.sum()))
            prob[user_seq_np[i]] = 1
            if perturbed_item != 0:
                prob[perturbed_item] = 0
            if perturbed_item != user_seq_np[i]:
                ind_tuple = np.nonzero(user_seq_np[i + 1:] == perturbed_item)
                ind_list = ind_tuple[0]
                if perturbed_item != 0 and len(ind_list) != 0:
                    ind = ind_list[0] + i + 1
                    user_seq_np[i], user_seq_np[ind] = user_seq_np[ind], user_seq_np[i]
                else:
                    user_seq_np[i] = perturbed_item
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
    for epsilon in [1, 2, 5, 10, 20, 50]:
        start_time = time.time()
        data_noisy = add_noise_to_dataset(data, 101, epsilon)
        print("epsilon {} cost time: {:.4f}".format(epsilon, time.time() - start_time))
        df_valid = data_noisy.groupby(['user_id']).tail(1)
        df_train = data_noisy.drop(df_valid.index, axis='index', inplace=False)
        df_train.to_csv(f'dataset4dasl/noisy_{dataset}_{epsilon}_train.csv', header=False, index=False)
        df_nouse = data_noisy.groupby(['user_id']).head(1)
        df_train_and_valid = data_noisy.drop(df_nouse.index, axis='index', inplace=False)
        df_train_and_valid.to_csv(f'dataset4dasl/noisy_{dataset}_{epsilon}_train_and_valid.csv', header=False,
                                  index=False)
