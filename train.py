import sys, time, random
import pandas as pd
import numpy as np
import tensorflow as tf
from model import Model
from utils import DataInput
from utils import compute_metric1
from collections import defaultdict
import argparse

tf.reset_default_graph()

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--memory_window', type=int, default=10)
parser.add_argument('--adam', type=str, default='true')
parser.add_argument('--inner', type=str, default='true')
parser.add_argument('--de', type=str, default='true')
parser.add_argument('--da', type=str, default='true')
parser.add_argument('--clip', type=float, default=-1.0)
parser.add_argument('--wei', type=float, default=1.0)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--all_metric_name', type=str)
parser.add_argument('--train_data1', type=str, default='dataset4dasl/fix_CDs_and_Vinyl_train.csv')
parser.add_argument('--train_data2', type=str, default='dataset4dasl/fix_Books_train.csv')
parser.add_argument('--test_data1', type=str, default='processed_data_all/CDs_and_Vinyl_valid.csv')
parser.add_argument('--test_data2', type=str, default='processed_data_all/Books_valid.csv')
parser.add_argument('--neg_data1', type=str, default='processed_data_all/CDs_and_Vinyl_negative.csv')
parser.add_argument('--neg_data2', type=str, default='processed_data_all/Books_negative.csv')
param = parser.parse_args()
param.adam = True if param.adam == 'true' else False
param.inner = True if param.inner == 'true' else False
param.de = True if param.de == 'true' else False
param.da = True if param.da == 'true' else False
print('\n'.join([str(k) + ': ' + str(v) for k, v in vars(param).items()]))
batch_size = param.batch_size
lr = param.lr
epoch = param.epoch
w = param.memory_window

train_data1 = pd.read_csv(param.train_data1,
                          header=None, names=['uid', 'iid', 'time'],
                          dtype={'uid': 'int32', 'iid': 'int32', 'time': 'int64'})
train_data2 = pd.read_csv(param.train_data2,
                          header=None, names=['uid', 'iid', 'time'],
                          dtype={'uid': 'int32', 'iid': 'int32', 'time': 'int64'})
test_data1 = pd.read_csv(param.test_data1,
                         header=None, names=['uid', 'iid', 'time'],
                         dtype={'uid': 'int32', 'iid': 'int32', 'time': 'int64'})
test_data2 = pd.read_csv(param.test_data2,
                         header=None, names=['uid', 'iid', 'time'],
                         dtype={'uid': 'int32', 'iid': 'int32', 'time': 'int64'})
user_count = train_data1.uid.max() + 1  # because start of 1
item_count = max(train_data1.iid.max(), train_data2.iid.max()) + 1  # also because of 1
data1_item = train_data1.iid.unique()
data1_item.sort()
data1_item = data1_item[data1_item.nonzero()]
data1_itemset = set(data1_item)
data2_item = train_data2.iid.unique()
data2_item.sort()
data2_item = data2_item[data2_item.nonzero()]
data2_itemset = set(data2_item)

data_dict_neg1 = defaultdict(list)
data_dict_neg1[0] = [0] * 100
with open(param.neg_data1, 'r') as f:
    for line in f:
        l = line.rstrip().split(',')
        u = int(l[0])
        for j in range(1, 101):
            i = int(l[j])
            data_dict_neg1[u].append(i)

data_dict_neg2 = defaultdict(list)
data_dict_neg2[0] = [0] * 100
with open(param.neg_data2, 'r') as f:
    for line in f:
        l = line.rstrip().split(',')
        u = int(l[0])
        for j in range(1, 101):
            i = int(l[j])
            data_dict_neg2[u].append(i)

trainset1, testset1, trainset2, testset2 = [], [], [], []
user_hist1, user_hist2 = {}, {}
user_nohist1, user_nohist2 = {}, {}
userid = train_data1.uid.unique()
for user in userid:
    train_user1 = train_data1.loc[train_data1['uid'] == user]
    train_user1 = train_user1.sort_values(['time'])
    hist1 = train_user1.iid.values
    user_hist1[user] = hist1[hist1.nonzero()]
    user_nohist1[user] = np.setdiff1d(data1_item, user_hist1[user])
    length1 = len(train_user1)
    assert length1 == 100
    train_user1.index = range(length1)
    train_user2 = train_data2.loc[train_data2['uid'] == user]
    train_user2 = train_user2.sort_values(['time'])
    hist2 = train_user2.iid.values
    user_hist2[user] = hist2[hist2.nonzero()]
    user_nohist2[user] = np.setdiff1d(data2_item, user_hist2[user])
    length2 = len(train_user2)
    assert length2 == 100
    train_user2.index = range(length2)
    for i in range(100 - w):
        if train_user1.iloc[i + w, 1] != 0:
            trainset1.append((train_user1.iloc[i + w, 0], list(train_user1.iloc[i:i + w, 1]),
                              list(train_user2.iloc[i:i + w, 1]), train_user1.iloc[i + w, 1]))
        if train_user2.iloc[i + w, 1] != 0:
            trainset2.append((train_user2.iloc[i + w, 0], list(train_user2.iloc[i:i + w, 1]),
                              list(train_user1.iloc[i:i + w, 1]), train_user2.iloc[i + w, 1]))
    test_user1 = test_data1.loc[test_data1['uid'] == user]
    length1 = len(test_user1)
    assert length1 <= 1
    test_user1.index = range(length1)
    test_user2 = test_data2.loc[test_data2['uid'] == user]
    length2 = len(test_user2)
    assert length2 <= 1
    test_user2.index = range(length2)
    if length1 >= 1:
        if test_user1.iloc[0, 1] in data1_itemset:
            testset1.append((test_user1.iloc[0, 0], list(train_user1.iloc[-w:, 1]),
                             list(train_user2.iloc[-w:, 1]), test_user1.iloc[0, 1]))
        else:
            testset1.append((0, [0] * w, [0] * w, 0))
    if length2 >= 1:
        if test_user2.iloc[0, 1] in data2_itemset:
            testset2.append((test_user2.iloc[0, 0], list(train_user2.iloc[-w:, 1]),
                             list(train_user1.iloc[-w:, 1]), test_user2.iloc[0, 1]))
        else:
            testset2.append((0, [0] * w, [0] * w, 0))
random.shuffle(trainset1)
random.shuffle(testset1)
random.shuffle(trainset2)
random.shuffle(testset2)
print("trainset1 len: {}".format(len(trainset1)))
print("testset1 len: {}".format(len(testset1)))
print("trainset2 len: {}".format(len(trainset2)))
print("testset2 len: {}".format(len(testset2)))
sys.stdout.flush()

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    model = Model(user_count, item_count,
                  param.hidden_size,
                  param.memory_window,
                  param.inner, param.de, param.da,
                  param.adam, param.clip, param.wei, param.dropout)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    test_ndcg_1, test_hr_1, test_pre_1, test_recall_1, test_f1_1 = compute_metric1(sess, model, testset1, 99999999,
                                                                                   data_dict_neg1)
    print('Init \t A NDCG@10: %.4F \t A HT@10: %.4f' % (test_ndcg_1[10], test_hr_1[10]))
    sys.stdout.flush()
    start_time = time.time()

    best_ndcg = 0
    best_epoch = 0
    all_metric = {}
    for epo in range(1, epoch + 1):
        loss_sum = 0.0
        for uij in DataInput(trainset1, batch_size):
            neg_list = []
            user_list = uij[0]
            for user in user_list:
                neg_user = np.random.choice(user_nohist1[user])
                neg_list.append(neg_user)
            loss = model.train_1(sess, uij, neg_list, lr)
            loss_sum += loss
        for uij in DataInput(trainset2, batch_size):
            neg_list = []
            user_list = uij[0]
            for user in user_list:
                neg_user = np.random.choice(user_nohist2[user])
                neg_list.append(neg_user)
            loss = model.train_2(sess, uij, neg_list, lr)
            loss_sum += loss
        model.train_orth(sess, lr)
        if epo < 10:
            test_ndcg_1, test_hr_1, test_pre_1, test_recall_1, test_f1_1 = compute_metric1(sess, model, testset1,
                                                                                           99999999, data_dict_neg1)
            print('Epoch %d \t A NDCG@10: %.4F \t A HT@10: %.4f' % (epo, test_ndcg_1[10], test_hr_1[10]))
            print('Epoch %d \tTrain_loss: %.4f' % (epo, loss_sum))
            print('Epoch %d DONE\tCost time: %.2f' % (epo, time.time() - start_time))
            sys.stdout.flush()
        if epo % 50 == 0:
            test_ndcg_1, test_hr_1, test_pre_1, test_recall_1, test_f1_1 = compute_metric1(sess, model, testset1,
                                                                                           99999999, data_dict_neg1)
            print('Epoch %d \t A NDCG@10: %.4F \t A HT@10: %.4f' % (epo, test_ndcg_1[10], test_hr_1[10]))
            print('Epoch %d \tTrain_loss: %.4f' % (epo, loss_sum))
            print('Epoch %d DONE\tCost time: %.2f' % (epo, time.time() - start_time))
            sys.stdout.flush()
            all_metric[epo] = {'ndcg': test_ndcg_1, 'hit_rate': test_hr_1, 'test_pre': test_pre_1,
                               'test_recall': test_recall_1, 'test_f1': test_f1_1}
            if np.isnan(loss_sum):
                break
            elif test_ndcg_1[10] > best_ndcg:
                best_ndcg = test_ndcg_1[10]
                best_epoch = epo

        model.global_epoch_step_op.eval()
    print('Epoch %d \t A best NDCG@10: %.4F ' % (best_epoch, best_ndcg))

    import pickle

    with open(param.all_metric_name + ".pkl", 'wb') as f:
        pickle.dump(all_metric, f)
