import numpy as np


class DataInput:
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration
        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size, len(self.data))]
        self.i += 1
        u, hist, hist_cross, i = [], [], [], []
        for t in ts:
            u.append(t[0])
            hist.append(t[1])
            hist_cross.append(t[2])
            i.append(t[3])
        return (u, hist, hist_cross, i)


def compute_metric1(sess, model, testset1, batch_size, data_dict_neg1, K=(1, 2, 3, 4, 5, 10, 15)):
    """
    :param sess:
    :param model:
    :param testset1:
    :param batch_size:
    :param data_dict_neg1:
    :param K: default:(1, 2, 3, 4, 5, 10, 15)
    :return: ndcg, ht, pre, recall, f1
    """
    ndcg = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 10: 0.0, 15: 0.0}
    ht = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 10: 0.0, 15: 0.0}
    pre = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 10: 0.0, 15: 0.0}
    recall = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 10: 0.0, 15: 0.0}
    f1 = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 10: 0.0, 15: 0.0}
    for uij_1 in DataInput(testset1, batch_size):
        score1 = model.test_ndcg1(sess, uij_1, data_dict_neg1)
        score1 = -np.array(score1)
        score1_rank = score1.argsort(axis=0).argsort(axis=0)
        score1_rank = score1_rank[0]
        for score_index in range(len(score1_rank)):
            if uij_1[0][score_index] != 0:
                score = score1_rank[score_index]
                for k in K:
                    if score < k:
                        ndcg[k] += 1 / np.log2(score + 2)
                        ht[k] += 1
                        pre[k] += 1.0 / k
                        recall[k] += 1
                        f1[k] += 2.0 / (k + 1.0)  # 2*(1.0/k*1.0)/(1.0/k+1.0)
    user_num = len(testset1)
    for k in K:
        ndcg[k] /= user_num
        ht[k] /= user_num
        pre[k] /= user_num
        recall[k] /= user_num
        f1[k] /= user_num
    return ndcg, ht, pre, recall, f1
