import os

import numpy as np
import lightgbm as lgb
from collections import OrderedDict

from numpy.random import uniform

from utils import Timeit, dump_obj, FLOAT_DTYPE


class LGBMSelGB:

    def __init__(self, n_estimators=100, n_iter_sample=1, p=0.1,
                 max_p=0.2, k_factor=0.98, delta=0.5, delta_pos=10,
                 max_resample=None, method='fixed'):
        self.n_estimators = n_estimators
        self.n_iter_sample = n_iter_sample
        self.p = p
        self.method = method
        self.max_p = max_p
        self.k_factor = k_factor
        self.delta = delta
        self.delta_pos = delta_pos
        self.max_resample = max_resample
        self.resample_count = None
        self.iter = 0
        self.booster = None
        self._eval_result = {}

    def _construct_dataset(self, X, y, group, reference=None):
        return lgb.Dataset(X, label=y, group=group, reference=reference)

    @Timeit('SelGB sample')
    def _sel_sample(self, X, y, group):
        print('[INFO] Selecting new sample.')
        # dtype used to keep track of idx while sorting
        pred_idx_dtype = np.dtype([('pred', FLOAT_DTYPE), ('idx', np.uint64)])
        # get top p% negative examples, based on group
        # also computes new groups (number of docs for each query)
        cum = 0
        group_new = []
        top_p_idx_neg = []
        for query_size in group:
            idx_query = np.array(range(cum, cum + query_size))
            idx_query_pos = np.array([x for x in idx_query if y[x] > 0])
            idx_query_neg = np.array([x for x in idx_query if y[x] == 0])
            if self.method == 'delta':
                preds = self.predict(X[idx_query])
                preds = np.array(list(zip(preds, idx_query)), dtype=pred_idx_dtype)
                preds = np.sort(preds, order='pred')[::-1]
                score = preds[min(self.delta_pos, preds.size - 1)]['pred']
                preds_neg = preds[np.isin(preds['idx'], idx_query_neg)]
                idx_delta = np.logical_and(score - self.delta < preds_neg['pred'],
                                           preds_neg['pred'] < score + self.delta)
                top_p_idx_neg += list(preds_neg[idx_delta]['idx'].astype(int))
                group_new.append(idx_delta.sum() + len(idx_query_pos))
            else:
                if idx_query_neg.size > 0:
                    tmp = X[idx_query_neg]
                    preds_neg = self.predict(tmp)
                    preds_neg = np.array(list(zip(preds_neg, idx_query_neg)), dtype=pred_idx_dtype)
                    preds_neg = np.sort(preds_neg, order='pred')[::-1]
                    self._update_p(preds_neg=preds_neg, idx_pos=idx_query_pos, end=False)
                    if self.max_resample is not None:
                        if type(self.max_resample) is int:
                            # each example can't be in training set more than max_resample times
                            which_examples = self.resample_count[idx_query_neg] < self.max_resample
                        elif type(self.max_resample) is float:
                            # each example can't be in training set more than max_resample times * iter times
                            which_examples = self.resample_count[idx_query_neg] < (self.max_resample * self.iter)
                        else:
                            raise ValueError('max_resample must be either None, int or float')
                        idx_query_neg = idx_query_neg[which_examples]
                        if not np.any(which_examples):
                            # we need at least 1 example otherwise lightgbm goes mad
                            preds_neg = preds_neg[:1]
                        else:
                            preds_neg = preds_neg[which_examples]
                    top_p = int(self.p * len(idx_query_neg))
                    if top_p < 1 and idx_query_neg.size > 0:
                        top_p = 1
                    top_p_idx_neg += list(preds_neg['idx'][:top_p])
                    if self.max_resample is not None:
                        self.resample_count[preds_neg['idx'][:top_p]] += 1
                else:
                    # no negative examples for this query
                    top_p = 0
                group_new.append(top_p + len(idx_query_pos))
            cum += query_size

        self._update_p(end=True)

        # final idx array (to keep relative ordering)
        final_idx = np.union1d(np.argwhere(y > 0).reshape(-1), top_p_idx_neg).astype(int)
        return X[final_idx], y[final_idx], group_new

    def _update_p(self, end, preds_neg=None, idx_pos=None, idx_neg=None):
        """

        :param end: if true, p changes after iteration, else it depends on query
        :param preds_neg: array of predictions ('pred', 'idx') of negative examples
        :param idx_pos: array of indexes of positive examples
        :param idx_neg: array of indexes of negative examples
        """
        if end:
            # p changes after one iteration
            if self.method == 'fixed':
                # sel gb standard
                pass
            elif self.method == 'random_iter':
                self.p = uniform(0.0, self.max_p)
            elif self.method == 'inverse':
                # decreases by k factor each iteration
                self.p = self.p * self.k_factor
        else:
            # p depends on query
            if self.method == 'false_positives':
                # only false positives
                self.p = (preds_neg['pred'] > 0).sum() / max(preds_neg.size, 1)
            elif self.method == 'equal_size':
                # number of neg equals number of pos
                self.p = min(len(idx_pos), preds_neg.size) / max(preds_neg.size, 1)
            elif self.method == 'random_query':
                self.p = uniform(0.0, self.max_p)

    @Timeit('SelGB fit')
    def fit(self, X, y, group=None, params=None, verbose=True,
            eval_set=None, eval_names=None, eval_group=None, early_stopping_rounds=None):
        if params is None:
            params = {
                'objective': 'rank_xendcg',
                'learning_rate': 0.05,
                'num_leaves': 64,
                'metric': ['ndcg'],
                'ndcg_eval_at': 10,
                'force_row_wise': True,
                'max_bin': 127
            }

        if group is None:
            raise ValueError("Should set group for ranking task")

        if eval_set is not None:
            if eval_group is None:
                raise ValueError("Eval_group cannot be None when eval_set is not None")
            elif len(eval_group) != len(eval_set):
                raise ValueError("Length of eval_group should be equal to eval_set")
            elif (isinstance(eval_group, dict)
                  and any(i not in eval_group or eval_group[i] is None for i in range(len(eval_group)))
                  or isinstance(eval_group, list)
                  and any(group is None for group in eval_group)):
                raise ValueError("Should set group for all eval datasets for ranking task; "
                                 "if you use dict, the index should start from 0")

        self._eval_result = {name: OrderedDict({'ndcg@10': []}) for name in eval_names}
        self.resample_count = np.zeros(X.shape[0])
        if type(X) != np.ndarray:
            # required since scipy has a nasty bug when dealing with very large matrices
            X = X.toarray()
        X_new, y_new, group_new = X, y, group
        n_trees = 0
        # we iterate N (ensemble size) mod n (#iterations between sampling) times
        for self.iter in range(1, (self.n_estimators // self.n_iter_sample) + 1):
            # create lightgbm dataset
            dstar = self._construct_dataset(X_new, y_new, group_new)
            valid_sets = []
            for j, valid_data in enumerate(eval_set):
                valid_set = self._construct_dataset(valid_data[0], valid_data[1], eval_group[j], dstar)
                valid_sets.append(valid_set)
            # each step, we fit n regression trees
            tmp_evals_result = {}
            print('[INFO] Datasets ready, training booster')
            self.booster = lgb.train(params, dstar,
                                     num_boost_round=self.n_iter_sample,
                                     valid_sets=valid_sets,
                                     valid_names=eval_names,
                                     init_model=self.booster,
                                     keep_training_booster=True,
                                     verbose_eval=verbose,
                                     evals_result=tmp_evals_result)
            del dstar, valid_sets
            self.update_eval_result(tmp_evals_result)
            n_trees += self.n_iter_sample
            # check for early stopping
            if early_stopping_rounds is not None and n_trees >= early_stopping_rounds + self.n_iter_sample:
                for i, j in zip(range(n_trees - self.n_iter_sample,
                                      n_trees),
                                range(n_trees - early_stopping_rounds - self.n_iter_sample,
                                      n_trees - early_stopping_rounds)):
                    if self._eval_result['valid']['ndcg@10'][i] <= self._eval_result['valid']['ndcg@10'][j]:
                        print('[SelGB] [Info] early stopping, best iteration:', j)
                        self.save_model('tmp_early_stop.txt', num_iteration=j)
                        self.booster = lgb.Booster(model_file='tmp_early_stop.txt')
                        break
            if 'tmp_early_stop.txt' in os.listdir():
                # we have early stopped
                os.remove('tmp_early_stop.txt')
                break
            # select new training data
            X_new, y_new, group_new = self._sel_sample(X, y, group)

        return self

    def predict(self, X):
        return self.booster.predict(X)

    def update_eval_result(self, tmp_eval_results):
        for key in tmp_eval_results:
            self._eval_result[key]['ndcg@10'] += tmp_eval_results[key]['ndcg@10']

    def get_eval_result(self):
        return self._eval_result

    def save_eval_result(self, path, filename=None):
        if filename is not None:
            dump_obj(self.get_eval_result(), path, filename)
        else:
            dump_obj(path, 'selgb-' + self.method)

    def save_model(self, filename, num_iteration=None):
        self.booster.save_model(filename, num_iteration=num_iteration)


if __name__ == '__main__':
    from utils import load_data
    base_path = os.getcwd()
    datasets_path = os.path.join(base_path, 'datasets')
    istellax_path = os.path.join(datasets_path, "istella-short")
    train_file = os.path.join(istellax_path, "train.txt")
    valid_file = os.path.join(istellax_path, "vali.txt")
    test_file = os.path.join(istellax_path, "test.txt")
    print('Loading data')
    train_data, train_labels, train_query_lens = load_data(train_file)
    print('[INFO] Training set loaded')
    valid_data, valid_labels, valid_query_lens = load_data(valid_file)
    print('[INFO] Validation set loaded')
    test_data, test_labels, test_query_lens = load_data(test_file)
    print('[INFO] Testing set loaded')
    model = LGBMSelGB(n_estimators=15, n_iter_sample=10, p=0.01, max_resample=10, method='fixed')
    print('[INFO] Starting fitting')
    eval_set = [(train_data, train_labels),
                (valid_data, valid_labels),
                (test_data, test_labels)]
    eval_group = [train_query_lens, valid_query_lens, test_query_lens]
    eval_names = ['train', 'valid', 'test']
    model.fit(train_data, train_labels, train_query_lens,
              eval_set=eval_set, eval_group=eval_group, eval_names=eval_names,
              verbose=5)
