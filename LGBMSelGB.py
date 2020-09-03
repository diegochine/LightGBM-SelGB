import os

import numpy as np
import lightgbm as lgb
from collections import OrderedDict
from numpy.random import uniform

from .utils import Timeit, dump_obj


class LGBMSelGB:

    def __init__(self, n_estimators=100, n_iter_sample=1, p=0.1,
                 max_p=0.5, k_factor=0.5, delta=0.5, delta_pos=10, method='fixed'):
        self.n_estimators = n_estimators
        self.n_iter_sample = n_iter_sample
        self.p = p
        self.method = method
        self.max_p = max_p
        self.k_factor = k_factor
        self.delta = delta
        self.delta_pos = delta_pos
        self.booster = None
        self._eval_result = {}

    def _construct_dataset(self, X, y, group, reference=None):
        return lgb.Dataset(X, label=y, group=group, reference=reference)

    @Timeit('SelGB sample')
    def _sel_sample(self, X, y, group):
        # dtype used to keep track of idx while sorting
        dtype = np.dtype([('pred', np.float64), ('idx', np.uint64)])
        # get positive and negative indexes
        idx_pos = np.argwhere(y > 0).reshape(-1)
        idx_pos_set = set(idx_pos)
        idx_neg_set = set(np.argwhere(y == 0).reshape(-1))
        # get top p% negative examples, based on group
        # also computes new groups (number of docs for each query)
        cum = 0
        group_new = []
        top_p_idx_neg = []
        for query_size in group:
            idx_query = [x for x in range(cum, cum + query_size)]
            idx_query_pos = [x for x in idx_query if x in idx_pos_set]
            idx_query_neg = [x for x in idx_query if x in idx_neg_set]
            preds = self.predict(X[idx_query])
            preds = np.array(list(zip(preds, idx_query)), dtype=dtype)
            preds.sort(order='pred')
            preds = preds[::-1]
            if self.method == 'delta':
                score = preds[min(self.delta_pos, preds.size - 1)]['pred']
                preds_neg = preds[np.isin(preds['idx'], idx_query_neg)]
                idx_delta = np.logical_and(score - self.delta < preds_neg['pred'],
                                           preds_neg['pred'] < score + self.delta)
                top_p_idx_neg += list(preds_neg[idx_delta]['idx'].astype(int))
                group_new.append(idx_delta.sum() + len(idx_query_pos))
            else:
                self._update_p(preds=preds, idx_pos=idx_query_pos, idx_neg=idx_query_neg, end=False)
                top_p = int(self.p * len(idx_query_neg))
                if top_p < 1 and idx_query_neg:
                    top_p = 1
                top_p_idx_neg += list(preds[np.isin(preds['idx'], idx_query_neg)]['idx'][:top_p].astype(int))
                group_new.append(top_p + len(idx_query_pos))
            cum += query_size

        self._update_p(end=True)

        # final idx array (to keep relative ordering)
        final_idx = np.union1d(idx_pos, top_p_idx_neg).astype(int)

        # return positive and selected negative examples
        return X[final_idx], y[final_idx], group_new

    def _update_p(self, end, preds=None, idx_pos=None, idx_neg=None):
        """

        :param end: if true, p changes after iteration, else it depends on query
        :param preds: array of predictions ('pred', 'idx')
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
            preds_neg = preds[np.isin(preds['idx'], idx_neg)]
            if self.method == 'false_positives':
                # only false positives
                if preds_neg.size > 0:
                    self.p = (preds_neg['pred'] > 0).sum() / preds_neg.size
                else:
                    self.p = 0
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
                'objective': 'lambdarank',
                'max_position': 10,
                'learning_rate': 0.05,
                'num_leaves': 64,
                'metric': ['ndcg'],
                'ndcg_eval_at': 10
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

        X_new, y_new, group_new = X, y, group
        n_trees = 0
        # we iterate N (ensemble size) mod n (#iterations between sampling) times
        for _ in range(self.n_estimators // self.n_iter_sample):
            # create lightgbm dataset
            dstar = self._construct_dataset(X_new, y_new, group_new)
            valid_sets = []
            for j, valid_data in enumerate(eval_set):
                valid_set = self._construct_dataset(valid_data[0], valid_data[1], eval_group[j], dstar)
                valid_sets.append(valid_set)
            # each step, we fit n regression trees
            tmp_evals_result = {}
            self.booster = lgb.train(params, dstar,
                                     num_boost_round=self.n_iter_sample,
                                     valid_sets=valid_sets,
                                     valid_names=eval_names,
                                     init_model=self.booster,
                                     keep_training_booster=True,
                                     verbose_eval=verbose,
                                     evals_result=tmp_evals_result)
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
