import numpy as np
import lightgbm as lgb
from collections import OrderedDict
from numpy.random import uniform

from .utils import timeit

class LGBMSelGB:

    def __init__(self, n_estimators=100, n_iter_sample=1, p=0.1,
                 max_p=0.5, k_factor=0.5, method='random'):
        self.n_estimators = n_estimators
        self.n_iter_sample = n_iter_sample
        self.p = p
        self.method = method
        self.max_p = max_p
        self.k_factor = k_factor
        self.booster = None
        self.enable_file = False
        self.evals_result = {}

    def _construct_dataset(self, X, y, group, reference=None):
        return lgb.Dataset(X, label=y, group=group, reference=reference)

    def _sel_sample(self, X, y, group):
        # dtype used to keep track of idx while sorting
        dtype = np.dtype([('pred', np.float64), ('idx', np.uint64)])
        # get positive and negative indexes
        idx_pos = np.argwhere(y > 0).reshape(-1)
        idx_neg = set(np.argwhere(y == 0).reshape(-1))
        # get top p% negative examples, based on group
        # also computes new groups (number of docs for each query)
        cum = 0
        group_new = []
        top_p_idx_neg = []
        for g in group:
            idx_pos_query = [x for x in range(cum, cum+g) if x in idx_pos]
            idx_neg_query = [x for x in range(cum, cum+g) if x in idx_neg]
            preds = self.predict(X[idx_neg_query])
            preds = np.array(list(zip(preds, idx_neg_query)), dtype=dtype)
            preds.sort(order='pred')
            self._update_p(preds=preds['pred'], pos_size=len(idx_pos_query), end=False)
            top_p = int(self.p * preds.shape[0])
            if top_p < 1 and idx_neg_query:
                top_p = 1
            top_p_idx_neg += list(preds['idx'][:top_p].astype(int))
            group_new.append(top_p + len(idx_pos_query))
            cum += g

        self._update_p(end=True)

        # final idx array (to keep relative ordering)
        final_idx = np.union1d(idx_pos, top_p_idx_neg)

        # return positive and selected negative examples
        return X[final_idx], y[final_idx], group_new

    def _update_p(self, end, preds=None, pos_size=None):
        if end:
            if self.method == 'fixed':
                pass
            elif self.method == 'random':
                self.p = uniform(0.0, self.max_p)
            elif self.method == 'inverse':
                self.p = self.p * self.k_factor
            print('[SelGB] [Info] new p:', self.p)
        else:
            if self.method == 'wrong_neg':
                if preds.size > 0:
                    self.p = (preds > 0).sum() / preds.shape[0]
                else:
                    self.p = 0

    def get_evals_result(self):
        return self.evals_result

    def fit(self, X, y, group=None, verbose=True,
            eval_set=None, eval_names=None, eval_group=None):
        # basic parameters for lgb train
        params = {
            'objective': 'lambdarank',
            'max_position': 10,
            'learning_rate': 0.1,
            'num_leaves': 16,
            'min_data_in_leaf': 5,
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

        self.evals_result = {name: OrderedDict({'ndcg@10': []}) for name in eval_names}

        X_new, y_new, group_new = X, y, group
        # we iterate N (ensemble size) mod n (#iterations between sampling) times
        for i in range(self.n_estimators // self.n_iter_sample):
            # create lightgbm dataset
            dstar = self._construct_dataset(X_new, y_new, group_new)
            valid_sets = []
            for i, valid_data in enumerate(eval_set):
                valid_set = self._construct_dataset(valid_data[0], valid_data[1], eval_group[i], dstar)
                valid_sets.append(valid_set)
            # each step, we fit n regression trees
            if self.enable_file:
                self.booster = 'tmp.txt'
            tmp_evals_result = {}
            self.booster = lgb.train(params, dstar,
                                     num_boost_round=self.n_iter_sample,
                                     valid_sets=valid_sets,
                                     valid_names=eval_names,
                                     init_model=self.booster,
                                     keep_training_booster=True,
                                     verbose_eval=verbose,
                                     evals_result=tmp_evals_result)
            self.update_evals_result(tmp_evals_result)
            if self.enable_file:
                self.booster.save_model('tmp.txt')
            # select new training data
            X_new, y_new, group_new = self._sel_sample(X, y, group)
        return self

    def predict(self, X):
        return self.booster.predict(X)

    def update_evals_result(self, tmp_evals_result):
        for key in tmp_evals_result:
            self.evals_result[key]['ndcg@10'] += tmp_evals_result[key]['ndcg@10']
