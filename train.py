#!/usr/bin/env python3.6

import os
import sys

import lightgbm as lgbm

from LGBMSelGB import LGBMSelGB
from utils import Timeit, compare_model_error, load_data

base_path = os.getcwd()
datasets_path = os.path.join(base_path, 'datasets')
train_file = os.path.join(datasets_path, "train.txt")
valid_file = os.path.join(datasets_path, "vali.txt")
test_file = os.path.join(datasets_path, "test.txt")
output_path = os.path.join(base_path, 'output')
models_path = os.path.join(output_path, 'models')
results_path = os.path.join(output_path, 'results')


def train():
    print('Loading data')
    train_data, train_labels, train_query_lens = load_data(train_file)
    print('Training set loaded')
    valid_data, valid_labels, valid_query_lens = load_data(valid_file)
    print('Validation set loaded')
    test_data, test_labels, test_query_lens = load_data(test_file)
    print('Testing set loaded')

    train_set = lgbm.Dataset(train_data, label=train_labels, group=train_query_lens)
    eval_set = [(train_data, train_labels),
                (valid_data, valid_labels),
                (test_data, test_labels)]
    eval_group = [train_query_lens, valid_query_lens, test_query_lens]
    eval_names = ['train', 'valid', 'test']
    valid_sets = []
    for i, valid_data in enumerate(eval_set):
        ds = lgbm.Dataset(valid_data[0], valid_data[1], group=eval_group[i], reference=train_set)
        valid_sets.append(ds)

    params = {
        'objective': 'lambdarank',
        'max_position': 10,
        'learning_rate': 0.05,
        'num_leaves': 64,
        'metric': ['ndcg'],
        'ndcg_eval_at': 10
    }

    @Timeit('LGBM train')
    def train_lgbm_model():
        evals_result = {}
        lgb_model = lgbm.train(params, train_set, num_boost_round=1000,
                               valid_sets=valid_sets, valid_names=eval_names,
                               verbose_eval=5, evals_result=evals_result)
        return lgb_model, evals_result

    lgb_model, lgb_info = train_lgbm_model()

    strategies = ('fixed', 'random_iter', 'random_query', 'inverse',
                  'wrong_neg', 'equal_size', 'delta')

    selgb_base = LGBMSelGB(n_estimators=1000, n_iter_sample=1, p=0.01, method='fixed')
    selgb_base.fit(train_data, train_labels, train_query_lens,
                   eval_set=eval_set, eval_group=eval_group, eval_names=eval_names,
                   verbose=5)

    compare_model_error(eval_data=[lgb_info, selgb_base.get_eval_result()],
                        model_names=['LightGBM', 'Selgb'])

    lgb_model.save_model(os.path.join(models_path, 'lightgbm.txt'))
    selgb_base.save_model(os.path.join(models_path, 'selgb_base.txt'))


def test_load():
    import numpy as np
    print('Loading data')
    train_data, train_labels, train_query_lens = load_data(train_file)
    print('Training set loaded')
    print('Shape:', train_data.shape)
    final_idx = np.sort(np.random.choice(list(range(train_data.shape[0])), size=int(train_data.shape[0]*0.1), replace=False))
    sliced = train_data[final_idx]
    print('Sliced1 shape:', sliced.shape)
    final_idx = np.sort(np.random.choice(list(range(train_data.shape[0])), size=int(train_data.shape[0]*0.01), replace=False))
    sliced = train_data[final_idx]
    print('Sliced2 shape:', sliced.shape)


if __name__ == '__main__':
    test_load()
    sys.exit(0)
