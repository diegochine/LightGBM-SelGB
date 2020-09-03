#!/usr/bin/env python3.6

import os
import sys

import lightgbm as lgb

from LGBMSelGB import LGBMSelGB
from utils import load_data, dump_obj

base_path = os.getcwd()
datasets_path = os.path.join(base_path, 'datasets')
train_file = os.path.join(datasets_path, "train.txt")
valid_file = os.path.join(datasets_path, "vali.txt")
test_file = os.path.join(datasets_path, "test.txt")
output_path = os.path.join(base_path, 'output')
models_path = os.path.join(output_path, 'models')
results_path = os.path.join(output_path, 'results')
logs_path = os.path.join(output_path, 'logs')


def train(algo):
    print('Loading data')
    train_data, train_labels, train_query_lens = load_data(train_file)
    print('Training set loaded')
    valid_data, valid_labels, valid_query_lens = load_data(valid_file)
    print('Validation set loaded')
    test_data, test_labels, test_query_lens = load_data(test_file)
    print('Testing set loaded')

    train_set = lgb.Dataset(train_data, label=train_labels, group=train_query_lens)
    eval_set = [(train_data, train_labels),
                (valid_data, valid_labels),
                (test_data, test_labels)]
    eval_group = [train_query_lens, valid_query_lens, test_query_lens]
    eval_names = ['train', 'valid', 'test']
    valid_sets = []
    for i, valid_data in enumerate(eval_set):
        ds = lgb.Dataset(valid_data[0], valid_data[1], group=eval_group[i], reference=train_set)
        valid_sets.append(ds)

    params = {
        'objective': 'lambdarank',
        'max_position': 10,
        'learning_rate': 0.05,
        'num_leaves': 64,
        'metric': ['ndcg'],
        'ndcg_eval_at': 10
    }

    strategies = ('fixed', 'random_iter', 'random_query', 'inverse',
                  'false_positives', 'equal_size', 'delta')
    n_estimators = 1000
    early_stopping_rounds = 100
    verbose = 5

    if algo in ('lgbm base', 'lgbm goss'):
        if algo == 'lgbm goss':
            params['boosting'] = 'goss'
        eval_results = {}
        model = lgb.train(params, train_set, num_boost_round=n_estimators,
                          valid_sets=valid_sets, valid_names=eval_names,
                          verbose_eval=verbose, evals_result=eval_results,
                          early_stopping_rounds=early_stopping_rounds)
    elif algo in strategies:
        model = LGBMSelGB(n_estimators=n_estimators, n_iter_sample=10, p=0.01, method=algo)
        model.fit(train_data, train_labels, train_query_lens,
                  eval_set=eval_set, eval_group=eval_group, eval_names=eval_names,
                  verbose=verbose)
        eval_results = model.get_eval_result()
    else:
        raise ValueError('algo parameter is wrong')

    model.save_model(os.path.join(models_path, algo + '.txt'))
    dump_obj(eval_results, results_path, algo)


if __name__ == '__main__':
    train(sys.argv[1])
    sys.exit(0)
