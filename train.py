#!/usr/bin/env python3.6

import os
import sys

import lightgbm as lgb

from LGBMSelGB import LGBMSelGB
from utils import load_data, dump_obj

base_path = os.getcwd()
datasets_path = os.path.join(base_path, 'datasets')
istellax_path = os.path.join(datasets_path, "istella-X")
train_file = os.path.join(istellax_path, "train.txt")
valid_file = os.path.join(istellax_path, "vali.txt")
test_file = os.path.join(istellax_path, "test.txt")
output_path = os.path.join(base_path, 'output_istella')
models_path = os.path.join(output_path, 'models')
results_path = os.path.join(output_path, 'results')
logs_path = os.path.join(output_path, 'logs')


def train():
    algo = sys.argv[1]
    print('[INFO] Chosen algo is:', algo)
    print('[INFO] Loading data')
    train_data, train_labels, train_query_lens = load_data(train_file)
    print('[INFO] Training set loaded')
    valid_data, valid_labels, valid_query_lens = load_data(valid_file)
    print('[INFO] Validation set loaded')
    test_data, test_labels, test_query_lens = load_data(test_file)
    print('[INFO] Testing set loaded')

    eval_set = [(train_data, train_labels),
                (valid_data, valid_labels),
                (test_data, test_labels)]
    eval_group = [train_query_lens, valid_query_lens, test_query_lens]
    eval_names = ['train', 'valid', 'test']
    params = {
        'objective': 'rank_xendcg',
        'learning_rate': 0.05,
        'num_leaves': 64,
        'metric': ['ndcg'],
        'ndcg_eval_at': 10,
        'force_row_wise': True,
        'max_bin': 127
    }

    strategies = ('fixed', 'random_iter', 'random_query', 'decay',
                  'false_positives', 'equal_size', 'delta', 'limit_resample')
    n_estimators = 1000
    early_stopping_rounds = 100
    n_iter_sample = 1
    verbose = 5
    print('[INFO] Starting training')

    if algo in ('lgbm base', 'lgbm goss', 'lambdarank'):
        if algo == 'lambdarank':
            params['objective'] = 'lambdarank'
        elif algo == 'lgbm goss':
            params['boosting'] = 'goss'
        train_set = lgb.Dataset(train_data, label=train_labels, group=train_query_lens)
        eval_results = {}
        valid_sets = [train_set]
        for i, data in enumerate(eval_set[1:]):
            ds = lgb.Dataset(data[0], data[1], group=eval_group[1:][i], reference=train_set)
            valid_sets.append(ds)
        model = lgb.train(params, train_set, num_boost_round=n_estimators,
                          valid_sets=valid_sets, valid_names=eval_names,
                          verbose_eval=verbose, evals_result=eval_results,
                          early_stopping_rounds=early_stopping_rounds)
        dump_obj(eval_results, results_path, algo)
    elif algo in strategies:
        if algo == 'fixed':
            print('[INFO] Starting fitting, tuning')
            eval_results = {}
            for p in (0.5, 0.25, 0.1, 0.05, 0.01):
                print('[INFO] p value:', p)
                model = LGBMSelGB(n_estimators=n_estimators, n_iter_sample=n_iter_sample, p=p, method=algo)
                model.fit(train_data, train_labels, train_query_lens,
                          eval_set=eval_set, eval_group=eval_group, eval_names=eval_names,
                          verbose=verbose, early_stopping_rounds=early_stopping_rounds)
                eval_results[p] = model.get_eval_result()
                dump_obj(eval_results, results_path, algo)
        elif algo in ('equal_size', 'false_positives'):
            print('[INFO] Starting fitting, no tuning')
            model = LGBMSelGB(n_estimators=n_estimators, n_iter_sample=n_iter_sample, method=algo)
            model.fit(train_data, train_labels, train_query_lens,
                      eval_set=eval_set, eval_group=eval_group, eval_names=eval_names,
                      verbose=verbose, early_stopping_rounds=early_stopping_rounds)
            eval_results = model.get_eval_result()
            dump_obj(eval_results, results_path, algo)
        elif algo.startswith('random'):
            print('[INFO] Starting fitting, with tuning')
            eval_results = {}
            for max_p in (0.5, 0.25, 0.1, 0.05, 0.02):
                print('[INFO] max_p value:', max_p)
                model = LGBMSelGB(n_estimators=n_estimators, n_iter_sample=n_iter_sample, max_p=max_p, method=algo)
                model.fit(train_data, train_labels, train_query_lens,
                          eval_set=eval_set, eval_group=eval_group, eval_names=eval_names,
                          verbose=verbose, early_stopping_rounds=early_stopping_rounds)
                eval_results[max_p] = model.get_eval_result()
                dump_obj(eval_results, results_path, algo)
        elif algo == 'delta':
            print('[INFO] Starting fitting, with tuning')
            eval_results = {}
            for delta_pos in (3, 5, 10):
                delta = 0.25
                print('[INFO] delta_pos:', delta_pos)
                print('[INFO] delta:', delta)
                model = LGBMSelGB(n_estimators=n_estimators, n_iter_sample=n_iter_sample,
                                  delta_pos=delta_pos, delta=delta, method=algo)
                model.fit(train_data, train_labels, train_query_lens,
                          eval_set=eval_set, eval_group=eval_group, eval_names=eval_names,
                          verbose=verbose, early_stopping_rounds=early_stopping_rounds)
                eval_results[(delta_pos, delta)] = model.get_eval_result()
                dump_obj(eval_results, results_path, algo)

            for delta in (1, 0.5, 0.1):
                delta_pos = 5
                print('[INFO] delta_pos:', delta_pos)
                print('[INFO] delta:', delta)
                model = LGBMSelGB(n_estimators=n_estimators, n_iter_sample=n_iter_sample,
                                  delta_pos=delta_pos, delta=delta, method=algo)
                model.fit(train_data, train_labels, train_query_lens,
                          eval_set=eval_set, eval_group=eval_group, eval_names=eval_names,
                          verbose=verbose, early_stopping_rounds=early_stopping_rounds)
                eval_results[(delta_pos, delta)] = model.get_eval_result()
                dump_obj(eval_results, results_path, algo)

        elif algo == 'decay':
            print('[INFO] Starting fitting, with tuning')
            eval_results = {}
            for p in (0.5, 0.75):
                k = 0.985
                print('[INFO] p:', p)
                print('[INFO] k:', k)
                model = LGBMSelGB(n_estimators=n_estimators, n_iter_sample=n_iter_sample,
                                  p=p, k_factor=k, method=algo)
                model.fit(train_data, train_labels, train_query_lens,
                          eval_set=eval_set, eval_group=eval_group, eval_names=eval_names,
                          verbose=verbose, early_stopping_rounds=early_stopping_rounds)
                eval_results[(p, k)] = model.get_eval_result()
                dump_obj(eval_results, results_path, algo)
            for k in (0.98, 0.99):
                p = 0.5
                print('[INFO] p:', p)
                print('[INFO] k:', k)
                model = LGBMSelGB(n_estimators=n_estimators, n_iter_sample=n_iter_sample,
                                  p=p, k_factor=k, method=algo)
                model.fit(train_data, train_labels, train_query_lens,
                          eval_set=eval_set, eval_group=eval_group, eval_names=eval_names,
                          verbose=verbose, early_stopping_rounds=early_stopping_rounds)
                eval_results[(p, k)] = model.get_eval_result()
                dump_obj(eval_results, results_path, algo)

        elif algo == 'limit_resample':
            print('[INFO] Starting fitting, with tuning')
            eval_results = {}
            algo = 'limit_resample_rndq'
            for max_resample in [250, 500, 0.5, 0.75]:
                print('[INFO] max_resample', max_resample)
                model = LGBMSelGB(n_estimators=n_estimators, n_iter_sample=n_iter_sample,
                                  max_resample=max_resample, method='random_query', max_p=0.05)
                model.fit(train_data, train_labels, train_query_lens,
                          eval_set=eval_set, eval_group=eval_group, eval_names=eval_names,
                          verbose=verbose, early_stopping_rounds=early_stopping_rounds)
                eval_results[max_resample] = model.get_eval_result()
                dump_obj(eval_results, results_path, algo)
    else:
        raise ValueError('algo parameter is wrong')
    print('FITTING OVER!')
    dump_obj(eval_results, results_path, algo)
    print('dumped results')
    model.save_model(os.path.join(models_path, algo + '.txt'))
    print('saved model')


if __name__ == '__main__':
    train()
    sys.exit(0)
