import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import groupby
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import ndcg_score


class Timeit:
    """
    Decorator class used to log a function's execution time
    """

    def __init__(self, f_name):
        self.f_name = f_name

    def __call__(self, f):
        def timed(*args, **kwargs):
            from time import time
            start = time()
            result = f(*args, **kwargs)
            end = time()
            print(self.f_name + ' execution took {:.2f} min'.format((end - start) / 60))
            return result

        return timed

    
def load_data(filename):
    raw = load_svmlight_file(filename, query_id=True)
    data = raw[0]
    labels = raw[1]
    query_lens = [len(list(group)) for key, group in groupby(raw[2])]
    return data, labels, query_lens


def compare_model_error(eval_data, model_names, metric='ndcg@10', plot=False, savefig=False, filename='foo.png'):
    fig, axes = plt.subplots(3, figsize=(10, 15))
    metric_150_trees = pd.DataFrame(data=np.zeros((len(model_names), len(eval_data[0].keys()))), dtype=np.float64,
                                    index=model_names, columns=eval_data[0].keys())
    metric_all_trees = pd.DataFrame(data=np.zeros((len(model_names), len(eval_data[0].keys()))), dtype=np.float64,
                                    index=model_names, columns=eval_data[0].keys())
    for eval_results, model_name in zip(eval_data, model_names):
        for ax, eval_set in zip(axes, eval_results):
            ax.plot(eval_results[eval_set][metric], label=model_name)
            ax.grid()
            ax.legend()
            ax.set_xlabel('#Trees')
            ax.set_ylabel(metric)
            ax.set_title('Model error on ' + eval_set + ' set')
            metric_150_trees.loc[model_name][eval_set] = eval_results[eval_set][metric][150]
            metric_all_trees.loc[model_name][eval_set] = eval_results[eval_set][metric][-1]
    if plot:
        plt.show()
    if savefig:
        plt.savefig(filename)
    return metric_150_trees, metric_all_trees

@Timeit('Randomization test')
def randomization_test(X, y_true, group, model_a, model_b, metric='ndcg@10', n_perm=10000):
    """
    This methods performs a randomization test (i.e. computes statistical significance)
    of the performance difference between model_a and model_b
    """
    # Get predictions 
    y_pred_a = model_a.predict(X)
    y_pred_b = model_b.predict(X)
    
    # per-query evaluation of given metric
    idx_start = 0
    n_queries = len(group)
    query_scores_a = np.zeros(n_queries, dtype=np.float32)
    query_scores_b = np.zeros(n_queries, dtype=np.float32)
    for i, query_size in enumerate(group):
        if metric.startswith('ndcg@'):
            idx_end = idx_start + query_size
            cutoff = int(metric.split('@')[-1])
            # ndcg_scores requires array of shape (n_samples, n_labels)
            query_scores_a[i] = ndcg_score(y_score=[y_pred_a[idx_start:idx_end]], 
                                           y_true=[y_true[idx_start:idx_end]], k=cutoff)
            query_scores_b[i] = ndcg_score(y_score=[y_pred_b[idx_start:idx_end]], 
                                           y_true=[y_true[idx_start:idx_end]], k=cutoff)
            idx_start += query_size
        else:
            raise ValueError('Wrong or unsupported metric')
    
    # randomization test
    if np.mean(query_scores_a) > np.mean(query_scores_b):
        best_scores = query_scores_a
        worst_scores = query_scores_b
    else:
        best_scores = query_scores_b
        worst_scores = query_scores_a
    
    diff = np.mean(best_scores) - np.mean(worst_scores)
    abs_diff = np.abs(diff)
    p1, p2 = 0.0, 0.0
    best_sum, worst_sum = np.sum(best_scores), np.sum(worst_scores)
    
    for i in range(n_perm):
        # select a random subset 
        subset = np.random.choice([False, True], n_queries)
        
        best_sub_sum = np.sum(best_scores[subset])
        worst_sub_sum = np.sum(worst_scores[subset])
        
        # compute avg performance of randomized models
        best_mean = (best_sum - best_sub_sum + worst_sub_sum) / float(n_queries)
        worst_mean = (worst_sum - worst_sub_sum + best_sub_sum) / float(n_queries)
        
        delta = best_mean - worst_mean
        
        if delta >= diff:
            p1 += 1
        if np.abs(delta) >= abs_diff:
            p2 += 1
            
    p1 /= n_perm
    p2 /= n_perm
    
    return p1, p2