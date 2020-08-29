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


def randomization_test(X, y_true, model_a, model_b, ndcg_pos=[10], n_perm=10000):
    """
    This methods performs a randomization test (i.e. computes statistical significance)
    of the performance difference between model_a and model_b
    """
    y_pred_a = model_a.predict(X)
    y_pred_b = model_b.predict(X)
    for pos in ndcg_pos:
        ndcg_a = ndcg_score(y_score=y_pred_a, y_true=y_true, k=pos)
        ndcg_b = ndcg_score(y_score=y_pred_b, y_true=y_true, k=pos)
        