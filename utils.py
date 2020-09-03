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


@Timeit('load_data')
def load_data(filename, dtype=np.float32):
    raw = load_svmlight_file(filename, dtype=dtype, query_id=True)
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
            # ndcg_score requires array of shape (n_samples, n_labels)
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


def scale_down(data, labels, group, max_docs):
    # dtype used to keep track of idx while sorting
    dtype = np.dtype([('bm25f', np.float32), ('idx', np.uint64)])
    chosen_idx_neg = []
    query_id_new = []
    idx_pos = np.argwhere(labels > 0).reshape(-1)
    idx_pos_set = set(idx_pos)
    idx_neg_set = set(np.argwhere(labels == 0).reshape(-1))
    cum = 0
    for q_id, query_size in enumerate(group):
        idx_query = [x for x in range(cum, cum + query_size)]
        idx_query_pos = [x for x in idx_query if x in idx_pos_set]
        idx_query_neg = [x for x in idx_query if x in idx_neg_set]
        bm25f = data[idx_query_neg][:, 44].toarray()
        bm25f_idx = np.array(list(zip(bm25f, idx_query_neg)), dtype=dtype)
        bm25f_idx = np.sort(bm25f_idx, order='bm25f')[::-1]
        rank = min(len(idx_query_neg), max_docs - len(idx_query_pos))
        if rank < 0:
            rank = 0
        chosen_idx_neg += list(bm25f_idx['idx'])[:rank]
        cum += query_size
        query_id_new += [q_id] * (len(idx_query_pos) + rank)
    final_idx = np.union1d(idx_pos, chosen_idx_neg).astype(int)
    return data[final_idx], labels[final_idx], query_id_new


if __name__ == '__main__':
    from sklearn.datasets import dump_svmlight_file
    base_dir = "../datasets/istella-short/sample"
    train_file = base_dir + "/train.txt"
    train_data, train_labels, train_query_lens = load_data(train_file, dtype=np.float32)
    print("Loaded training set")
    new_data, new_labels, new_query_id = scale_down(train_data, train_labels, train_query_lens, 20)
    print('scaled')
    dump_svmlight_file(new_data, new_labels, f=base_dir+'/scaled.txt', query_id=new_query_id)
    print('dumped')
