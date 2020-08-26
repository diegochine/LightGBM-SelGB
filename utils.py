import matplotlib.pyplot as plt
import pandas as pd


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


def compare_model_error(data, names, plot=False, savefig=False):
    """
    :param data: list of dict with keys "train", "valid", "test"
    :param names: list of model names
    :param savefig:
    :return:
    """
    fig, axes = plt.subplots(2, figsize=(9.6, 7.2))
    metric_150_trees = pd.DataFrame(data=np.zeros((len(model_names), len(data[0].keys()))), dtype=np.float32,
                                    index=model_names, columns=data[0].keys())
    metric_all_trees = pd.DataFrame(data=np.zeros((len(model_names), len(data[0].keys()))), dtype=np.float32,
                                    index=model_names, columns=data[0].keys())
    for eval_results, model_name in zip(eval_data, model_names):
        for axes_row, eval_set in zip(axes, eval_results):
            for ax in axes_row:
                ax.plot(eval_results[eval_set][metric], label=model_name)
                ax.grid()
                ax.legend()
                ax.xlabel('#Trees')
                ax.ylabel(metric)
                ax.title('Model error on', eval_set, 'set')
                metric_150_trees.loc[model_name][eval_set] = eval_results['train'][metric][150]
                metric_all_trees.loc[model_name][eval_set] = eval_results['train'][metric][-1]
        if plot:
            plt.show()
        if savefig:
            plt.savefig(filename)
        print('{}, 150 trees: {:.4f} - {:.4f} - {:.4f}'.format(model_name,
                                                               eval_results['train'][metric][150],
                                                               eval_results['valid'][metric][150],
                                                               eval_results['test'][metric][150]))
        print('{}, all trees: {:.4f} - {:.4f} - {:.4f}'.format(model_name,
                                                               eval_results['train'][metric][-1],
                                                               eval_results['valid'][metric][-1],
                                                               eval_results['test'][metric][-1]))
        print('{}, 150 trees: {:.4f} - {:.4f} - {:.4f}'.format(model_name,
                                                               eval_results['train'][metric][150],
                                                               eval_results['test'][metric][150]))
        print('{}, all trees: {:.4f} - {:.4f} - {:.4f}'.format(model_name,
                                                               eval_results['train'][metric][-1],
                                                               eval_results['test'][metric][-1]))
    return metric_150_trees, metric_all_trees


def plot_doc_score():
    """
    Plots a graph showing the variation in document scores while training the model
    """
    pass
