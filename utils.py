import matplotlib.pyplot as plt


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
    for eval_results, model_name in zip(data, names):
        if plot:
            plt.figure()
            for key in eval_results:
                plt.plot(eval_results[key]['ndcg@10'], label=model_name + ' ' + key)
            plt.grid()
            plt.legend()
            plt.xlabel('#Trees')
            plt.ylabel('ndcg@10')
            plt.title('Model error')
            plt.show()
            if savefig:
                plt.savefig('foo.png')
        print('{}, 150 trees: {:.4f} - {:.4f} - {:.4f}'.format(model_name,
                                                               eval_results['train']['ndcg@10'][150],
                                                               eval_results['valid']['ndcg@10'][150],
                                                               eval_results['test']['ndcg@10'][150]))
        print('{}, all trees: {:.4f} - {:.4f} - {:.4f}'.format(model_name,
                                                               eval_results['train']['ndcg@10'][-1],
                                                               eval_results['valid']['ndcg@10'][-1],
                                                               eval_results['test']['ndcg@10'][-1]))


def plot_doc_score():
    """
    Plots a graph showing the variation in document scores while training the model
    """
    pass
