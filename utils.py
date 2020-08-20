import matplotlib.pyplot as plt

def timeit(fun):
    # decorator to log training time
    def timed(*args, **kwargs):
        from time import time
        start = time()
        result = fun(*args, **kwargs)
        end = time()
        print('Execution took {:.2f} min'.format((end-start)/60))


def compare_model_error(lgb_info, selgb_info, savefig=False):
    '''
    :param lgb_info:
    :param selgb_info:
    :param savefig:
    :return:
    '''
    plt.figure()  # tight_layout=True)
    plt.plot(lgb_info['train']['ndcg@10'], label='lgbm training')
    plt.plot(lgb_info['valid']['ndcg@10'], label='lgbm validation')
    plt.plot(lgb_info['test']['ndcg@10'], label='lgbm testing')
    plt.plot(selgb_info['train']['ndcg@10'], label='selgb training')
    plt.plot(selgb_info['valid']['ndcg@10'], label='selgb validation')
    plt.plot(selgb_info['test']['ndcg@10'], label='selgb testing')
    plt.grid()
    plt.legend()
    plt.xlabel('#Trees')
    plt.ylabel('ndcg@10')
    plt.title('Model error')
    if savefig:
        plt.savefig('foo.png')
    plt.show()


def plot_doc_score_training():
    '''
    Plots a graph showing the variation in document scores while training the model
    '''
    pass
