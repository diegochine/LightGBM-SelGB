from itertools import groupby
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
import lightgbm as lgb

from LGBMSelGB import LGBMSelGB


def compare_model_error(lgb_info, selgb_info, savefig=False):
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


print('start')
# prepare dataset
base_dir = "C:\\Users\\Diego\\Learning\\tesi\\datasets\\istella-short\\sample"
train_file = base_dir + "\\train.txt"
valid_file = base_dir + "\\vali.txt"
test_file = base_dir + "\\test.txt"

train_raw = load_svmlight_file(train_file, query_id=True)
train_data = train_raw[0]
train_labels = train_raw[1]
train_query_lens = [len(list(group)) for key, group in groupby(train_raw[2])]
print("Loaded training set")

valid_raw = load_svmlight_file(valid_file, query_id=True)
valid_data = valid_raw[0]
valid_labels = valid_raw[1]
valid_query_lens = [len(list(group)) for key, group in groupby(valid_raw[2])]
print("Loaded validation set")

test_raw = load_svmlight_file(test_file, query_id=True)
test_data = test_raw[0]
test_labels = test_raw[1]
test_query_lens = [len(list(group)) for key, group in groupby(test_raw[2])]
print("Loaded testing set")

train_set = lgb.Dataset(train_data, label=train_labels, group=train_query_lens)
try:
    eval_set = [(train_data, train_labels),
                (valid_data, valid_labels),
                (test_data, test_labels)]
    eval_group = [train_query_lens, valid_query_lens, test_query_lens]
    eval_names = ['train', 'valid', 'test']
    valid_sets = []
    for i, valid_data in enumerate(eval_set):
        ds = lgb.Dataset(valid_data[0], valid_data[1], group=eval_group[i], reference=train_set)
        valid_sets.append(ds)
except Exception as e:
    print(e)
    eval_set = []
    eval_group = []
    eval_names = []

lgb_info = {}
params = {
    'objective': 'lambdarank',
    'max_position': 10,
    'learning_rate': 0.1,
    'num_leaves': 16,
    'min_data_in_leaf': 5,
    'metric': ['ndcg'],
    'ndcg_eval_at': 10
}

#lgb_model = lgb.train(params, train_set, num_boost_round=50,
 #                     valid_sets=valid_sets, valid_names=eval_names,
  #                    verbose_eval=10, evals_result=lgb_info)


selgb_model = LGBMSelGB(200, 10, 0.1)
print('Starting fitting')
selgb_model.fit(train_data, train_labels, train_query_lens,
                eval_set=eval_set, eval_group=eval_group, eval_names=eval_names,
                verbose=10)

compare_model_error(lgb_info, selgb_model.get_evals_result())
