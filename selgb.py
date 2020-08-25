from itertools import groupby
from sklearn.datasets import load_svmlight_file
import lightgbm as lgb

from LGBMSelGB import LGBMSelGB
from utils import compare_model_error, Timeit

print('START')
# prepare dataset
base_dir = "../datasets/istella-short/sample"
train_file = base_dir + "/train.txt"
valid_file = base_dir + "/vali.txt"
test_file = base_dir + "/test.txt"

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
    lgb_model = lgb.train(params, train_set, num_boost_round=200,
                          valid_sets=valid_sets, valid_names=eval_names,
                          verbose_eval=10, evals_result=evals_result)
    return evals_result


lgb_info = train_lgbm_model()

strategies = ('fixed', 'random_iter', 'random_query', 'inverse',
              'wrong_neg', 'equal_size', 'delta')

selgb1 = LGBMSelGB(n_estimators=200, n_iter_sample=10, delta=2.5, delta_pos=5, method='delta')
selgb1.fit(train_data, train_labels, train_query_lens,
           eval_set=eval_set, eval_group=eval_group, eval_names=eval_names,
           verbose=10)

selgb2 = LGBMSelGB(n_estimators=200, n_iter_sample=10, method='equal_size')
selgb2.fit(train_data, train_labels, train_query_lens,
           eval_set=eval_set, eval_group=eval_group, eval_names=eval_names,
           verbose=10)

compare_model_error(data=[lgb_info, selgb1.get_evals_result(), selgb2.get_evals_result()],
                    names=['LightGBM', 'Selgb delta', 'Selgb eq_size'], plot=True)
