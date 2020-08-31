import lightgbm as lgbm

from .LGBMSelGB import LGBMSelGB
from .utils import compare_model_error, Timeit, load_data

print('START')
# prepare dataset
base_dir = "../datasets/istella-short/sample"
train_file = base_dir + "/train.txt"
valid_file = base_dir + "/vali.txt"
test_file = base_dir + "/test.txt"

train_data, train_labels, train_query_lens = load_data(train_file)
print("Loaded training set")

valid_data, valid_labels, valid_query_lens = load_data(valid_file)
print('Validation set loaded')

test_data, test_labels, test_query_lens = load_data(test_file)
print('Testing set loaded')

train_set = lgbm.Dataset(train_data, label=train_labels, group=train_query_lens)
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
    lgb_model = lgbm.train(params, train_set, num_boost_round=200,
                          valid_sets=valid_sets, valid_names=eval_names,
                          verbose_eval=10, evals_result=evals_result)
    return evals_result


lgb_info = train_lgbm_model()

strategies = ('fixed', 'random_iter', 'random_query', 'inverse',
              'wrong_neg', 'equal_size', 'delta')

selgb1 = LGBMSelGB(n_estimators=200, n_iter_sample=1, max_p=0.2, method='random_iter')
selgb1.fit(train_data, train_labels, train_query_lens,
           eval_set=eval_set, eval_group=eval_group, eval_names=eval_names,
           verbose=10)

selgb2 = LGBMSelGB(n_estimators=200, n_iter_sample=1, max_p=0.2, method='random_query')
selgb2.fit(train_data, train_labels, train_query_lens,
           eval_set=eval_set, eval_group=eval_group, eval_names=eval_names,
           verbose=10)

compare_model_error(data=[lgb_info, selgb1.get_eval_result(), selgb2.get_eval_result()],
                    names=['LightGBM', 'Selgb rnd iter', 'Selgb rnd query'], plot=True)
