import os

from src.LGBMSelGB import LGBMSelGB
from src.utils import load_data

def test():
    base_path = os.getcwd()
    datasets_path = os.path.join(base_path, 'datasets')
    istellax_path = os.path.join(datasets_path, "istella-short")
    train_file = os.path.join(istellax_path, "train.txt")
    valid_file = os.path.join(istellax_path, "vali.txt")
    test_file = os.path.join(istellax_path, "test.txt")
    print('Loading data')
    train_data, train_labels, train_query_lens = load_data(train_file)
    print('[INFO] Training set loaded')
    valid_data, valid_labels, valid_query_lens = load_data(valid_file)
    print('[INFO] Validation set loaded')
    test_data, test_labels, test_query_lens = load_data(test_file)
    print('[INFO] Testing set loaded')
    model = LGBMSelGB(n_estimators=100, n_iter_sample=5, max_p=0.05, p=1, max_resample=100, method='fixed')
    print('[INFO] Starting fitting')
    eval_set = [(train_data, train_labels),
                (valid_data, valid_labels),
                (test_data, test_labels)]
    eval_group = [train_query_lens, valid_query_lens, test_query_lens]
    eval_names = ['train', 'valid', 'test']
    model.fit(train_data, train_labels, train_query_lens,
              eval_set=eval_set, eval_group=eval_group, eval_names=eval_names,
              verbose=1, early_stopping_rounds=10)


if __name__ == '__main__':
    test()
