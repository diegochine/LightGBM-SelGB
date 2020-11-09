# LightGBM-SelGB

Implementation of the Selective Gradient Boosting algorithm [[1]](#1), along with several heuristic variants.

## Usage
    python3 train.py variant
where variant is one of: 'lambdarank', 'lgbm goss', 'lgbm base', 'fixed', 'random_iter', 'random_query', 'decay', 'false_positives', 'equal_size', 'delta', 'limit_resample'

The resulting model is saved in output/models, the evaluation (NDCG@10 computed during training) is saved in output/results.


## References
<a id="1">[1]</a> 
Claudio Lucchese, Franco Maria Nardini, Raffaele Perego, Salvatore Orlando, Salvatore Trani. 2018. 
Selective Gradient Boosting for Effective Learning to Rank. 
In Proceedings of the 41th International ACM SIGIR conference on Research and Development in Information Retrieval (SIGIR ’18). ACM, New York, NY, USA. 
DOI: http://dx.doi.org/10.1145/3209978.3210048
