# Product Search track at TREC 2024

The code and results are for TREC 2024 Product Search track ([link](https://trec-product-search.github.io/index.html)).

## Code
*indexing.py* to create different indexes.\
*retrieval.py* to retrieve and save in trec format.

## Results
Please find the results below in descending infAP values.


|  Method              |   infAP                  |   ndcg_cut_10            |   ndcg_cut_100           |   recall_10              |   recall_100             |
|:---------------------|-------------------------:|-------------------------:|-------------------------:|-------------------------:|-------------------------:|
| BM25-QE              |                   0.5501 |                   0.633  |                   0.6965 |                   0.0609 |                   0.4043 |
| BM25QE-TAS-B-fusion  |                   0.5391 |                   0.6671 |                   0.6938 |                   0.0637 |                   0.3953 |
| BM25-TAS-B-fusion    |                   0.5351 |                   0.6734 |                   0.6931 |                   0.0683 |                   0.3931 |
| Rerank               |                   0.5343 |                   0.6308 |                   0.6645 |                   0.06   |                   0.3866 |
| BM25QE-SPLADE-fusion |                   0.5282 |                   0.6707 |                   0.6791 |                   0.066  |                   0.3761 |
| BM25                 |                   0.5259 |                   0.6463 |                   0.6936 |                   0.065  |                   0.3939 |
| BM25-SPLADE-fusion   |                   0.5147 |                   0.6625 |                   0.6724 |                   0.0622 |                   0.3694 |
| TAS-B                |                   0.4452 |                   0.5254 |                   0.5692 |                   0.0495 |                   0.3412 |
| SPLADE               |                   0.4008 |                   0.5365 |                   0.5246 |                   0.0416 |                   0.2944 |
