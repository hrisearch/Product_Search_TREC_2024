import csv
import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_pisa import PisaIndex
from pyterrier.measures import *
import pyterrier_dr
from pyterrier_dr import FlexIndex, TctColBert, TasB
from ir_measures import *
import re
import pandas as pd
import pyt_splade

#queries = pd.read_csv("trec-data/2023-test-queries.tsv", delimiter="\t")
#print(queries)


queries = pd.read_csv("trec-data/2024_test_queries.tsv", delimiter="\t")
queries.columns = ['qid', 'query']
for index, row in queries.iterrows():
    queries.at[index, 'query'] = re.sub(r'[^\w\s]', ' ', row['query'])
    queries.at[index, 'qid'] = str(row['qid'])
    # print(row['query'])
    # break
print(queries)
#exit()
qrels = pd.read_csv("trec-data/2024test.qrel", delimiter="\t")
qrels.columns = ['qid', 'asdfasdf', 'docno', 'label']
qrels['docno'] = qrels['docno'].astype(str)
qrels['qid'] = qrels['qid'].astype(str)
print(qrels)

########### BM25

bm25indexname = './trec.lexical'
bm25index = pt.IndexFactory.of(bm25indexname)
bm25 = pt.BatchRetrieve(bm25index, wmodel="BM25")
bm25rr = pt.BatchRetrieve(bm25index, wmodel="BM25", num_results=100)
bm25qe = pt.BatchRetrieve(bm25index, wmodel="BM25", controls={"qe":"on", "qemodel" : "Bo1"})

bm25res = bm25.transform(queries)
bm25res.loc[:, 'docid']='Q0'
bm25res = bm25res.rename(columns={'query': 'runid', 'docid': 'unused'})
bm25res.loc[:, 'runid']='BM25'
print(bm25res)

########### BM25-QE

bm25qeres = bm25qe.transform(queries)
bm25qeres.loc[:, 'docid']='Q0'
bm25qeres = bm25qeres.rename(columns={'query': 'runid', 'docid': 'unused'})
bm25qeres.loc[:, 'runid']='BM25-QE'
print(bm25qeres)

########### Dense PIPELINES

# model = TctColBert('castorini/tct_colbert-v2-hnp-msmarco')
model = TasB.dot()
idx = FlexIndex('trec.tasb.flex')

pipeline = model.query_encoder() >> idx.np_retriever()
rrpipeline = bm25 >> model.query_encoder() >> idx.scorer()
factory = pyt_splade.SpladeFactory()
query_encoder = factory.query()
sppipe = query_encoder >> pt.BatchRetrieve('./trec.splade', wmodel='Tf', num_results=1000, verbose=True)

sfpipea = 3*sppipe + 50*bm25
sfpipe = sfpipea % 1000

dfpipea = pipeline + 2*bm25
dfpipe = dfpipea % 1000

sqefpipea = 3*sppipe + 50*bm25qe
sqefpipe = sqefpipea % 1000

dqefpipea = pipeline + 2*bm25qe
dqefpipe = dqefpipea % 1000

########### TASB

denres = pipeline.transform(queries)
denres.loc[:, 'docid']='Q0'
denres = denres.rename(columns={'query': 'runid', 'docid': 'unused'})
denres.loc[:, 'runid']='TAS-B'
denres = denres.loc[:,['qid', 'unused', 'docno', 'rank', 'score', 'runid']]
print(denres)

########### RERANK

rrres = rrpipeline.transform(queries)
rrres.loc[:, 'docid']='Q0'
rrres = rrres.rename(columns={'query': 'runid', 'docid': 'unused'})
rrres.loc[:, 'runid']='Rerank'
rrres = rrres.loc[:,['qid', 'unused', 'docno', 'rank', 'score', 'runid']]
print(rrres)

########### SPLADE

spres = sppipe.transform(queries)
spres.loc[:, 'docid']='Q0'
spres = spres.rename(columns={'query': 'runid', 'docid': 'unused'})
spres.loc[:, 'runid']='SPLADE++'
spres = spres.loc[:,['qid', 'unused', 'docno', 'rank', 'score', 'runid']]
print(spres)

########### Fusion

########### TAS-B BM25
dfres = dfpipe.transform(queries)
dfres.loc[:, 'docid']='Q0'
dfres = dfres.rename(columns={'query': 'runid', 'docid': 'unused'})
dfres.loc[:, 'runid']='BM25-TAS-B-fusion'
dfres = dfres.loc[:,['qid', 'unused', 'docno', 'rank', 'score', 'runid']]
dfres =  dfres.sort_values(['qid', 'rank'], ascending=[True, True])
print(dfres)

########### SPLADE BM25
sfres = sfpipe.transform(queries)
sfres.loc[:, 'docid']='Q0'
sfres = sfres.rename(columns={'query': 'runid', 'docid': 'unused'})
sfres.loc[:, 'runid']='BM25-SPLADE++-fusion'
sfres = sfres.loc[:,['qid', 'unused', 'docno', 'rank', 'score', 'runid']]
sfres =  sfres.sort_values(['qid', 'rank'], ascending=[True, True])
print(sfres)

########### TAS-B BM25-QE
dqefres = dqefpipe.transform(queries)
dqefres.loc[:, 'docid']='Q0'
dqefres = dqefres.rename(columns={'query': 'runid', 'docid': 'unused'})
dqefres.loc[:, 'runid']='BM25QE-TAS-B-fusion'
dqefres = dqefres.loc[:,['qid', 'unused', 'docno', 'rank', 'score', 'runid']]
dqefres =  dqefres.sort_values(['qid', 'rank'], ascending=[True, True])
print(dqefres)

########### SPLADE BM25-QE
sqefres = sqefpipe.transform(queries)
sqefres.loc[:, 'docid']='Q0'
sqefres = sqefres.rename(columns={'query': 'runid', 'docid': 'unused'})
sqefres.loc[:, 'runid']='BM25QE-SPLADE++-fusion'
sqefres = sqefres.loc[:,['qid', 'unused', 'docno', 'rank', 'score', 'runid']]
sqefres =  sqefres.sort_values(['qid', 'rank'], ascending=[True, True])
print(sqefres)


bm25res.to_csv('24runs/trec24-bm25.tsv', sep='\t', index=False, header=False)
bm25qeres.to_csv('24runs/trec24-bm25qe.tsv', sep='\t', index=False, header=False)
denres.to_csv('24runs/trec24-tasb.tsv', sep='\t', index=False, header=False)
rrres.to_csv('24runs/trec24-rerank.tsv', sep='\t', index=False, header=False)
spres.to_csv('24runs/trec24-splade.tsv', sep='\t', index=False, header=False)
dfres.to_csv('24runs/trec24-tasb-bm25-fusion.tsv', sep='\t', index=False, header=False)
sfres.to_csv('24runs/trec24-splade-bm25-fusion.tsv', sep='\t', index=False, header=False)
dqefres.to_csv('24runs/trec24-tasb-bm25qe-fusion.tsv', sep='\t', index=False, header=False)
sqefres.to_csv('24runs/trec24-splade-bm25qe-fusion.tsv', sep='\t', index=False, header=False)


met = pt.Experiment(
    [bm25res, bm25qeres, denres, rrres, spres, dfres, sfres, dqefres, sqefres],
    queries,
    qrels,
    eval_metrics=[nDCG@10, R@10, R@100, R@1000],
    names=['BM25', 'BM25-QE', 'TAS-B', 'Rerank', 'SPLADE++', 'BM25-TAS-B-fusion', 'BM25-SPLADE++-fusion', 'BM25QE-TAS-B-fusion', 'BM25QE-SPLADE++-fusion']
)

print(met)