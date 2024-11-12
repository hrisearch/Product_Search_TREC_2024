import csv
import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_pisa import PisaIndex
from pyterrier.measures import *
import pyterrier_dr
from pyterrier_dr import FlexIndex, TctColBert, TasB
import pyt_splade

# with open("trec-data/collection.trec") as file:
#     tsv_file = csv.reader(file, delimiter="\t")
#     for line in tsv_file:
#         print(line)
#         exit()

def generate_trec():
    with open("trec-data/collection.trec", 'r', encoding="UTF8") as file:
        tsv_file = csv.reader((line.replace('\x00', '') for line in file), delimiter="\t")

        for line in tsv_file:
            if len(line) == 3:
                yield {'docno': line[0], 'text': line[1] + ' ' + line[2]}
            elif len(line) == 2:
                yield {'docno': line[0], 'text': line[1]}
            elif len(line) > 3:
                print('error')

# trecgen = generate_trec()
# i = 0
# for p in trecgen:
#     i += 1
#     #trecgen.close()
# print(i)
# exit()

bm25indexname = './trec.lexical'
iter_indexer = pt.IterDictIndexer(bm25indexname, meta={'docno': 48, 'text': 8192})
iter_indexer.index(generate_trec())

model = TasB.dot()
idx = FlexIndex('trec.tasb.flex')
pipeline = model.doc_encoder() >> idx
pipeline.index(generate_trec())


model = TctColBert('castorini/tct_colbert-v2-hnp-msmarco')
idx = FlexIndex('trec.tct.flex')
pipeline = model.doc_encoder() >> idx
pipeline.index(generate_trec())


indexer = pt.IterDictIndexer('./trec.splade', meta={'docno': 48, 'text': 100000})
#indexer.setProperty("termpipelines", "")
#indexer.setProperty("tokeniser", "WhitespaceTokeniser")
factory = pyt_splade.SpladeFactory()
doc_encoder = factory.indexing()

indxr_pipe = (doc_encoder >> pyt_splade.toks2doc() >> indexer)
index_ref = indxr_pipe.index(generate_trec(), batch_size=16)
