from modules.dataloader_utils import *
from modules.eval_utils import *
from modules.train import *
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import os
import re
import sys
import glob
import random
from torch.utils.data import DataLoader
import torch.nn as nn
import json

with open('test_docs_2019.pkl', 'rb') as f:
    test_docs = pickle.load(f)

model = torch.load('bert2019_full_dset_ngrams.pth')
path = '/ssd_scratch/cvit/dhawals1939/'
filename = path+'test_docs_2019_ngrams_withoutsep_num_0.pkl'
new_test_docs = {}
for j, k in enumerate(test_docs.keys()):
    queries = test_docs[k]['cite_text']
    corpus = test_docs[k]['corpus']
    q_embed_lis = []
    s_embed_lis = []
    for i in range(len(queries)):
        query = queries[i]
        citant_ngrams = get_ngrams(query)
        per_cor_sent_q_embed_lis = []
        per_cor_sent_embed_lis = []
        
#         corpus_embedding = model.encode(corpus)
        for c_sente in corpus:
            rp_ngrams = get_ngrams(c_sente)
            lis = np.intersect1d(list(citant_ngrams), list(rp_ngrams))
            lis = " ".join(lis)
            query_embedding = model.encode(query+' '+lis)
            sent_embedding = model.encode(c_sente+' '+lis)
            per_cor_sent_q_embed_lis.append(query_embedding)
            per_cor_sent_embed_lis.append(sent_embedding)
            
        q_embed_lis.append(per_cor_sent_q_embed_lis)
        s_embed_lis.append(per_cor_sent_embed_lis)

    new_test_docs[k] = test_docs[k]
    new_test_docs[k]['citant_ngram_embed'] = q_embed_lis
    new_test_docs[k]['corpus_ngram_embed'] = s_embed_lis
    if j % 10 == 0:
        filename = path+f'test_docs_2019_ngrams_withoutsep_num_{j}.pkl'
        print(filename)
        with open(filename, 'wb') as handle:
            pickle.dump(new_test_docs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        new_test_docs = {}
    print(k)

filename = path+f'test_docs_2019_ngrams_withoutsep_num_{j+1}.pkl'
print(new_test_docs)
print(filename)
with open(filename, 'wb') as handle:
    pickle.dump(new_test_docs, handle, protocol=pickle.HIGHEST_PROTOCOL)