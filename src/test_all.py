## calculates all metrics on various scenarios

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
import glob
import pickle5 as pickle
from pca import create_weightedPCAremoved_ngram

from argparse import ArgumentParser
from util import *


parser = ArgumentParser()
parser.add_argument("--dataset", default="2018_19", help="dataset")
parser.add_argument("--dataset_path", default="data", help="dataset")
parser.add_argument("--ngrams", default=1,type=int, help="type:0 if normal, 1 if with ngrams")
parser.add_argument("--ngrams_without_sep", default=1,type=int, help="whether to use separator while concatenating n-grams to words")
parser.add_argument("--checkpoint_path", default='bert_2018_19_full_dset_ngrams.pth')





parser.set_defaults(verbose=False)
opt = parser.parse_args()
dataset = opt.dataset

with open('test_docs_'+dataset+'.pkl', 'rb') as f:
    test_docs = pickle.load(f)

model = torch.load(opt.checkpoint_path)
path = opt.dataset_path

def generate_embeddings(test_docs,path,ngrams,ngrams_without_sep):
    # save path for embeddings
    if ngrams==1:
        filename = path+'test_docs_'+dataset+'_ngrams_num_0.pkl'
    if ngrams==0:
        filename = path+'test_docs_'+dataset+'.pkl'

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
            
            for c_sente in corpus:
                rp_ngrams = get_ngrams(c_sente)
                lis = np.intersect1d(list(citant_ngrams), list(rp_ngrams))
                lis = " ".join(lis)
                if ngrams==1:
                    if ngrams_without_sep:
                        query_embedding = model.encode(query+' '+lis)
                        sent_embedding = model.encode(c_sente+' '+lis)
                    else:
                        query_embedding = model.encode(query+' sey '+lis)
                        sent_embedding = model.encode(c_sente+' sey '+lis)
                else:
                    query_embedding = model.encode(query)
                    sent_embedding = model.encode(c_sente)

                per_cor_sent_q_embed_lis.append(query_embedding)
                per_cor_sent_embed_lis.append(sent_embedding)
                
            q_embed_lis.append(per_cor_sent_q_embed_lis)
            s_embed_lis.append(per_cor_sent_embed_lis)

        new_test_docs[k] = test_docs[k]
        if ngrams==1:
            new_test_docs[k]['citant_ngram_embed'] = q_embed_lis
            new_test_docs[k]['corpus_ngram_embed'] = s_embed_lis
        else:
            new_test_docs[k]['citant_embed'] = q_embed_lis
            new_test_docs[k]['corpus_embed'] = s_embed_lis

        if j % 10 == 0:
            if ngrams==1:
                filename = path+'test_docs_'+dataset+'_ngrams_num_{}.pkl'.format(j)
            else:
                filename = path+'test_docs_'+dataset+'_num_{}.pkl'.format(j)
            print(filename)
            with open(filename, 'wb') as handle:
                pickle.dump(new_test_docs, handle, protocol=pickle.HIGHEST_PROTOCOL)
            new_test_docs = {}
        print(k)
        
    if ngrams==1:
        filename = path+'test_docs_'+dataset+'_ngrams_num_{}.pkl'.format(j+1)
    else:
        filename = path+'test_docs_'+dataset+'_num_{}.pkl'.format(j+1)

    print(new_test_docs)
    print(filename)

    with open(filename, 'wb') as handle:
        pickle.dump(new_test_docs, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("***********************Generating embeddings from model******************************")
generate_embeddings(test_docs,path,opt.ngrams,opt.ngrams_without_sep)

print("***********************running pca******************************")
a = glob.glob(path+'test_docs_'+dataset+'_ngrams*')

for file in a:
    with open(file,'rb') as f:
        d = pickle.load(f)
    new_file = file.replace('test','updated_test')
    print(new_file)
    create_weightedPCAremoved_ngram(d, des = new_file)

d = {}
a = glob.glob(path+'updated_test_docs_'+dataset+'_ngrams*')
for file in a:
    print(file)
    with open(file,'rb') as f:
        d.update(pickle.load(f))


with open('wmd_'+str(dataset)+'.pkl', 'rb') as fw:
    wmd = pickle.load(fw)

print("********************running inferenceee*********************")
for topk in [10]:
    results = get_results_ngram(d, wmd,topk=topk,dataset = str(dataset))
    with open('results_all_ngrams'+str(dataset)+'_top'+str(topk)+'.pkl', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


