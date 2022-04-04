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
from rouge_score import rouge_scorer  #https://pypi.org/project/rouge-score/
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
from pythonrouge.pythonrouge import Pythonrouge
    
import scipy
from scipy.spatial import distance
from pickle5 import pickle
import time
from argparse import ArgumentParser
import gensim.downloader as api


def cosine_distance_for_ngram(a,b):
    ''' Here in the dictionary, each document is a key, and in that there are two n_gram related embeddings.
    The way they are shaped is : number_of_citant x number_of_sentences_in_corpus x embedding_length.
    citant[i] + intersection_of_citant[i]_with_corpus_sentence[j] , ie for each citant, 
    we found ngram_intersection with each sentence in the corpus is stored in citant_ngra_embed.
    So for each citant, there are 'n' number of embeddings, wehere n = number_of_sentences_in_corpus .
    Similarly for
    each sentence in the corpus also has 'n' number of different embeddings, where n = number_of_citant
    '''
    a_unit = np.array(a)/np.linalg.norm(a,axis = -1).reshape(a.shape[0],a.shape[1],1)
    b_unit = np.array(b)/np.linalg.norm(b,axis = -1).reshape(b.shape[0],b.shape[1],1)
    mul = a_unit*b_unit
    sum_mul = np.sum(mul, axis =-1)
      
    return 1-np.array(sum_mul) #cs_dist

def get_res2018(results, docid,doc,thresh=None, topk=None,verbose=False):
    gt = doc['ref_off']
    retrieved = []
    total_nums_correct = 0
    total_retrieved = 0
    total_relevent = 0
    rouge1 = []
    rouge2 = []
    rouge_su4 = []
    
    for i in range(len(doc['cite_text'])):
        if i in gt:
            if thresh:
                for k,dist in results:

                    if dist >= thresh:
                        retrieved.append(k)
            elif topk: #retrieve topk most matching sentences
                retrieved = results[i][:topk]
            nums_correct = len(np.intersect1d(retrieved,list(gt[i])))
            total_nums_correct += nums_correct
            total_relevent += len(gt[i])
            total_retrieved += len(retrieved)
            query = doc['cite_text'][i]
            corpus = None
            if isinstance(corpus,pd.Series):
                corpus = doc['corpus'].values
            else:
                corpus = doc['corpus']

            
            for idx in retrieved:
                scores = scorer.score(corpus[idx].strip(), query)
                rouge = Pythonrouge(summary_file_exist=False,
                        summary=[[corpus[idx].strip()]], reference=[[[query]]],
                        n_gram=2, ROUGE_SU4=True, ROUGE_L=False,
                        recall_only=False, stemming=True, stopwords=True,
                        word_level=True, length_limit=True, length=50,
                        use_cf=False, cf=95, scoring_formula='average',
                        resampling=True, samples=1000, favor=True, p=0.5)

                score = rouge.calc_score()
                rouge1.append(score['ROUGE-1-F'])
                rouge2.append(score['ROUGE-2-F'])
                rouge_su4.append(score['ROUGE-SU4-F'])
        else:
            if verbose:
                print(f"not having gt of {i} for doc{docid}" )
            continue

            
    return total_nums_correct, total_relevent,total_retrieved, np.mean(rouge1), np.mean(rouge2), np.mean(rouge_su4)


def get_res(results, docid,doc,thresh=None, topk=None, dataset='2018',verbose=False):
    if dataset=='2018' or dataset=='2018_19':
        return get_res2018(results, docid,doc,thresh=thresh, topk=topk,verbose=verbose)
    else:
        gt = doc['ref_off']
        retrieved = []
        total_nums_correct = 0
        total_retrieved = 0
        total_relevent = 0
        rouge1 = []
        rouge2 = []
        rouge_su4 = []
        
        for i in range(len(gt)):
            if thresh:
                for k,dist in results:

                    if dist >= thresh:
                        retrieved.append(k)
            elif topk: #retrieve topk most matching sentences
                retrieved = results[i][:topk]
            nums_correct = len(np.intersect1d(retrieved,list(gt[i])))
            total_nums_correct += nums_correct
            total_relevent += len(gt[i])
            total_retrieved += len(retrieved)
            query = doc['cite_text'][i]
            corpus = None
            if isinstance(corpus,pd.Series):
                corpus = doc['corpus'].values
            else:
                corpus = doc['corpus']

            gt = doc['ref_off']
            for idx in retrieved:
                scores = scorer.score(corpus[idx].strip(), query)
                rouge = Pythonrouge(summary_file_exist=False,
                        summary=[[corpus[idx].strip()]], reference=[[[query]]],
                        n_gram=2, ROUGE_SU4=True, ROUGE_L=False,
                        recall_only=False, stemming=True, stopwords=True,
                        word_level=True, length_limit=True, length=50,
                        use_cf=False, cf=95, scoring_formula='average',
                        resampling=True, samples=1000, favor=True, p=0.5)

                score = rouge.calc_score()
                rouge1.append(score['ROUGE-1-F'])
                rouge2.append(score['ROUGE-2-F'])
                rouge_su4.append(score['ROUGE-SU4-F'])

                
        return total_nums_correct, total_relevent,total_retrieved, np.mean(rouge1), np.mean(rouge2), np.mean(rouge_su4)

def get_results(updated_test, wmd,topk,dataset='2018'):

    rouge1_lis = []
    rouge2_lis = []
    rouge_su4_lis = []
    tp_big = 0
    tot_relevent = 0
    tot_retrieved = 0

    rouge1_lis_cosine_pca3 = []
    rouge2_lis_cosine_pca3 = []
    rouge_su4_lis_cosine_pca3 = []
    tp_big_cosine_pca3 = 0
    tot_relevent_cosine_pca3 = 0
    tot_retrieved_cosine_pca3 = 0

    rouge1_lis_cosine_pca5 = []
    rouge2_lis_cosine_pca5 = []
    rouge_su4_lis_cosine_pca5 = []
    tp_big_cosine_pca5 = 0
    tot_relevent_cosine_pca5 = 0
    tot_retrieved_cosine_pca5 = 0

    rouge1_lis_wmd = []
    rouge2_lis_wmd = []
    rouge_su4_lis_wmd = []
    tp_big_wmd = 0
    tot_relevent_wmd = 0
    tot_retrieved_wmd = 0

    rouge1_lis_wmd_normal = []
    rouge2_lis_wmd_normal = []
    rouge_su4_lis_wmd_normal = []
    tp_big_wmd_normal = 0
    tot_relevent_wmd_normal = 0
    tot_retrieved_wmd_normal = 0

    rouge1_lis_wmd_pca3 = []
    rouge2_lis_wmd_pca3 = []
    rouge_su4_lis_wmd_pca3 = []
    tp_big_wmd_pca3 = 0
    tot_relevent_wmd_pca3 = 0
    tot_retrieved_wmd_pca3 = 0

    rouge1_lis_wmd_pca5 = []
    rouge2_lis_wmd_pca5 = []
    rouge_su4_lis_wmd_pca5 = []
    tp_big_wmd_pca5 = 0
    tot_relevent_wmd_pca5 = 0
    tot_retrieved_wmd_pca5 = 0

    
    tic = time.time()
    for docid in updated_test:

        cosine_distances_normal = scipy.spatial.distance.cdist(updated_test[docid]['cite_text_embed'], updated_test[docid]['corpus_embed'], "cosine")
        ## with pca removed embeddings finding the distances
        cosine_distance_pca3 = scipy.spatial.distance.cdist(updated_test[docid]['citant_pca_removed3_embed'], updated_test[docid]['corpus_pca_removed3_embed'], "cosine")
        cosine_distance_pca5 = scipy.spatial.distance.cdist(updated_test[docid]['citant_pca_removed5_embed'], updated_test[docid]['corpus_pca_removed5_embed'], "cosine")
        distance_wmd = np.array(wmd[docid])
        cosine_wmd_normal = cosine_distances_normal + distance_wmd
        cosine_wmd_pca3 = cosine_distance_pca3 + distance_wmd
        cosine_wmd_pca5 = cosine_distance_pca5 + distance_wmd

        sorted_indx_cosine = np.argsort(cosine_distances_normal, axis = -1)
        sorted_indx_cosine_pca3 = np.argsort(cosine_distance_pca3, axis = -1)
        sorted_indx_cosine_pca5 = np.argsort(cosine_distance_pca5, axis = -1)
        sorted_indx_wmd = np.argsort(distance_wmd, axis = -1)
        sorted_indx_cs_wmd_normal = np.argsort(cosine_wmd_normal, axis = -1)
        sorted_indx_cs_wmd_pca3 = np.argsort(cosine_wmd_pca3, axis = -1)
        sorted_indx_cs_wmd_pca5 = np.argsort(cosine_wmd_pca5, axis = -1)

        tp, rele,ret, rouge1, rouge2, rouge_su4 = get_res(sorted_indx_cosine,docid,updated_test[docid],topk=topk, dataset=dataset,verbose=True)
        print("\ntime",time.time()-tic)
        print("\ndoc_id",docid)
        tp_big += tp
        tot_relevent += rele
        tot_retrieved += ret
        rouge1_lis.append(rouge1)
        rouge2_lis.append(rouge2)
        rouge_su4_lis.append(rouge_su4)

        tp, rele,ret, rouge1, rouge2, rouge_su4 = get_res(sorted_indx_cosine_pca3,docid,updated_test[docid],topk=topk, dataset=dataset)
        tp_big_cosine_pca3 += tp
        tot_relevent_cosine_pca3 += rele
        tot_retrieved_cosine_pca3 += ret
        rouge1_lis_cosine_pca3.append(rouge1)
        rouge2_lis_cosine_pca3.append(rouge2)
        rouge_su4_lis_cosine_pca3.append(rouge_su4)

        tp, rele,ret, rouge1, rouge2, rouge_su4 = get_res(sorted_indx_cosine_pca5,docid,updated_test[docid],topk=topk, dataset=dataset)
        tp_big_cosine_pca5 += tp
        tot_relevent_cosine_pca5 += rele
        tot_retrieved_cosine_pca5 += ret
        rouge1_lis_cosine_pca5.append(rouge1)
        rouge2_lis_cosine_pca5.append(rouge2)
        rouge_su4_lis_cosine_pca5.append(rouge_su4)


        tp, rele,ret, rouge1, rouge2, rouge_su4 = get_res(sorted_indx_wmd,docid,updated_test[docid],topk=topk, dataset=dataset)
        tp_big_wmd += tp
        tot_relevent_wmd += rele
        tot_retrieved_wmd += ret
        rouge1_lis_wmd.append(rouge1)
        rouge2_lis_wmd.append(rouge2)
        rouge_su4_lis_wmd.append(rouge_su4)

        tp, rele,ret, rouge1, rouge2, rouge_su4 = get_res(sorted_indx_cs_wmd_normal,docid,updated_test[docid],topk=topk, dataset=dataset)
        tp_big_wmd_normal += tp
        tot_relevent_wmd_normal += rele
        tot_retrieved_wmd_normal += ret
        rouge1_lis_wmd_normal.append(rouge1)
        rouge2_lis_wmd_normal.append(rouge2)
        rouge_su4_lis_wmd_normal.append(rouge_su4)

        tp, rele,ret, rouge1, rouge2, rouge_su4 = get_res(sorted_indx_cs_wmd_pca3,docid,updated_test[docid],topk=topk, dataset=dataset)
        tp_big_wmd_pca3 += tp
        tot_relevent_wmd_pca3 += rele
        tot_retrieved_wmd_pca3 += ret
        rouge1_lis_wmd_pca3.append(rouge1)
        rouge2_lis_wmd_pca3.append(rouge2)
        rouge_su4_lis_wmd_pca3.append(rouge_su4)

        tp, rele,ret, rouge1, rouge2, rouge_su4 = get_res(sorted_indx_cs_wmd_pca5,docid,updated_test[docid],topk=topk, dataset=dataset)
        tp_big_wmd_pca5 += tp
        tot_relevent_wmd_pca5 += rele
        tot_retrieved_wmd_pca5 += ret
        rouge1_lis_wmd_pca5.append(rouge1)
        rouge2_lis_wmd_pca5.append(rouge2)
        rouge_su4_lis_wmd_pca5.append(rouge_su4)

    recall = tp_big/tot_relevent
    precision = tp_big/tot_retrieved
    f1 = 2*recall*precision/(recall+precision+1e-10)

    recall_cosine_pca3 = tp_big_cosine_pca3/tot_relevent_cosine_pca3
    precision_cosine_pca3 = tp_big_cosine_pca3/tot_retrieved_cosine_pca3
    f1_cosine_pca3 = 2*recall_cosine_pca3*precision_cosine_pca3/(recall_cosine_pca3+precision_cosine_pca3+1e-10)

    recall_cosine_pca5 = tp_big_cosine_pca5/tot_relevent_cosine_pca5
    precision_cosine_pca5 = tp_big_cosine_pca5/tot_retrieved_cosine_pca5
    f1_cosine_pca5 = 2*recall_cosine_pca5*precision_cosine_pca5/(recall_cosine_pca5+precision_cosine_pca5+1e-10)

    recall_wmd = tp_big_wmd/tot_relevent_wmd
    precision_wmd = tp_big_wmd/tot_retrieved_wmd
    f1_wmd = 2*recall_wmd*precision_wmd/(recall_wmd+precision_wmd+1e-10)

    recall_wmd_normal = tp_big_wmd_normal/tot_relevent_wmd_normal
    precision_wmd_normal = tp_big_wmd_normal/tot_retrieved_wmd_normal
    f1_wmd_normal = 2*recall_wmd_normal*precision_wmd_normal/(recall_wmd_normal+precision_wmd_normal+1e-10)

    recall_wmd_pca3 = tp_big_wmd_pca3/tot_relevent_wmd_pca3
    precision_wmd_pca3 = tp_big_wmd_pca3/tot_retrieved_wmd_pca3
    f1_wmd_pca3 = 2*recall_wmd_pca3*precision_wmd_pca3/(recall_wmd_pca3+precision_wmd_pca3+1e-10)

    recall_wmd_pca5 = tp_big_wmd_pca5/tot_relevent_wmd_pca5
    precision_wmd_pca5 = tp_big_wmd_pca5/tot_retrieved_wmd_pca5
    f1_wmd_pca5 = 2*recall_wmd_pca5*precision_wmd_pca5/(recall_wmd_pca5+precision_wmd_pca5+1e-10)

  
    return {
        'cosine':[recall,precision,f1, np.mean(rouge1_lis), np.mean(rouge2_lis), np.mean(rouge_su4_lis)],
        'cosine_pca3':[recall_cosine_pca3,precision_cosine_pca3,f1_cosine_pca3, np.mean(rouge1_lis_cosine_pca3), np.mean(rouge2_lis_cosine_pca3), np.mean(rouge_su4_lis_cosine_pca3)],
        'cosine_pca5':[recall_cosine_pca5,precision_cosine_pca5,f1_cosine_pca5, np.mean(rouge1_lis_cosine_pca5), np.mean(rouge2_lis_cosine_pca5), np.mean(rouge_su4_lis_cosine_pca5)],
        'wmd':[recall_wmd,precision_wmd,f1_wmd, np.mean(rouge1_lis_wmd), np.mean(rouge2_lis_wmd), np.mean(rouge_su4_lis_wmd)],
        'wmd_normal':[recall_wmd_normal,precision_wmd_normal,f1_wmd_normal, np.mean(rouge1_lis_wmd_normal), np.mean(rouge2_lis_wmd_normal), np.mean(rouge_su4_lis_wmd_normal)],
        'wmd_pca3':[recall_wmd_pca3,precision_wmd_pca3,f1_wmd_pca3, np.mean(rouge1_lis_wmd_pca3), np.mean(rouge2_lis_wmd_pca3), np.mean(rouge_su4_lis_wmd_pca3)],
        'wmd_pca5':[recall_wmd_pca5,precision_wmd_pca5,f1_wmd_pca5, np.mean(rouge1_lis_wmd_pca5), np.mean(rouge2_lis_wmd_pca5), np.mean(rouge_su4_lis_wmd_pca5)]
    }

def calc_wmd(test_docs, dset_name):
    model_wmd = api.load('word2vec-google-news-300')    
    wmd = {} 
    for k in test_docs:
        queries = test_docs[k]['cite_text']
        corpus = test_docs[k]['corpus']
        q_wise_wmd = []
        print(k)
        for i in range(len(queries)):
            query = queries[i]
            wmd_distances = []
            for c_sente in corpus:
                wmd_distances.append(model_wmd.wmdistance(query, c_sente))
            q_wise_wmd.append(wmd_distances)
        wmd[k] = q_wise_wmd

    with open('wmd_'+dset_name+'.pkl', 'wb') as handle:
        pickle.dump(wmd, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_results_ngram(updated_test, wmd,topk,dataset='2018'):

    rouge1_lis = []
    rouge2_lis = []
    rouge_su4_lis = []
    tp_big = 0
    tot_relevent = 0
    tot_retrieved = 0

    rouge1_lis_cosine_pca3 = []
    rouge2_lis_cosine_pca3 = []
    rouge_su4_lis_cosine_pca3 = []
    tp_big_cosine_pca3 = 0
    tot_relevent_cosine_pca3 = 0
    tot_retrieved_cosine_pca3 = 0

    rouge1_lis_cosine_pca5 = []
    rouge2_lis_cosine_pca5 = []
    rouge_su4_lis_cosine_pca5 = []
    tp_big_cosine_pca5 = 0
    tot_relevent_cosine_pca5 = 0
    tot_retrieved_cosine_pca5 = 0

    rouge1_lis_wmd = []
    rouge2_lis_wmd = []
    rouge_su4_lis_wmd = []
    tp_big_wmd = 0
    tot_relevent_wmd = 0
    tot_retrieved_wmd = 0

    rouge1_lis_wmd_normal = []
    rouge2_lis_wmd_normal = []
    rouge_su4_lis_wmd_normal = []
    tp_big_wmd_normal = 0
    tot_relevent_wmd_normal = 0
    tot_retrieved_wmd_normal = 0

    rouge1_lis_wmd_pca3 = []
    rouge2_lis_wmd_pca3 = []
    rouge_su4_lis_wmd_pca3 = []
    tp_big_wmd_pca3 = 0
    tot_relevent_wmd_pca3 = 0
    tot_retrieved_wmd_pca3 = 0

    rouge1_lis_wmd_pca5 = []
    rouge2_lis_wmd_pca5 = []
    rouge_su4_lis_wmd_pca5 = []
    tp_big_wmd_pca5 = 0
    tot_relevent_wmd_pca5 = 0
    tot_retrieved_wmd_pca5 = 0

    
    tic = time.time()
    for docid in updated_test:
        a=np.array(updated_test[docid]['citant_ngram_embed'])
        b=np.array(updated_test[docid]['corpus_ngram_embed'])
        cosine_distances_normal = cosine_distance_for_ngram(a,b)

        ## with pca removed embeddings finding the distances
        a=np.array(updated_test[docid]['citant_ngram_pca_removed3_embed'])
        b=np.array(updated_test[docid]['corpus_ngram_pca_removed3_embed'])
        cosine_distance_pca3 = cosine_distance_for_ngram(a,b)
        
        a=np.array(updated_test[docid]['citant_ngram_pca_removed5_embed'])
        b=np.array(updated_test[docid]['corpus_ngram_pca_removed5_embed'])
        cosine_distance_pca5 = cosine_distance_for_ngram(a,b)

        distance_wmd = np.array(wmd[docid])
        cosine_wmd_normal = cosine_distances_normal + distance_wmd
        cosine_wmd_pca3 = cosine_distance_pca3 + distance_wmd
        cosine_wmd_pca5 = cosine_distance_pca5 + distance_wmd

        sorted_indx_cosine = np.argsort(cosine_distances_normal, axis = -1)
        sorted_indx_cosine_pca3 = np.argsort(cosine_distance_pca3, axis = -1)
        sorted_indx_cosine_pca5 = np.argsort(cosine_distance_pca5, axis = -1)
        sorted_indx_wmd = np.argsort(distance_wmd, axis = -1)
        sorted_indx_cs_wmd_normal = np.argsort(cosine_wmd_normal, axis = -1)
        sorted_indx_cs_wmd_pca3 = np.argsort(cosine_wmd_pca3, axis = -1)
        sorted_indx_cs_wmd_pca5 = np.argsort(cosine_wmd_pca5, axis = -1)

        tp, rele,ret, rouge1, rouge2, rouge_su4 = get_res(sorted_indx_cosine,docid,updated_test[docid],topk=topk, dataset=dataset,verbose=True)
        print("\ntime",time.time()-tic)
        print("\ndoc_id",docid)
        tp_big += tp
        tot_relevent += rele
        tot_retrieved += ret
        rouge1_lis.append(rouge1)
        rouge2_lis.append(rouge2)
        rouge_su4_lis.append(rouge_su4)

        tp, rele,ret, rouge1, rouge2, rouge_su4 = get_res(sorted_indx_cosine_pca3,docid,updated_test[docid],topk=topk, dataset=dataset)
        tp_big_cosine_pca3 += tp
        tot_relevent_cosine_pca3 += rele
        tot_retrieved_cosine_pca3 += ret
        rouge1_lis_cosine_pca3.append(rouge1)
        rouge2_lis_cosine_pca3.append(rouge2)
        rouge_su4_lis_cosine_pca3.append(rouge_su4)

        tp, rele,ret, rouge1, rouge2, rouge_su4 = get_res(sorted_indx_cosine_pca5,docid,updated_test[docid],topk=topk, dataset=dataset)
        tp_big_cosine_pca5 += tp
        tot_relevent_cosine_pca5 += rele
        tot_retrieved_cosine_pca5 += ret
        rouge1_lis_cosine_pca5.append(rouge1)
        rouge2_lis_cosine_pca5.append(rouge2)
        rouge_su4_lis_cosine_pca5.append(rouge_su4)


        tp, rele,ret, rouge1, rouge2, rouge_su4 = get_res(sorted_indx_wmd,docid,updated_test[docid],topk=topk, dataset=dataset)
        tp_big_wmd += tp
        tot_relevent_wmd += rele
        tot_retrieved_wmd += ret
        rouge1_lis_wmd.append(rouge1)
        rouge2_lis_wmd.append(rouge2)
        rouge_su4_lis_wmd.append(rouge_su4)

        tp, rele,ret, rouge1, rouge2, rouge_su4 = get_res(sorted_indx_cs_wmd_normal,docid,updated_test[docid],topk=topk, dataset=dataset)
        tp_big_wmd_normal += tp
        tot_relevent_wmd_normal += rele
        tot_retrieved_wmd_normal += ret
        rouge1_lis_wmd_normal.append(rouge1)
        rouge2_lis_wmd_normal.append(rouge2)
        rouge_su4_lis_wmd_normal.append(rouge_su4)

        tp, rele,ret, rouge1, rouge2, rouge_su4 = get_res(sorted_indx_cs_wmd_pca3,docid,updated_test[docid],topk=topk, dataset=dataset)
        tp_big_wmd_pca3 += tp
        tot_relevent_wmd_pca3 += rele
        tot_retrieved_wmd_pca3 += ret
        rouge1_lis_wmd_pca3.append(rouge1)
        rouge2_lis_wmd_pca3.append(rouge2)
        rouge_su4_lis_wmd_pca3.append(rouge_su4)

        tp, rele,ret, rouge1, rouge2, rouge_su4 = get_res(sorted_indx_cs_wmd_pca5,docid,updated_test[docid],topk=topk, dataset=dataset)
        tp_big_wmd_pca5 += tp
        tot_relevent_wmd_pca5 += rele
        tot_retrieved_wmd_pca5 += ret
        rouge1_lis_wmd_pca5.append(rouge1)
        rouge2_lis_wmd_pca5.append(rouge2)
        rouge_su4_lis_wmd_pca5.append(rouge_su4)

    recall = tp_big/tot_relevent
    precision = tp_big/tot_retrieved
    f1 = 2*recall*precision/(recall+precision+1e-10)

    recall_cosine_pca3 = tp_big_cosine_pca3/tot_relevent_cosine_pca3
    precision_cosine_pca3 = tp_big_cosine_pca3/tot_retrieved_cosine_pca3
    f1_cosine_pca3 = 2*recall_cosine_pca3*precision_cosine_pca3/(recall_cosine_pca3+precision_cosine_pca3+1e-10)

    recall_cosine_pca5 = tp_big_cosine_pca5/tot_relevent_cosine_pca5
    precision_cosine_pca5 = tp_big_cosine_pca5/tot_retrieved_cosine_pca5
    f1_cosine_pca5 = 2*recall_cosine_pca5*precision_cosine_pca5/(recall_cosine_pca5+precision_cosine_pca5+1e-10)

    recall_wmd = tp_big_wmd/tot_relevent_wmd
    precision_wmd = tp_big_wmd/tot_retrieved_wmd
    f1_wmd = 2*recall_wmd*precision_wmd/(recall_wmd+precision_wmd+1e-10)

    recall_wmd_normal = tp_big_wmd_normal/tot_relevent_wmd_normal
    precision_wmd_normal = tp_big_wmd_normal/tot_retrieved_wmd_normal
    f1_wmd_normal = 2*recall_wmd_normal*precision_wmd_normal/(recall_wmd_normal+precision_wmd_normal+1e-10)

    recall_wmd_pca3 = tp_big_wmd_pca3/tot_relevent_wmd_pca3
    precision_wmd_pca3 = tp_big_wmd_pca3/tot_retrieved_wmd_pca3
    f1_wmd_pca3 = 2*recall_wmd_pca3*precision_wmd_pca3/(recall_wmd_pca3+precision_wmd_pca3+1e-10)

    recall_wmd_pca5 = tp_big_wmd_pca5/tot_relevent_wmd_pca5
    precision_wmd_pca5 = tp_big_wmd_pca5/tot_retrieved_wmd_pca5
    f1_wmd_pca5 = 2*recall_wmd_pca5*precision_wmd_pca5/(recall_wmd_pca5+precision_wmd_pca5+1e-10)

  
    return {
        'cosine':[recall,precision,f1, np.mean(rouge1_lis), np.mean(rouge2_lis), np.mean(rouge_su4_lis)],
        'cosine_pca3':[recall_cosine_pca3,precision_cosine_pca3,f1_cosine_pca3, np.mean(rouge1_lis_cosine_pca3), np.mean(rouge2_lis_cosine_pca3), np.mean(rouge_su4_lis_cosine_pca3)],
        'cosine_pca5':[recall_cosine_pca5,precision_cosine_pca5,f1_cosine_pca5, np.mean(rouge1_lis_cosine_pca5), np.mean(rouge2_lis_cosine_pca5), np.mean(rouge_su4_lis_cosine_pca5)],
        'wmd':[recall_wmd,precision_wmd,f1_wmd, np.mean(rouge1_lis_wmd), np.mean(rouge2_lis_wmd), np.mean(rouge_su4_lis_wmd)],
        'wmd_normal':[recall_wmd_normal,precision_wmd_normal,f1_wmd_normal, np.mean(rouge1_lis_wmd_normal), np.mean(rouge2_lis_wmd_normal), np.mean(rouge_su4_lis_wmd_normal)],
        'wmd_pca3':[recall_wmd_pca3,precision_wmd_pca3,f1_wmd_pca3, np.mean(rouge1_lis_wmd_pca3), np.mean(rouge2_lis_wmd_pca3), np.mean(rouge_su4_lis_wmd_pca3)],
        'wmd_pca5':[recall_wmd_pca5,precision_wmd_pca5,f1_wmd_pca5, np.mean(rouge1_lis_wmd_pca5), np.mean(rouge2_lis_wmd_pca5), np.mean(rouge_su4_lis_wmd_pca5)]
    }
