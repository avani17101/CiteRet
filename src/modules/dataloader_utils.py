from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import os
import re
import sys
import glob
import random
import torch.nn as nn
import xml.etree.ElementTree as ET
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import random
from os import listdir
from os.path import isfile, join
from sklearn import preprocessing
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from transformers import Trainer, TrainingArguments
# from datasets import load_metric
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification
import ast
import scipy
from nltk import ngrams

multi_lb_classi = 0
def get_ngrams(sent):
    return list(ngrams(sent.split(), 2))

def preprocess(example_sent):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(str(example_sent).lower())
    filtered_sentence = [w for w in word_tokens if not w in stop_words and w.isalpha()]
    new = " " 
    a = new.join(filtered_sentence)
    return a

def get_dataset_2019(files, folder,ngrams=False,take_ngrams_prev_doc=False):
    pairs_lis = []
    prev_corpus = []
    docs_wise = {}
    
    for z,f in enumerate(files):
#         print(f)
        cit_text_lis = []
        ref_text_lis = []
        cit_off_lis = []
        ref_off_lis = []
        new_corpus = []

#         try:
#         print(folder+"/"+str(f)+"/citing_sentences.json")
        citants = pd.read_json(folder+"/"+str(f)+"/citing_sentences.json")
        citants = citants[['citance_No','clean_text']]
        queries = list(citants['clean_text'])
        cite_no = list(citants.citance_No)

        a = folder+'/'+f+"/Reference_XML/"+f+".xml"
        tree = ET.parse(a)
        root = tree.getroot()
        final =[]
        for a in root:
            final.append(a.text)
            break

        total = len(root)
        for a in root:
            for b in a:
                if b.text!='':
                    final.append(b.text)

        d={'col1':final}
        rp = pd.DataFrame(data=d)
        corpus = rp.col1
        new_corpus = corpus.apply(lambda x: preprocess(x))

        data = None
        ann = None
        a_folder = folder+'/'+f+"/annotation/"
        file = os.listdir(a_folder)[0]

        ann = folder+'/'+f+"/annotation/"+f+".ann.txt"
        with open(ann,"r") as fi:
            data = fi.read()
        idx = {'Citance Number':0,'Reference Article':1,'Citing Article':2,'Citation Marker Offset':3,'Citation Marker':4,'Citation Offset':5,'Reference Offset':7}
        gt = {}
        lines = data.split('\n')
        for line in lines:
            lis = line.split('|')
            if len(lis)==11:
                cit_no = lis[0].split(':')[1]
                ref_off = lis[7].split(':')[1].strip()
                import ast
                ref_off = ast.literal_eval(ref_off)
                ref_off = list(map(int, ref_off))
                gt[int(cit_no)] = ref_off 
        corpus = rp.col1
        new_corpus = corpus.apply(lambda x: preprocess(x))

        cit_text_lis = []
        for q in queries:
            a = preprocess(q)
            if len(a)>1:
                cit_text_lis.append(a)
        del queries

#         assert(len(cit_text_lis)==len(ref_off_lis))
        ref_off_lis = []
        for i in range(len(cit_text_lis)):
            citant = cit_text_lis[i]
            citant_ngrams = get_ngrams(citant)
            if cite_no[i] in gt:
                ref_off_lis.append(gt[cite_no[i]])
                for j in gt[cite_no[i]]:
                    j = int(j)
                    if j< len(new_corpus):
                        if ngrams==False:
                            cited_text_spans = new_corpus[j]
                            pairs_lis.append(InputExample(texts=[citant,new_corpus[int(j)]],label=1.0)) #positive pairs
                            pairs_lis.append(InputExample(texts=[citant,new_corpus[random.randint(0,len(new_corpus)-1)]],label=0.3)) #negative pairs
                            pairs_lis.append(InputExample(texts=[citant,new_corpus[random.randint(0,len(new_corpus)-1)]],label=0.3))
                            pairs_lis.append(InputExample(texts=[citant,new_corpus[random.randint(0,len(new_corpus)-1)]],label=0.3))
                        else: #ngrams=True
                            cited_text_spans = new_corpus[j]
                            cited_text_ngrams =  get_ngrams(cited_text_spans)
                            lis = np.intersect1d(list(citant_ngrams), list(cited_text_ngrams))
                            lisone = np.intersect1d(citant, list(cited_text_spans))
                            lis = " ".join(lis) + " " + " ".join(lisone)
                            pairs_lis.append(InputExample(texts=[citant+" sey "+lis,new_corpus[int(j)]],label=1.0)) #positive pairs
                            pairs_lis.append(InputExample(texts=[citant+" sey "+lis,new_corpus[random.randint(0,len(new_corpus)-1)]],label=0.3)) #negative pairs
                            pairs_lis.append(InputExample(texts=[citant+" sey "+lis,new_corpus[random.randint(0,len(new_corpus)-1)]],label=0.3))
                            pairs_lis.append(InputExample(texts=[citant+" sey "+lis,new_corpus[random.randint(0,len(new_corpus)-1)]],label=0.3))



        if (z!=0 and len(new_corpus)!=0 and len(prev_corpus)!=0):
            if ngrams==False:
                pairs_lis.append(InputExample(texts = [prev_corpus[random.randint(0,len(prev_corpus)-1)],new_corpus[random.randint(0,len(new_corpus)-1)]],label=0.0))
            else: #ngrams
                if take_ngrams_prev_doc==True:
                    sent1 = prev_corpus[random.randint(0,len(prev_corpus)-1)]
                    sent2 = new_corpus[random.randint(0,len(new_corpus)-1)]
                    lis = np.intersect1d(list(get_ngrams(sent1)), list(get_ngrams(sent2)))
                    lisone = np.intersect1d(sent1, list(sent2))
                    lis = " ".join(lis) + " " + " ".join(lisone)
                    pairs_lis.append(InputExample(texts = [sent1,sent2],label=0.0))
                else:
                    pairs_lis.append(InputExample(texts = [prev_corpus[random.randint(0,len(prev_corpus)-1)],new_corpus[random.randint(0,len(new_corpus)-1)]],label=0.0))
        docs_wise[f] = {'corpus':new_corpus,  'cite_text':cit_text_lis, 'ref_off':ref_off_lis}
        prev_corpus = new_corpus
#         except Exception as e: 
#             print(f,e)
    
    return pairs_lis, docs_wise    
    

def get_dataset_2018(files, folder,ngrams=False,take_ngrams_prev_doc=False):
    pairs_lis = []
    prev_corpus = []
    docs_wise = {}
    
    for z,f in enumerate(files):
        cit_text_lis = []
        ref_text_lis = []
        cit_off_lis = []
        ref_off_lis = []
        new_corpus = []

        try:
            a = folder+f+"/Reference_XML/"+f+".xml"
            tree = ET.parse(a)
            root = tree.getroot()
            final =[]
            for a in root:
                final.append(a.text)
                break
            
            total = len(root)
            for a in root:
                for b in a:
                    if b.text!='':
                        final.append(b.text)
             
            d={'col1':final}
            rp = pd.DataFrame(data=d)
            corpus = rp.col1
            new_corpus = corpus.apply(lambda x: preprocess(x))

            data = None
            ann = None
            a_folder = folder+f+"/annotation/"
            file = os.listdir(a_folder)[0]

            ann = a_folder+file
            with open(ann,"r") as file:
                data = file.read()

                cit_text = re.findall("Citation Text:\s+([^|]*)", data)
                pattern = r'\<.*?\>'
                pattern2 = r'\(.*?\)'
                for c in cit_text:
                    c = re.sub(pattern2,'',re.sub(pattern, '', c))
                    c = preprocess(c)
                    cit_text_lis.append(c)


                ref_text = re.findall("Reference Text:\s+([^|]*)", data)
                pattern = r'\<.*?\>'
                pattern2 = r'\(.*?\)'
                for ref in ref_text:
                    ref = re.sub(pattern2,'',re.sub(pattern, '', ref))
                    ref = preprocess(ref)
                    ref_text_lis.append(ref)


                ref_off = re.findall("Reference Offset:\s+([^|]*)", data)
                for r in ref_off:
                    r = ast.literal_eval(r)
                    ref_off_lis.append(r)


            for i in range(len(cit_text_lis)):
                citant = cit_text_lis[i]
                citant_ngrams = get_ngrams(citant)

                for j in ref_off_lis[i]:
                    j = int(j)
                    if j< len(new_corpus):
                        if ngrams==False:
                            cited_text_spans = new_corpus[j]
                            pairs_lis.append(InputExample(texts=[citant,new_corpus[int(j)]],label=1.0)) #positive pairs
                            pairs_lis.append(InputExample(texts=[citant,new_corpus[random.randint(0,len(new_corpus)-1)]],label=0.3)) #negative pairs
                            pairs_lis.append(InputExample(texts=[citant,new_corpus[random.randint(0,len(new_corpus)-1)]],label=0.3))
                            pairs_lis.append(InputExample(texts=[citant,new_corpus[random.randint(0,len(new_corpus)-1)]],label=0.3))
                        else: #ngrams=True
                            cited_text_spans = new_corpus[j]
                            cited_text_ngrams =  get_ngrams(cited_text_spans)
                            lis = np.intersect1d(list(citant_ngrams), list(cited_text_ngrams))
                            lisone = np.intersect1d(citant, list(cited_text_spans))
                            lis = " ".join(lis) + " " + " ".join(lisone)
                            pairs_lis.append(InputExample(texts=[citant+" sey "+lis,new_corpus[int(j)]],label=1.0)) #positive pairs
                            pairs_lis.append(InputExample(texts=[citant+" sey "+lis,new_corpus[random.randint(0,len(new_corpus)-1)]],label=0.3)) #negative pairs
                            pairs_lis.append(InputExample(texts=[citant+" sey "+lis,new_corpus[random.randint(0,len(new_corpus)-1)]],label=0.3))
                            pairs_lis.append(InputExample(texts=[citant+" sey "+lis,new_corpus[random.randint(0,len(new_corpus)-1)]],label=0.3))
                            
                          

            if (z!=0 and len(new_corpus)!=0 and len(prev_corpus)!=0):
                if ngrams==False:
                    pairs_lis.append(InputExample(texts = [prev_corpus[random.randint(0,len(prev_corpus)-1)],new_corpus[random.randint(0,len(new_corpus)-1)]],label=0.0))
                else: #ngrams
                    if take_ngrams_prev_doc==True:
                        sent1 = prev_corpus[random.randint(0,len(prev_corpus)-1)]
                        sent2 = new_corpus[random.randint(0,len(new_corpus)-1)]
                        lis = np.intersect1d(list(get_ngrams(sent1)), list(get_ngrams(sent2)))
                        lisone = np.intersect1d(sent1, list(sent2))
                        lis = " ".join(lis) + " " + " ".join(lisone)
                        pairs_lis.append(InputExample(texts = [sent1,sent2],label=0.0))
                    else:
                        pairs_lis.append(InputExample(texts = [prev_corpus[random.randint(0,len(prev_corpus)-1)],new_corpus[random.randint(0,len(new_corpus)-1)]],label=0.0))

            docs_wise[f] = {'corpus':new_corpus,  'cite_text':cit_text_lis, 'ref_off':ref_off_lis}
            prev_corpus = new_corpus
        except Exception as e: 
            print(f,e)
    return pairs_lis, docs_wise

def test_if_valid_data(docs):
    for k in docs:
        try:
            assert(len(docs[k]['ref_off']) == len(docs[k]['cite_text']))
        except:
            print("test failed: length of queries and gt not equal")
            return
    print("unit test passed")
    
import pickle
def get_embed_docs(docs,model):
    for doc_id in docs.keys():
        docs[doc_id]['corpus_embed'] = model.encode(docs[doc_id]['corpus'])
        docs[doc_id]['cite_text_embed'] = model.encode(docs[doc_id]['cite_text'])
    return docs

def ref_off(d, names,files,path='/ssd_scratch/cvit/dhawals1939/scisumm-2018/Test-Gold/Task1/'):
    ref_off_union = {}
    for name in names:
        f_n = d+'_'+name+'.csv'
        if f_n in files:
            df = pd.read_csv(path+f_n)
            ref_offs = list(df['Reference Offset'])
            if(df['Reference Offset'].dtype!=np.float64):
                for i,r in enumerate(ref_offs):
                    r = str(r).replace("'","")
                    if ',' in r:
                        r = r.split(',')
                        for a in r:
                            try:
                                if i in ref_off_union:
                                    ref_off_union[i].add(int(a))
                                else:
                                    ref_off_union[i] = {int(a)}
                            except:
                                pass
                    else:
                        try:
                            if i in ref_off_union:
                                ref_off_union[i].add(int(r))
                            else:
                                ref_off_union[i] = {int(r)}
                        except:
                            pass
            else:
                for i,r in enumerate(ref_offs):
                    if i in ref_off_union:
                        try:
                            ref_off_union[i].add(int(r))
                        except:
                            pass
                    else:
                        try:
                            ref_off_union[i] = {int(r)}
                        except:
                            pass
    return ref_off_union            

def get_Test_dataset_2018(files, folder):
    names = ['aakansha','vardha','swastika','sweta']
    docs_wise = {}
    for z,f in enumerate(files):
        cit_text_lis = []
        ref_text_lis = []
        cit_off_lis = []
#         ref_off_lis = []
        new_corpus = []

        a = folder+f+"/Reference_XML/"+f+".xml"
        tree = ET.parse(a)
        root = tree.getroot()
        final =[]
        total = len(root)
        
        title = None
        for a in root:
            title = a.text
            break
            
        final.append(title)   
        for a in root:
            for b in a:
                final.append(b.text)
                
        d={'col1':final}
        rp = pd.DataFrame(data=d)
        corpus = rp.col1
        new_corpus = corpus.apply(lambda x: preprocess(x))

        path = '/ssd_scratch/cvit/dhawals1939/scisumm-2018/Test-Gold/Task1/'
        files = [f_ for f_ in listdir(path) if isfile(join(path, f_))]
        
        fi = None
        df = None
        for name in names:
            fi = f+'_'+name+'.csv'
            if fi in files:
                df = pd.read_csv(path+fi)
                break
        
        cit_text = list(df['Citation Text Clean'])
        pattern = r'\<.*?\>'
        pattern2 = r'\(.*?\)'
        for c in cit_text:
            c = re.sub(pattern2,'',re.sub(pattern, '', c))
            c = preprocess(c)
            cit_text_lis.append(c)


        ref_text = list(df['Reference Text'])
        pattern = r'\<.*?\>'
        pattern2 = r'\(.*?\)'

        for ref in ref_text:
            ref = re.sub(pattern2,'',re.sub(pattern, '', str(ref)))
            ref = preprocess(ref)
            ref_text_lis.append(ref)

        ref_off_lis = ref_off(f,names,files)
      
        docs_wise[f] = {'corpus':new_corpus.values,  'cite_text':cit_text_lis, 'ref_off':ref_off_lis}

            
    return docs_wise
