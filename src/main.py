from modules.dataloader_utils import *
from modules.eval_utils import *
from modules.train import *
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from torch.utils.data import DataLoader
import torch.nn as nn
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import gensim.downloader as api


parser = ArgumentParser()
parser.add_argument("--dataset", default="2018_19", help="dataset")
parser.add_argument("--dataset_path", default="data", help="dataset")

parser.set_defaults(verbose=False)
opt = parser.parse_args()

if opt.dataset=='2018_19':
    #2018
    docs = os.listdir(opt.dataset_path+"/scisumm-2018/Training")
    train_rps = docs
    train_data, train_docs = get_dataset_2018(train_rps,opt.dataset_path+"/scisumm-2018/Training/")
    test_if_valid_data(train_docs)

    test_path = opt.dataset_path+"/scisumm-2018/Test/"
    test_rps = os.listdir(test_path)
    test_docs = get_Test_dataset_2018(test_rps,test_path)
    test_if_valid_data(test_docs)

    #2019
    docs = os.listdir(opt.dataset_path+"/From-ScisummNet-2019")
    docs = sorted(docs)
    train_end_idx = int(len(docs)*0.75)
    train_rps = docs[0:train_end_idx]
    test_rps = docs[train_end_idx:]
    train_data2, train_docs2 = get_dataset_2019(train_rps,opt.dataset_path+"/From-ScisummNet-2019/")
    test_if_valid_data(train_docs2)

    test_data2, test_docs2 = get_dataset_2019(test_rps,opt.dataset_path+"/From-ScisummNet-2019/")
    test_if_valid_data(test_docs2)
    for docid in test_docs2:
        test_docs2[docid]['ref_off'] = dict(zip(np.arange(len(test_docs2[docid]['ref_off'])),test_docs2[docid]['ref_off']))    

    #appending 2018_19 datasets for combined training and testing
    train_data = train_data+train_data2
    train_docs.update(train_docs2)
    test_docs.update(test_docs2)

elif opt.dataset=='2018':
    docs = os.listdir(opt.dataset_path+"/scisumm-2018/Training")
    train_rps = docs
    train_data, train_docs = get_dataset_2018(train_rps,opt.dataset_path+"/scisumm-2018/Training/")
    
    test_path = opt.dataset_path+"/scisumm-2018/Test/"
    test_rps = os.listdir(test_path)
    test_docs = get_Test_dataset_2018(test_rps,test_path)
    test_if_valid_data(test_docs)
    
elif opt.dataset=='2019':
    docs = os.listdir(opt.dataset_path+"/From-ScisummNet-2019")
    docs = sorted(docs)
    train_end_idx = int(len(docs)*0.75)
    train_rps = docs[0:train_end_idx]
    test_rps = docs[train_end_idx:]
    train_data, train_docs = get_dataset_2019(train_rps,opt.dataset_path+"/From-ScisummNet-2019/")
    
    test_data, test_docs = get_dataset_2019(test_rps,opt.dataset_path+"/From-ScisummNet-2019/")
    test_if_valid_data(test_docs)

    for docid in test_docs:
        test_docs[docid]['ref_off'] = dict(zip(np.arange(len(test_docs[docid]['ref_off'])),test_docs2[docid]['ref_off']))    
    
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)


#Define the model. Either from scratch of by loading a pre-trained model
model = SentenceTransformer('bert-base-nli-mean-tokens')
train_loss = losses.CosineSimilarityLoss(model)
#Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3, warmup_steps=100)
torch.save(model,'bert2018_19_full_dset.pth')


# train_docs = get_embed_docs(train_docs,model)
# with open('train_docs_'+opt.dataset+'.pkl', 'wb') as handle:
#     pickle.dump(train_docs, handle, protocol=pickle.HIGHEST_PROTOCOL)

# test_docs = get_embed_docs(test_docs,model)
# with open('test_docs_'+opt.dataset+'.pkl', 'wb') as handle:
#     pickle.dump(test_docs, handle, protocol=pickle.HIGHEST_PROTOCOL)


model_wmd = api.load('word2vec-google-news-300') 
#change topk parameter and others like n-grams etc based on your need
recall,precision,f1, rouge1, rouge2, rouge_su4 = evaluate(model,test_docs,model_wmd,topk=10)   
print("metrics obtained in train: recall {}, precision {}, f1-score {}, rouge1 {}, rouge2 {}, rouge_su4 {}".format(recall,precision,f1, rouge1, rouge2, rouge_su4))
