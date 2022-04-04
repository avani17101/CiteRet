from rouge_score import rouge_scorer  #https://pypi.org/project/rouge-score/
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
from pythonrouge.pythonrouge import Pythonrouge
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import scipy
from nltk import ngrams
from rouge_score import rouge_scorer  #https://pypi.org/project/rouge-score/
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
from pythonrouge.pythonrouge import Pythonrouge
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import scipy
from nltk import ngrams

def get_ngrams(sent):
    return list(ngrams(sent.split(), 2))

def get_cs_and_wmd_based_matching_sentences(model,model_wmd,corpus,queries,gt, topk=None):
    #f1 score will be same as precision and recall in our case: since documents
    # Get a vector for each headline (sentence) in the corpus
    # Define search queries and embed them to vectors as well
    
    query_embeddings = model.encode(queries)
    # For each search term return 3 closest sentences
    total_nums_correct = 0
    total_retrieved = 0
    total_relevent = 0
    rouge1 = []
    rouge2 = []
    rouge_su4 = []
    
    for i in range(len(queries)):
        query = queries[i]
        distances = []
        wmd_distances = []
        for c_sente in corpus:
            query_embedding = model.encode(query)
            rp_embedding = model.encode(c_sente)
            distances.append(scipy.spatial.distance.cdist([query_embedding], [rp_embedding], "cosine")[0])
            wmd_distances.append(model_wmd.wmdistance(query, c_sente))

        distances = np.array(distances)
        wmd_distances = np.array(wmd_distances)
        wmd_distances = wmd_distances.reshape((wmd_distances.shape[0],1))
        distance_cosine_wmd =  distances + wmd_distances
        results = zip(range(len(distance_cosine_wmd)), distance_cosine_wmd)
        results = sorted(results, key=lambda x: x[1]) 

        retrieved = []
        indexes = results[0:topk]
        retrieved = []
        for l,k in indexes:
            retrieved.append(l)
        if i in gt:
            nums_correct = len(np.intersect1d(retrieved,list(gt[i])))
            total_nums_correct += nums_correct
            total_relevent += len(gt[i])
            total_retrieved += len(retrieved)
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
            pass

            
    return total_nums_correct, total_relevent,total_retrieved, np.mean(rouge1), np.mean(rouge2), np.mean(rouge_su4)

def get_cs_based_matching_sentences(model,corpus,queries,gt, thresh=0.6,topk=3,cs_thresh_based=True,ngrams=False):
    #f1 score will be same as precision and recall in our case: since documents
    # Get a vector for each headline (sentence) in the corpus
    # Define search queries and embed them to vectors as well
    
    query_embeddings = model.encode(queries)
    # For each search term return 3 closest sentences
    total_nums_correct = 0
    total_retrieved = 0
    total_relevent = 0
    rouge1 = []
    rouge2 = []
    rouge_su4 = []
    
    for i in range(len(queries)):
        query = queries[i]
        citant_ngrams = get_ngrams(query)
        distances = []
        for c_sente in corpus:
            if ngrams==True:
                rp_ngrams = get_ngrams(c_sente)
                lis = np.intersect1d(list(citant_ngrams), list(rp_ngrams))
                lis = " ".join(lis)
                query_embedding = model.encode(query+lis)
                rp_embedding = model.encode(c_sente)
                distances.append(scipy.spatial.distance.cdist([query_embedding], [rp_embedding], "cosine")[0])
            else:
                query_embedding = model.encode(query)
                rp_embedding = model.encode(c_sente)
                distances.append(scipy.spatial.distance.cdist([query_embedding], [rp_embedding], "cosine")[0])

        distances = np.array(distances)
        distances = 1- distances
        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: -x[1])

        retrieved = []
        if cs_thresh_based:
            for k,dist in results:
                if dist >= thresh:
                    retrieved.append(k)
        else: #retrieve topk most matching sentences
            indexes = results[0:topk]
            retrieved = []
            for l,k in indexes:
                retrieved.append(l)
        if i in gt:
            nums_correct = len(np.intersect1d(retrieved,list(gt[i])))
            total_nums_correct += nums_correct
            total_relevent += len(gt[i])
            total_retrieved += len(retrieved)
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
            pass

            
    return total_nums_correct, total_relevent,total_retrieved, np.mean(rouge1), np.mean(rouge2), np.mean(rouge_su4)

def get_ngrams(sent):
    return list(ngrams(sent.split(), 2))

def get_cs_and_wmd_based_matching_sentences(model,model_wmd,corpus,queries,gt, topk=3):
    #f1 score will be same as precision and recall in our case: since documents
    # Get a vector for each headline (sentence) in the corpus
    # Define search queries and embed them to vectors as well
    
    query_embeddings = model.encode(queries)
    # For each search term return 3 closest sentences
    total_nums_correct = 0
    total_retrieved = 0
    total_relevent = 0
    rouge1 = []
    rouge2 = []
    rouge_su4 = []
    
    for i in range(len(queries)):
        query = queries[i]
        distances = []
        wmd_distances = []
        for c_sente in corpus:
            query_embedding = model.encode(query)
            rp_embedding = model.encode(c_sente)
            distances.append(scipy.spatial.distance.cdist([query_embedding], [rp_embedding], "cosine")[0])
            wmd_distances.append(model_wmd.wmdistance(query, c_sente))

        distances = np.array(distances)
        wmd_distances = np.array(wmd_distances)
        wmd_distances = wmd_distances.reshape((wmd_distances.shape[0],1))
        distance_cosine_wmd =  distances + wmd_distances
        results = zip(range(len(distance_cosine_wmd)), distance_cosine_wmd)
        results = sorted(results, key=lambda x: x[1]) 

        retrieved = []
        
        indexes = results[0:topk]
        retrieved = []
        for l,k in indexes:
            retrieved.append(l)
        
        nums_correct = len(np.intersect1d(retrieved,list(gt[i])))
        total_nums_correct += nums_correct
        total_relevent += len(gt[i])
        total_retrieved += len(retrieved)

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

def get_cs_based_matching_sentences(model,corpus,queries,gt, thresh=None,topk=None,ngrams=False):
    #f1 score will be same as precision and recall in our case: since documents
    # Get a vector for each headline (sentence) in the corpus
    # Define search queries and embed them to vectors as well
    
    query_embeddings = model.encode(queries)
    # For each search term return 3 closest sentences
    total_nums_correct = 0
    total_retrieved = 0
    total_relevent = 0
    rouge1 = []
    rouge2 = []
    rouge_su4 = []
    
    for i in range(len(queries)):
        query = queries[i]
        citant_ngrams = get_ngrams(query)
        distances = []
        for c_sente in corpus:
            if ngrams==True:
                print("+ngrams")
                rp_ngrams = get_ngrams(c_sente)
                lis = np.intersect1d(list(citant_ngrams), list(rp_ngrams))
                lis = " ".join(lis)
                query_embedding = model.encode(query+lis)
                rp_embedding = model.encode(c_sente)
                distances.append(scipy.spatial.distance.cdist([query_embedding], [rp_embedding], "cosine")[0])
            else:
                query_embedding = model.encode(query)
                rp_embedding = model.encode(c_sente)
                distances.append(scipy.spatial.distance.cdist([query_embedding], [rp_embedding], "cosine")[0])

        distances = np.array(distances)
        distances = 1- distances
        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: -x[1])

        retrieved = []
        if thresh:
            print("retrieve based on thresh")
            for k,dist in results:
                if dist >= thresh:
                    retrieved.append(k)
        elif topk: #retrieve topk most matching sentences
            print("retrieve top ",topk)
            indexes = results[0:topk]
            retrieved = []
            for l,k in indexes:
                retrieved.append(l)
        
        nums_correct = len(np.intersect1d(retrieved,list(gt[i])))
        total_nums_correct += nums_correct
        total_relevent += len(gt[i])
        total_retrieved += len(retrieved)
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

os.environ['TOKENIZERS_PARALLELISM']= '0'
def evaluate(model, docs, model_wmd=None,thresh=None,topk=None,ngrams=False):
    '''
    args: model, docs, thresh=0.6,topk=3,cs_thresh_based=True,cs_based=True,ngrams=False
    returns: recall,precision,f1, rouge1, rouge2,rouge_su4
    '''
    rouge1_lis = []
    rouge2_lis = []
    rouge_su4_lis = []
    tp_big = 0
    tot_relevent = 0
    tot_retrieved = 0

    for k in docs:
        if model_wmd == None:
            print("doing normal cs")
            tp, rele,ret, rouge1, rouge2, rouge_su4 = get_cs_based_matching_sentences(model,docs[k]['corpus'], docs[k]['cite_text'], docs[k]['ref_off'],thresh,topk,ngrams)
        else: #cs+wmd
            print("cs+wmd")
            tp, rele,ret, rouge1, rouge2, rouge_su4 = get_cs_and_wmd_based_matching_sentences(model,model_wmd,docs[k]['corpus'], docs[k]['cite_text'], docs[k]['ref_off'],topk)
        print("doc_id",k,tp, rele,ret, rouge1, rouge2, rouge_su4)
        
        tp_big += tp
        tot_relevent += rele
        tot_retrieved += ret
        rouge1_lis.append(rouge1)
        rouge2_lis.append(rouge2)
        rouge_su4_lis.append(rouge_su4)
        
    recall = tp_big/tot_relevent
    precision = tp_big/tot_retrieved
    f1 = 2*recall*precision/(recall+precision+1e-10)
    return recall,precision,f1, np.mean(rouge1_lis), np.mean(rouge2_lis), np.mean(rouge_su4_lis)