## the file path

import pickle, sys
import numpy as np
from scipy.spatial import distance
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD
import gensim.models.keyedvectors as word2vec


    

    
def create_weightedPCAremoved_ngram(dictionary, des):
    
    
    corpus_pca_removed1_embed = []
    corpus_pca_removed3_embed = []
    corpus_pca_removed5_embed = []
    citant_pca_removed1_embed = []
    citant_pca_removed3_embed = []
    citant_pca_removed5_embed = []
    
    count = 0
    
    for keys in dictionary.keys():
#         if count < 2:
            count+=1
            
            ## embedding vectors of corpus and citant sentences
            
            original_corpus_shape = np.array(dictionary[keys]['corpus_ngram_embed']).shape
            original_citant_shape = np.array(dictionary[keys]['citant_ngram_embed']).shape
        
            corpus_sentence_embed_arr = np.array(dictionary[keys]['corpus_ngram_embed']).reshape(-1,768)
            citance_sentence_embed_arr = np.array(dictionary[keys]['citant_ngram_embed']).reshape(-1,768) # 18 x 768
            print("\n count :", count)
            print("\n corpus_sentence_embed_arr shape ", corpus_sentence_embed_arr.shape)
            print("\n citance_sentence_embed_arr shape", citance_sentence_embed_arr.shape)

            svd3_corpus = TruncatedSVD(n_components=3, random_state=0).fit(corpus_sentence_embed_arr)
            svd3_citant = TruncatedSVD(n_components=3, random_state=0).fit(citance_sentence_embed_arr)

            svd5_corpus = TruncatedSVD(n_components=5, random_state=0).fit(corpus_sentence_embed_arr)
            svd5_citant = TruncatedSVD(n_components=5, random_state=0).fit(citance_sentence_embed_arr)

            ## normalised eigen values to find weighted projections on eigen vectors/ principal directions
            svd3_corpus_singular_values = svd3_corpus.singular_values_/svd3_corpus.singular_values_.sum()
            svd5_corpus_singular_values = svd5_corpus.singular_values_/svd5_corpus.singular_values_.sum()

            svd3_citant_singular_values = svd3_citant.singular_values_/svd3_citant.singular_values_.sum()
            svd5_citant_singular_values = svd5_citant.singular_values_/svd5_citant.singular_values_.sum()

            
            ## principal directions for corpus and citant sentences
            svd3_corpus_comp = svd3_corpus.components_ # 3 x 768
            svd5_corpus_comp = svd5_corpus.components_ # 5 x 768

            svd3_citant_comp = svd3_citant.components_
            svd5_citant_comp = svd5_citant.components_

#             print("\n svd3_corpus_comp shape",svd3_corpus_comp.shape)
#             print("\n svd5_corpus_comp shape",svd5_corpus_comp.shape)
#             print("\n svd3_citant_comp shape",svd3_citant_comp.shape)
#             print("\n svd5_citant_comp shape",svd5_citant_comp.shape)

            
            
            ## scaler value of embeddings projected on the principal directions. We will multiply
            ## them with the eigen values and Then we will multiply
            ## it with unit eigen vector/ principal directions to find the common component vectors.
            ## then we will add them all. and subtract from the embeddings.
            
            
            ##scaler values
            svd3_corpus_projections = corpus_sentence_embed_arr@svd3_corpus_comp.T #(203, 3) number of sen x no of pc
            svd5_corpus_projections = corpus_sentence_embed_arr@svd5_corpus_comp.T #(203, 5)

            svd3_citant_projections = citance_sentence_embed_arr@svd3_citant_comp.T
            svd5_citant_projections = citance_sentence_embed_arr@svd5_citant_comp.T

#             print("\n svd3_corpus_projections shape",svd3_corpus_projections.shape)
#             print("\n svd5_corpus_projections shape",svd5_corpus_projections.shape)
#             print("\n svd3_citant_projections shape",svd3_citant_projections.shape)
#             print("\n svd5_citant_projections shape",svd5_citant_projections.shape)

            
            ##sclaer values of projections after multiplying with the eigen values (point wise)
            value_weighted_svd3_corpus_projections = svd3_corpus_singular_values * svd3_corpus_projections
            value_weighted_svd5_corpus_projections = svd5_corpus_singular_values * svd5_corpus_projections
            value_weighted_svd3_citant_projections = svd3_citant_singular_values * svd3_citant_projections
            value_weighted_svd5_citant_projections = svd5_citant_singular_values * svd5_citant_projections
            
#             print("\n value_weighted_svd3_corpus_projections shape",value_weighted_svd3_corpus_projections.shape)
#             print("\n value_weighted_svd5_corpus_projections shape",value_weighted_svd5_corpus_projections.shape)
#             print("\n value_weighted_svd3_citant_projections shape",value_weighted_svd3_citant_projections.shape)
#             print("\n value_weighted_svd5_citant_projections shape",value_weighted_svd5_citant_projections.shape)
   
            
    
            ##reshaping
            new_value_weighted_svd3_corpus_projections = value_weighted_svd3_corpus_projections[:,None,:]
            new_value_weighted_svd5_corpus_projections = value_weighted_svd5_corpus_projections[:,None,:]
            new_value_weighted_svd3_citant_projections = value_weighted_svd3_citant_projections[:,None,:]
            new_value_weighted_svd5_citant_projections = value_weighted_svd5_citant_projections[:,None,:]
            
            
            ##reshaping
            new_svd3_corpus_comp = svd3_corpus_comp.T[None,:]
            new_svd5_corpus_comp = svd5_corpus_comp.T[None,:]
            new_svd3_citant_comp = svd3_citant_comp.T[None,:]
            new_svd5_citant_comp = svd5_citant_comp.T[None,:]
            
#             print("\n new_svd3_corpus_comp shape",new_svd3_corpus_comp.shape)
#             print("\n new_svd5_corpus_comp shape",new_svd5_corpus_comp.shape)
#             print("\n new_svd3_citant_comp shape",new_svd3_citant_comp.shape)
#             print("\n new_svd5_citant_comp shape",new_svd5_citant_comp.shape)
   
            
            
            ## multiplying to get vectors
            weighted_projection_vector_svd3_corpus = new_value_weighted_svd3_corpus_projections*new_svd3_corpus_comp
            weighted_projection_vector_svd5_corpus = new_value_weighted_svd5_corpus_projections*new_svd5_corpus_comp
            weighted_projection_vector_svd3_citant = new_value_weighted_svd3_citant_projections*new_svd3_citant_comp
            weighted_projection_vector_svd5_citant = new_value_weighted_svd5_citant_projections*new_svd5_citant_comp
            
            
            sum_weighted_projection_vector_svd3_corpus = weighted_projection_vector_svd3_corpus.sum(axis=2)
            sum_weighted_projection_vector_svd5_corpus = weighted_projection_vector_svd5_corpus.sum(axis=2)
            sum_weighted_projection_vector_svd3_citant = weighted_projection_vector_svd3_citant.sum(axis=2)
            sum_weighted_projection_vector_svd5_citant = weighted_projection_vector_svd5_citant.sum(axis=2)
            
#             print("\n sum_weighted_projection_vector_svd3_corpus shape",sum_weighted_projection_vector_svd3_corpus.shape)
#             print("\n sum_weighted_projection_vector_svd5_corpus shape",sum_weighted_projection_vector_svd5_corpus.shape)
#             print("\n sum_weighted_projection_vector_svd3_citant shape",sum_weighted_projection_vector_svd3_citant.shape)
#             print("\n sum_weighted_projection_vector_svd5_citant shape",sum_weighted_projection_vector_svd5_citant.shape)
   
            
            
            ## finally removing the common components from the embeddings: 
            
            

            
            corpus_pca_removed3_embed = corpus_sentence_embed_arr - sum_weighted_projection_vector_svd3_corpus
            corpus_pca_removed5_embed = corpus_sentence_embed_arr - sum_weighted_projection_vector_svd5_corpus
            citant_pca_removed3_embed = citance_sentence_embed_arr - sum_weighted_projection_vector_svd3_citant
            citant_pca_removed5_embed = citance_sentence_embed_arr - sum_weighted_projection_vector_svd5_citant
            
            
#             print("\n corpus_pca_removed3_embed shape",corpus_pca_removed3_embed.shape)
#             print("\n corpus_pca_removed5_embed shape",corpus_pca_removed5_embed.shape)
#             print("\n citant_pca_removed3_embed shape",citant_pca_removed3_embed.shape)
#             print("\n citant_pca_removed5_embed shape",citant_pca_removed5_embed.shape)
            
            dictionary[keys]['corpus_ngram_pca_removed3_embed'] = corpus_pca_removed3_embed.reshape(original_corpus_shape)
            dictionary[keys]['corpus_ngram_pca_removed5_embed'] = corpus_pca_removed5_embed.reshape(original_corpus_shape)
            dictionary[keys]['citant_ngram_pca_removed3_embed'] = citant_pca_removed3_embed.reshape(original_citant_shape)
            dictionary[keys]['citant_ngram_pca_removed5_embed'] = citant_pca_removed5_embed.reshape(original_citant_shape)
   
    with open(des, 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
        
        
#         corpus_pca_removed3_embed = 
#         corpus_pca_removed5_embed = 
        
#         citant_pca_removed3_embed =  
#         citant_pca_removed5_embed = 
        
        
        
        
        
     