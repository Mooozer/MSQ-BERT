# !/usr/bin/env python
# coding: utf-8

import re
import pandas as pd
import numpy as np

def mapping_cID_with_fID(QuAD, tokenizer):
    '''
    since some long content will be splited into more than 1 parts, here we record content id and its feature id(s)
    input: QuAD: new QuAD dataset with new_id
           tokenizer. 
    output: list 
    '''
    validation_features = QuAD.map(lambda x: prepare_validation_features(x, tokenizer),
                                             batched=True, remove_columns=QuAD.column_names)
    validation_features.set_format(type=validation_features.format["type"],  
                                        columns=list(validation_features.features.keys()))
    
    example_id_to_index = {k: i for i, k in enumerate(QuAD["new_id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(validation_features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)
        
    return features_per_example


from sklearn.preprocessing import normalize
def normalize_alone_feature(score_matrix):
    '''
    input: score_matrix (#question,  #token)
    output: normalized score_matrix (#question,  #token)
    '''
    norm_score_matrix  =  score_matrix - score_matrix.min(1).reshape(-1,1)
    norm_score_matrix  =  norm_score_matrix/(score_matrix.max(1).reshape(-1,1) - score_matrix.min(1).reshape(-1,1))
    return norm_score_matrix  

import nltk
from nltk.corpus import stopwords
Stopwords = stopwords.words('english')
def Freq_score(passage, question_list):
    '''
    input:  passage: string
            question_list: list of strings
    output: freq_score (#question, 1) 
    '''    
    f_score = [1] + [0.05]*(len(question_list)-1) #initialize     
    for q in range(len(question_list)):
        question = question_list[q]
        question_word = question.lower().split(' ')
        f_score[q] += np.mean([[w for w in passage.lower().split(' ') if w not in Stopwords].count(qw) 
                               for qw in question_word if qw not in Stopwords])
    
    return np.array(f_score).reshape(-1,1)

def sigmoid(matrix):
    '''
    input: matrix (m,n)
    output: sigmoid(matrix) = element-wise 1/(1+e^(-x))
    '''
    return 1/(1+np.exp(-matrix))

    
def SVD_approx(matrix, rank):
    '''
    input: matrix (m,n), rank
    output: SVD matrix approximation (m,n)
    '''
    approx_matrix = np.zeros(matrix.shape)
    U, S, V = np.linalg.svd(matrix) 
    for r in range(rank):
        approx_matrix = approx_matrix + S[r] * np.outer(U[:,r], np.transpose(V[r,:]))
    return approx_matrix

from scipy.special import softmax
def softmax_fun(matrix):
    '''
    input: matrix (m,n), rank
    output: matrix (m,n), softmax for each row 
    '''
    return softmax(matrix, axis=1)

from collections import Counter
def multi_score_2_single_score(new_SQuAD, multi_raw_pred, Q_index_aug_split_dic, rank):
    '''
    input: 
        new_SQuAD: multiple questions SQuAD with 'new_id'
        multi_raw_pred: (2, #all questions, 384) = (Start/end, #rows in new_SQuAD, 384)
        rank: rank in SVD approximation
    output:
        scores in 'single' format (2, #original questions, 384) = (Start/end, #original, 384)
    '''
    rangeIndex= np.cumsum([0]+[v['aug'] * v['split'] for v in Q_index_aug_split_dic.values()])
    singleQ_format_start_score = np.empty((0,384))     #store the s/e scores obtained by MSQ method
    singleQ_format_end_score = np.empty((0,384))
    originalQ_format_start_score = np.empty((0,384)) #store the s/e scores obtained by original score 
    originalQ_format_end_score = np.empty((0,384)) 
    j , aug_num = 0, 0 #record question number, and augumentation questions 
    matrix_for_plot_s = [[],[],[],[],[],[],[]] 
    matrix_for_plot_e = [[],[],[],[],[],[],[]] 
    for i,v in Q_index_aug_split_dic.items(): #for i-th original question  
        for s in range(v['split']):
            #Get MSQ scores
            er = int((rangeIndex[j+1]-rangeIndex[j])/(v['split'])) #each split range length
            start_score_matrix_qi = multi_raw_pred[0][rangeIndex[j]+s*er : rangeIndex[j]+(s+1)*er]
            end_score_matrix_qi = multi_raw_pred[1][rangeIndex[j]+s*er : rangeIndex[j]+(s+1)*er]
            matrix_for_plot_s[0].append(start_score_matrix_qi)
            matrix_for_plot_e[0].append(end_score_matrix_qi)            

            #Get the original question scores: 
            originalQ_format_start_score = np.append(originalQ_format_start_score, [start_score_matrix_qi[0]],axis=0) 
            originalQ_format_end_score = np.append(originalQ_format_end_score, [end_score_matrix_qi[0]],axis=0) 


            #Softmax:
            start_score_matrix_qi = softmax_fun(start_score_matrix_qi)
            end_score_matrix_qi = softmax_fun(end_score_matrix_qi)
            matrix_for_plot_s[1].append(start_score_matrix_qi)
            matrix_for_plot_e[1].append(end_score_matrix_qi)            

            #normailze: 
            start_score_matrix_qi = normalize_alone_feature(start_score_matrix_qi)
            end_score_matrix_qi = normalize_alone_feature(end_score_matrix_qi)
            matrix_for_plot_s[2].append(start_score_matrix_qi)
            matrix_for_plot_e[2].append(end_score_matrix_qi)                        

            #sigmoid
            start_score_matrix_qi = sigmoid(start_score_matrix_qi)
            end_score_matrix_qi = sigmoid(end_score_matrix_qi)
            matrix_for_plot_s[3].append(start_score_matrix_qi)
            matrix_for_plot_e[3].append(end_score_matrix_qi)                        
 
            #fs
            passage = new_SQuAD[aug_num]['context']
            question_list = [new_SQuAD[u]['question'] for u in range(aug_num, aug_num+v['aug'])]
            fs = Freq_score(passage, question_list) 
            start_score_matrix_qi = start_score_matrix_qi * fs
            end_score_matrix_qi = end_score_matrix_qi* fs
            matrix_for_plot_s[4].append(start_score_matrix_qi)
            matrix_for_plot_e[4].append(end_score_matrix_qi)   
            
            #Softmax:
            start_score_matrix_qi = softmax_fun(start_score_matrix_qi)
            end_score_matrix_qi = softmax_fun(end_score_matrix_qi)
            matrix_for_plot_s[5].append(start_score_matrix_qi)
            matrix_for_plot_e[5].append(end_score_matrix_qi)   
            
            # SVD: 
            start_score_matrix_qi = SVD_approx(start_score_matrix_qi,rank=1)
            end_score_matrix_qi = SVD_approx(end_score_matrix_qi,rank=1)
            matrix_for_plot_s[6].append(start_score_matrix_qi)
            matrix_for_plot_e[6].append(end_score_matrix_qi)   
            matrix_for_plot = matrix_for_plot_s, matrix_for_plot_e

            #MSQ Single score by mean: 
            start_score_matrix_qi = np.mean(start_score_matrix_qi, axis = 0)
            end_score_matrix_qi = np.mean(end_score_matrix_qi, axis = 0)

            #Get the MSQ scores: 
            singleQ_format_start_score = np.append(singleQ_format_start_score, [start_score_matrix_qi],axis=0) 
            singleQ_format_end_score = np.append(singleQ_format_end_score, [end_score_matrix_qi],axis=0) 
        j+=1 
        aug_num += v['aug']
        
    MSQ_scores = np.array([singleQ_format_start_score, singleQ_format_end_score])
    ori_scores = np.array([originalQ_format_start_score, originalQ_format_end_score])
    return ori_scores, MSQ_scores, matrix_for_plot
            