# !/usr/bin/env python
# coding: utf-8

import re
import pandas as pd
import numpy as np
from datasets import Dataset

def padded_question(question_list, tokenizer):
    '''
    input: question list, tokenizer
           tokenizer
    output: padded question list
    '''
    QList = [question_list[i].strip() for i in range(len(question_list))] #lower case, remove last space    
    QList_token_len = [len(tokenizer(QList)['input_ids'][i]) for i in range(len(QList))]
    QList_token_len_max = max(QList_token_len)
    QList_padded = [["[PAD] " * (QList_token_len_max - QList_token_len[i]) + QList[i]] for i in range(len(QList))]
    return QList_padded


wiki_contents_file = pd.read_csv("./Wiki_ST-QuAAD/wiki_contents.csv")
wiki_answers_file1 = pd.read_csv("./Wiki_ST-QuAAD/wiki_answers1.csv")
wiki_answers_file2 = pd.read_csv("./Wiki_ST-QuAAD/wiki_answers2.csv")
wiki_answers_file3 = pd.read_csv("./Wiki_ST-QuAAD/wiki_answers3.csv")
wiki_aug_ques_list = pd.read_csv("./Wiki_ST-QuAAD/Ques_argumentAA.csv") 

def generate_new_Wiki_QUAD(tokenizer):
    '''
    Pad questions using tokenizer, then generate the new Wiki_QUAD
    '''
    Symptoms_words = ['Fever', 'Cough', 'Shortness of breath',  'Myalgia', 'Headache', 'Anosmia', 
                  'Sore throat', 'Nasal congestion', 'Rhinorrhea', 'Nausea', 'Vomiting', 'Diarrhea',
                  'Abdominal pain','Blood in stool','Chest pain','Constipation','Dysphagia',
                  'Palpitations','Knee pain','Low back pain','Neck pain','Paresthesia', 'Rash','Hemoptysis',
                  'Pneumonia','Delayed onset muscle soreness','Back pain','Xerostomia','Dry eye syndrome',
                  'Insomnia','Sleep deprivation','Cyanosis','Somnolence','Heartburn','Tremor','Chronic pain'] 
    Symptoms = ['Fever', 'Cough', 'Shortness_of_breath',  'Myalgia', 'Headache', 'Anosmia', 
            'Sore_throat', 'Nasal_congestion', 'Rhinorrhea', 'Nausea', 'Vomiting', 'Diarrhea',
           'Abdominal_pain','Blood_in_stool','Chest_pain','Constipation','Dysphagia',
           'Palpitations','Knee_pain','Low_back_pain','Neck_pain','Paresthesia','Rash','Hemoptysis',
            'Pneumonia','Delayed_onset_muscle_soreness','Back_pain','Xerostomia','Dry_eye_syndrome',
           'Insomnia','Sleep_deprivation','Cyanosis','Somnolence','Heartburn','Tremor','Chronic_pain']
    
    vaild_qs = [q.lower() for q in wiki_aug_ques_list['Ques_argument'] if 'aa' in q.lower()]
    aug_qs_temp = list(dict.fromkeys(vaild_qs)) #keep unique and keep order 
    aug_qs_temp = padded_question(aug_qs_temp, tokenizer)  #pad questions
    aug_qs_temp = sum(aug_qs_temp, []) #convert to a single list   
    
    Wiki_data_dic = {'answers':[], 'context':[], "id":[], "question":[], "title":[], "new_id":[]}
    passage_id = 0 
    
    for s in range(len(Symptoms)): 
        sym = Symptoms[s]
        tem_content =  wiki_contents_file[sym].dropna().tolist()
        tem_answers1 =  wiki_answers_file1[sym].dropna().tolist()
        tem_answers2 =  wiki_answers_file2[sym].dropna().tolist()
        tem_answers3 =  wiki_answers_file3[sym].dropna().tolist()
        for c in range(len(tem_content)):
            content_c = tem_content[c]
            
            for q in range(len(aug_qs_temp)):
                
                Wiki_data_dic['context'] += [str(tem_content[c])]
                
                Wiki_data_dic['question'] += [aug_qs_temp[q].replace("aa", Symptoms_words[s].lower())]
                
                ans_dic = {'answer_start': sum([[i for i in range(len(content_c)) if content_c.startswith(a, i)] 
                                                 for a in [tem_answers1[c],tem_answers2[c],tem_answers3[c]] if a],[]),
                                   'text': sum([[a for i in range(len(content_c)) if content_c.startswith(a, i)] 
                                                 for a in [tem_answers1[c],tem_answers2[c],tem_answers3[c]] if a],[])}
                Wiki_data_dic['answers'] += [ans_dic]

                Wiki_data_dic['id'] += [sym+str(c)+str(q)]
                Wiki_data_dic['title'] += [sym+str(c)+str(q)]
                Wiki_data_dic['new_id'] += [str(passage_id)+"_"+str(q)]
            passage_id+=1 
    new_Wiki_QUAD = Dataset.from_dict(Wiki_data_dic)
    return new_Wiki_QUAD 



def prepare_validation_features(examples, tokenizer):
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["new_id"][sample_index])

        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]
    return tokenized_examples


import torch
import collections
from transformers import BertForQuestionAnswering,BertTokenizer,BertModel,AutoTokenizer # AdamW, BertConfig
from datasets import Dataset, load_dataset, load_metric
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
tokenizer_biobert_large = AutoTokenizer.from_pretrained('dmis-lab/biobert-large-cased-v1.1-squad')


from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import default_data_collator

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
max_length = 384 # The maximum length of a feature (question and context)
doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
pad_on_right = tokenizer.padding_side == "right"
new_Wiki_STQuAAD = generate_new_Wiki_QUAD(tokenizer)

print('---------- predict ST-QuAAD raw scores with BERT large ----------',)
bert_large_finetuned_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
data_collator = default_data_collator

new_Wiki_STQuAAD_features =  new_Wiki_STQuAAD.map(
    lambda x: prepare_validation_features(x, tokenizer = tokenizer),
    batched=True,
    remove_columns=new_Wiki_STQuAAD.column_names) 

trainer_BERTlarge = Trainer(
    bert_large_finetuned_model,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

raw_pred_BERTlarge_STQuAAD = trainer_BERTlarge.predict(new_Wiki_STQuAAD_features) 
np.save('./Wiki_ST-QuAAD/raw_pred_BERTlarge_STQuAAD.npy',raw_pred_BERTlarge_STQuAAD.predictions) 



from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import default_data_collator
tokenizer_biobert_large = AutoTokenizer.from_pretrained('dmis-lab/biobert-large-cased-v1.1-squad')

max_length = 384 # The maximum length of a feature (question and context)
doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
pad_on_right = tokenizer_biobert_large.padding_side == "right"
new_Wiki_STQuAAD = generate_new_Wiki_QUAD(tokenizer_biobert_large)


print('---------- predict ST-QuAAD raw scores with BioBERT large ----------',)
biobert_large_finetuned_model = BertForQuestionAnswering.from_pretrained('dmis-lab/biobert-large-cased-v1.1-squad')
data_collator = default_data_collator

new_Wiki_STQuAAD_features =  new_Wiki_STQuAAD.map(
    lambda x: prepare_validation_features(x, tokenizer = tokenizer_biobert_large),
    batched=True,
    remove_columns=new_Wiki_STQuAAD.column_names) 


trainer_BioBERTlarge = Trainer(
    biobert_large_finetuned_model,
    data_collator=data_collator,
    tokenizer=tokenizer_biobert_large,
)

raw_pred_BioBERTlarge_STQuAAD = trainer_BioBERTlarge.predict(new_Wiki_STQuAAD_features)
np.save('./Wiki_ST-QuAAD/raw_pred_BioBERTlarge_STQuAAD.npy',raw_pred_BioBERTlarge_STQuAAD.predictions) 
    






