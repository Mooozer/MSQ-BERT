# !/usr/bin/env python
# coding: utf-8

import re
import pandas as pd
import numpy as np
from datasets import Dataset

import torch
import collections
from transformers import BertForQuestionAnswering,BertTokenizer,BertModel,AutoTokenizer # AdamW, BertConfig
from datasets import Dataset, load_dataset, load_metric
tokenizer_biobert_large = AutoTokenizer.from_pretrained('dmis-lab/biobert-large-cased-v1.1-squad')

new_BioASQ_6B_SQuAD = Dataset.from_dict(np.load('./BioASQ/Task6BGoldenEnriched/new_BioASQ_6B_SQuAD_BioBERT.npy',allow_pickle='TRUE').item())
new_BioASQ_7B_SQuAD = Dataset.from_dict(np.load('./BioASQ/Task7BGoldenEnriched/new_BioASQ_7B_SQuAD_BioBERT.npy',allow_pickle='TRUE').item())
new_BioASQ_8B_SQuAD = Dataset.from_dict(np.load('./BioASQ/Task8BGoldenEnriched/new_BioASQ_8B_SQuAD_BioBERT.npy',allow_pickle='TRUE').item())
new_BioASQ_9B_SQuAD = Dataset.from_dict(np.load('./BioASQ/Task9BGoldenEnriched/new_BioASQ_9B_SQuAD_BioBERT.npy',allow_pickle='TRUE').item())



max_length = 384 # The maximum length of a feature (question and context)
doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
pad_on_right = tokenizer_biobert_large.padding_side == "right"

def prepare_validation_features(examples, tokenizer = tokenizer_biobert_large):
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




from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import default_data_collator

max_length = 384 # The maximum length of a feature (question and context)
doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
pad_on_right = tokenizer_biobert_large.padding_side == "right"

for b in ['6','7','8','9']:
    print('---------- predict new_BioASQ_',b,'B_SQuAD ----------',)
    new_BioASQ_xB_SQuAD = Dataset.from_dict(np.load('./BioASQ/Task'+b+'BGoldenEnriched/new_BioASQ_'+b+'B_SQuAD_BioBERT.npy',allow_pickle='TRUE').item())
    new_BioASQ_xB_features = new_BioASQ_xB_SQuAD.map(
        lambda x: prepare_validation_features(x, tokenizer = tokenizer_biobert_large),   
        batched=True,
        remove_columns=new_BioASQ_xB_SQuAD.column_names) 
   
    biobert_large_finetuned_model = BertForQuestionAnswering.from_pretrained('dmis-lab/biobert-large-cased-v1.1-squad')
    data_collator = default_data_collator
    
    trainer_BioBERTlarge = Trainer(
        biobert_large_finetuned_model,
        data_collator=data_collator,
        tokenizer=tokenizer_biobert_large,
    )

    raw_predictions_BioBERTlarge = trainer_BioBERTlarge.predict(new_BioASQ_xB_features) ## size (2, #xx, 384) = (start/end, #example features, length)
    np.save('./BioASQ/Task'+b+'BGoldenEnriched/raw_pred_BioBERTlarge_newBioASQ_'+b+'B.npy',raw_predictions_BioBERTlarge.predictions) 




