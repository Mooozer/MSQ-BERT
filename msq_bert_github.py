#  1.Data from Wiki

import urllib.request
from bs4 import BeautifulSoup
import re
import pandas as pd

def get_treatment_from_wiki(sym):
    '''
    sym: input, symptoms string 
    output: treatment text
    '''
    url = 'https://en.wikipedia.org/wiki/' +  sym
    response = urllib.request.urlopen(url)
    html_doc = response.read().decode(encoding='UTF-8')
    parsed = BeautifulSoup(html_doc, "html.parser")
    
    soup = parsed.find("span",{'class':'mw-headline', "id":"Treatment"})#First look at Treatment
    if not soup:
        soup = parsed.find("span",{'class':'mw-headline', "id":"Management"})#If there is no Treatment, use Management
        if not soup:
            soup = parsed.find("span",{'class':'mw-headline', "id":"Treatments"})
            if not soup: 
                treatment = ["NA"]
                return treatment
        
    last_parent = list(soup.parents)[0]
    close_siblings = list(last_parent.next_siblings)
    
    treatment = []
    for i in range(len(close_siblings)):
        if close_siblings[i].name == 'h2':  #do not include next chapter
            break
        if close_siblings[i].name == 'p':   #main body of Treatment/Management
            ori_text = close_siblings[i].text
            ori_text = re.sub(r"\xa0", "", ori_text)  #remove "\xa0"
            ori_text = re.sub(r"\[\d*\]", "", ori_text) #remove cite
            ori_text = re.sub(r"\n", "", ori_text) #remove "\n"
            treatment.append(ori_text)
        else:
            continue
            
    return treatment

Sym_Tre_dic = {}
Symptoms = ['Fever', 'Cough', 'Shortness_of_breath',  'Myalgia', 'Headache', 'Anosmia', 
            'Sore_throat', 'Nasal_congestion', 'Rhinorrhea', 'Nausea', 'Vomiting', 'Diarrhea',
           'Abdominal_pain','Blood_in_stool','Chest_pain','Constipation','Dysphagia',
           'Palpitations','Knee_pain','Low_back_pain','Neck_pain','Paresthesia','Rash','Hemoptysis',
            'Pneumonia','Delayed_onset_muscle_soreness','Back_pain','Xerostomia','Dry_eye_syndrome',
           'Insomnia','Sleep_deprivation','Cyanosis','Somnolence','Heartburn','Tremor','Chronic_pain'] 
for s in Symptoms:
    Sym_Tre_dic[s] = get_treatment_from_wiki(s)
Sym_Tre_dic['Fever'] = Sym_Tre_dic['Fever'][0:-1] # remove the last one of "Fever" since it does not include treatment 
Sym_Tre_dic['Cough'] = Sym_Tre_dic['Cough'][0:-1] # remove the last one of "Cough" since it does not include treatment

"""# 2.SQUAD Data (Version 1)"""

#!pip install datasets
#!pip install transformers
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

squad_v2 = False
datasets = load_dataset("squad_v2" if squad_v2 else "squad") 
datasets['train'] = load_dataset("squad_v2" if squad_v2 else "squad", split='train[0:]')   
datasets['validation'] = load_dataset("squad_v2" if squad_v2 else "squad", split='validation[0:]')

max_length = 384 # The maximum length of a feature (question and context)
doc_stride = 128 # The overlap between two part of the context
pad_on_right = tokenizer.padding_side == "right"

"""## 2.1 prepare_train_features"""

def prepare_train_features(examples, tokenizer = tokenizer):
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
    offset_mapping = tokenized_examples.pop("offset_mapping")
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

tokenized_datasets = datasets.map(lambda x: prepare_train_features(x, tokenizer = tokenizer),
                                  batched=True, 
                                  remove_columns = datasets["train"].column_names)

"""# 3.Base BERT Fine-tuning

"""

from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

args = TrainingArguments(
    f"test-squad",
    evaluation_strategy = "epoch",
    learning_rate=3e-5,  
    per_device_train_batch_size= 16,  
    per_device_eval_batch_size= 16,   
    num_train_epochs=2, 
    weight_decay=0.01,
)

from transformers import default_data_collator
data_collator = default_data_collator

model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

"""### 3.1 Uncomment for Training """

##############################
#trainer.train()
#trainer.save_model("test-squad-trained") 
##############################

"""# 4.Several BERT models"""

from transformers import BertForQuestionAnswering,BertTokenizer,BertModel,AutoTokenizer 
import torch
import collections

#model 1: fine-tuned bert_base  
output_dir2 = "./test-squad-trained"
local_model_2 = BertForQuestionAnswering.from_pretrained(output_dir2)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

#model 2: fine-tuned bert_large 
bert_large_finetuned_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

#model 3: fine-tuned biobert_base  
biobert_base_finetuned_model = BertForQuestionAnswering.from_pretrained('dmis-lab/biobert-base-cased-v1.1-squad')
tokenizer_biobert_base = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1-squad')

#model 4: fine-tuned biobert_large
biobert_large_finetuned_model = BertForQuestionAnswering.from_pretrained('dmis-lab/biobert-large-cased-v1.1-squad')
tokenizer_biobert_large = AutoTokenizer.from_pretrained('dmis-lab/biobert-large-cased-v1.1-squad')

"""# 5.Evaluation

## 5.1 Evaluation on SQUAD
"""

def prepare_validation_features(examples, tokenizer = tokenizer):
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
        tokenized_examples["example_id"].append(examples["id"][sample_index])
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]
    return tokenized_examples

"""## 5.2 local_model_2"""

args2 = TrainingArguments(
    f"test-squad",
    evaluation_strategy = "epoch",
    learning_rate= 3e-5,  
    per_device_train_batch_size = 16,  
    per_device_eval_batch_size= 16,   
    num_train_epochs= 2, 
    weight_decay=0.01,
)
trainer2 = Trainer(
    local_model_2, 
    args2,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

raw_predictions2 = trainer2.predict(validation_features)

"""## 5.3 bert_large_finetuned_model

"""

args_large = TrainingArguments(
    f"test-squad",
    evaluation_strategy = "epoch",
    learning_rate= 3e-5,  
    per_device_train_batch_size = 16,  
    per_device_eval_batch_size= 16,   
    num_train_epochs= 2, 
    weight_decay=0.01,
)
trainer_large = Trainer(
    bert_large_finetuned_model,
    args_large,
    tokenizer=tokenizer,
)

raw_predictions_large = trainer_large.predict(validation_features)

"""## 5.4 Biobert_base_finetuned_model """

validation_features_for_biobert_base = datasets["validation"].map(
    lambda x: prepare_validation_features(x, tokenizer = tokenizer_biobert_base),
    batched=True,
    remove_columns=datasets["validation"].column_names
)

args_biobert_base = TrainingArguments(
    f"test-squad",
    evaluation_strategy = "epoch",
    learning_rate= 3e-5,  
    per_device_train_batch_size = 16,  
    per_device_eval_batch_size= 16,   
    num_train_epochs= 2, 
    weight_decay=0.01,
)
trainer_biobert_base = Trainer(
    biobert_base_finetuned_model,
    args_biobert_base,
    tokenizer=tokenizer_biobert_base,
)

raw_predictions_biobert_base = trainer_biobert_base.predict(validation_features_for_biobert_base)

"""## 5.5 Biobert_large_finetuned_model  raw prediction"""

validation_features_for_biobert_large = datasets["validation"].map(
    lambda x: prepare_validation_features(x, tokenizer = tokenizer_biobert_large),
    batched=True,
    remove_columns=datasets["validation"].column_names
)

args_biobert_large = TrainingArguments(
    f"test-squad",
    evaluation_strategy = "epoch",
    learning_rate= 3e-5,  
    per_device_train_batch_size = 16,  
    per_device_eval_batch_size= 16,   
    num_train_epochs= 2, 
    weight_decay=0.01,
)
trainer_biobert_large = Trainer(
    biobert_large_finetuned_model,
    args_biobert_large,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer_biobert_large,
)

raw_predictions_biobert_large = trainer_biobert_large.predict(validation_features_for_biobert_large) #(2,1020,384)=(start/end, #eg, len)

"""## 5.6 Evaluate"""

validation_features.set_format(type=validation_features.format["type"], 
                               columns=list(validation_features.features.keys()))

validation_features_for_biobert_base.set_format(type=validation_features_for_biobert_base.format["type"], 
                               columns=list(validation_features_for_biobert_base.features.keys()))

validation_features_for_biobert_large.set_format(type=validation_features_for_biobert_large.format["type"], 
                               columns=list(validation_features_for_biobert_large.features.keys()))

from tqdm.auto import tqdm
import numpy as np
def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size = 20, max_answer_length = 30):
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)
    predictions = collections.OrderedDict()
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    for example_index, example in enumerate(tqdm(examples)):
        feature_indices = features_per_example[example_index]

        min_null_score = None 
        valid_answers = []
        
        context = example["context"]
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": 0.0}
        if not squad_v2:
            predictions[example["id"]] = best_answer["text"]
        else:
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            predictions[example["id"]] = answer

    return predictions

final_predictions2 = postprocess_qa_predictions(datasets["validation"], 
                                               validation_features, 
                                               raw_predictions2.predictions)
final_predictions_large = postprocess_qa_predictions(datasets["validation"], 
                                               validation_features, 
                                               raw_predictions_large.predictions)
final_predictions_biobert_base  = postprocess_qa_predictions(datasets["validation"], 
                                               validation_features_for_biobert_base, 
                                               raw_predictions_biobert_base.predictions)
final_predictions_biobert_large  = postprocess_qa_predictions(datasets["validation"], 
                                               validation_features_for_biobert_large, 
                                               raw_predictions_biobert_large.predictions)

metric = load_metric("squad_v2" if squad_v2 else "squad")

formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions2.items()]
references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]
print('local_model_2' ,metric.compute(predictions=formatted_predictions, references=references))

formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions_large.items()]
references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]
print('model_large' , metric.compute(predictions=formatted_predictions, references=references))

formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions_biobert_base.items()]
references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]
print('model_biobert_base' , metric.compute(predictions=formatted_predictions, references=references))

formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions_biobert_large.items()]
references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]
print('model_biobert_large' , metric.compute(predictions=formatted_predictions, references=references))

"""# 6.Wiki_symptom_QA_data-v2 """

from datasets import Dataset
wiki_contents_file = pd.read_csv("./Wiki_symptom_QA_data-v2/wiki_contents.csv")
wiki_answers_file1 = pd.read_csv("./Wiki_symptom_QA_data-v2/wiki_answers1.csv")
wiki_answers_file2 = pd.read_csv("./Wiki_symptom_QA_data-v2/wiki_answers2.csv")
wiki_answers_file3 = pd.read_csv("./Wiki_symptom_QA_data-v2/wiki_answers3.csv")

def generate_Wiki_QUAD(question_pair_list):
  '''
  question_pair_list should be a list with two elements:
  the first one is the question string before <symptom>,
  the second one is the question string after <symptom>,
  e.g. question_pair_list = ["What is the treatment plan for ", "?"]
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
           
  Wiki_data_dic = {'id':[], 'title':[], "context":[], "answers":[], "question":[] }
  for k in range(len(Symptoms)):
      i = Symptoms[k]
      tem_content =  wiki_contents_file[i].tolist()
      tem_answers1 =  wiki_answers_file1[i].tolist()
      tem_answers2 =  wiki_answers_file2[i].tolist()
      tem_answers3 =  wiki_answers_file3[i].tolist()
      for a in range(len(tem_content)):
          if type(tem_content[a])==str:
              Wiki_data_dic['context'] += [tem_content[a]]
              ans_dic = {'answer_start': [tem_content[a].find(tem_answers1[a])] + [tem_content[a].find(tem_answers2[a])] + [tem_content[a].find(tem_answers3[a])],
                        'text':[tem_answers1[a]]+[tem_answers2[a]]+[tem_answers3[a]]}
              Wiki_data_dic['answers'] += [ans_dic]
              
              Wiki_data_dic['id'] += [i+str(a)]
              Wiki_data_dic['title'] += ['QA_pair_'+i+str(a)]
              Wiki_data_dic['question'] += [question_pair_list[0] + Symptoms_words[k] +question_pair_list[1]]
  Wiki_QUAD = Dataset.from_dict(Wiki_data_dic)
  return Wiki_QUAD 

#e.g.
Wiki_QUAD = generate_Wiki_QUAD(["How to reduce symptoms of ","?"])
Wiki_QUAD

import collections
def calculate_wiki_ex_f1(start_score, end_score, Ques_pair, tokenizer = tokenizer):
  Wiki_QUAD = generate_Wiki_QUAD(Ques_pair)

  validation_features_wiki = Wiki_QUAD.map(
    lambda x: prepare_validation_features(x, tokenizer),
    batched=True,
    remove_columns=Wiki_QUAD.column_names)  
  
  validation_features_wiki.set_format(type=validation_features_wiki.format["type"],  columns=list(validation_features_wiki.features.keys()))
  examples_wiki = Wiki_QUAD
  features_wiki = validation_features_wiki
  example_wiki_id_to_index = {k: i for i, k in enumerate(examples_wiki["id"])}
  features_per_example_wiki = collections.defaultdict(list)

  for i, feature in enumerate(features_wiki):
    features_per_example_wiki[example_wiki_id_to_index[feature["example_id"]]].append(i)
  
  predictions_wiki_predictions =  (start_score, end_score)
  final_predictions_wiki = postprocess_qa_predictions(Wiki_QUAD, 
                                           validation_features_wiki, 
                                           predictions_wiki_predictions)
  
  metric = load_metric("squad_v2" if squad_v2 else "squad")
  formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions_wiki.items()]
  references = [{"id": ex["id"], "answers": ex["answers"]} for ex in Wiki_QUAD]
  return metric.compute(predictions=formatted_predictions, references=references)

"""## 6.1 Questions Augmentation

"""

lang_tgt_list= ['af','sq','ar','hy','az','eu','be','bg','ca','zh','hr','cs','da','nl','et','tl','fi','fr','gl','ka','de','el','ht','iw','hi','hu','is','id','ga','it',
 'ja','ko','lv','lt','mk','ms','mt','no','fa','pl','pt','ro','ru','sr','sk','sl','es','sw','sv','th','tr','uk','ur','vi','cy','yi']
language_list = ['Afrikaans','Albanian','Arabic','Armenian','Azerbaijani','Basque','Belarusian','Bulgarian','Catalan','Chinese','Croatian','Czech',
 'Danish','Dutch','Estonian','Filipino','Finnish','French','Galician','Georgian','German','Greek','Haitian Creole','Hebrew','Hindi',
 'Hungarian','Icelandic','Indonesian','Irish','Italian','Japanese','Korean','Latvian','Lithuanian','Macedonian','Malay','Maltese','Norwegian',
 'Persian','Polish','Portuguese','Romanian','Russian','Serbian','Slovak','Slovenian','Spanish','Swahili','Swedish','Thai','Turkish','Ukrainian',
 'Urdu','Vietnamese','Welsh','Yiddish']

!pip install google_trans_new
from google_trans_new import google_translator  
import time

rep_dic = {(i, lang_tgt_list[i]) : 0 for i in range(len(lang_tgt_list)) } 
translator = google_translator()  
translate_input = "What possible therapeutical method is helpful to treat " + "AA" +"?"
Ques_argument = [translate_input] 
for i in range(len(lang_tgt_list)):
  time.sleep(1)  #to make API request slower
  translate_middle = translator.translate(translate_input, lang_src='en', lang_tgt=lang_tgt_list[i])  
  translate_output = translator.translate(translate_middle, lang_src=lang_tgt_list[i], lang_tgt='en')  
  if translate_output in Ques_argument:
    rep_dic[(i,lang_tgt_list[i])] += 1
  Ques_argument.append(translate_output)

import pandas as pd
Ques_argument = pd.read_csv("./Ques_argumentAA.csv")
Ques_argument = Ques_argument['Ques_argument'].tolist()
Ques_argument = [Ques_argument[i].lower().strip() for i in range(len(Ques_argument))] 
Ques_argument = [Ques_argument[i] for i in range(len(Ques_argument)) if (Ques_argument[i][0]=='w' and "aa" in Ques_argument[i])]
Ques_argument = list(dict.fromkeys(Ques_argument)) 
Qu_pairs = [Ques_argument[i].split("aa") for i in range(len(Ques_argument))] 

Ques_max_len = max([len(tokenizer(Qu_pairs)['input_ids'][i]) for i in range(len(Qu_pairs))])
Qu_pairs_padded = [["[PAD] " * (Ques_max_len - len(tokenizer(Qu_pairs)['input_ids'][i])) + Qu_pairs[i][0],  Qu_pairs[i][1]] for i in range(len(Qu_pairs)) ]

"""## 6.2Make Tables """

import pandas as pd
import collections

col_names =  ['Question formula', 'local_model_2', 'bert_large_finetuned','biobert_base','biobert_large'] 
col_names_answer = ['Question formula'] + ["Q"+str(i) for i in range(343)] 
f1_result  = pd.DataFrame(columns = col_names)
exact_result  = pd.DataFrame(columns = col_names) 

local_model_2_answer  = pd.DataFrame(columns = col_names_answer) 
model_large_answer  = pd.DataFrame(columns = col_names_answer) 
biobert_base_answer  = pd.DataFrame(columns = col_names_answer) 
biobert_large_answer  = pd.DataFrame(columns = col_names_answer) 


for q in range(len(Qu_pairs_padded)):
  Wiki_QUAD = generate_Wiki_QUAD(Qu_pairs_padded[q])
  
  validation_features_wiki = Wiki_QUAD.map(
    lambda x: prepare_validation_features(x, tokenizer), batched=True, remove_columns=Wiki_QUAD.column_names
    )
  raw_predictions2_wiki = trainer2.predict(validation_features_wiki) 
  raw_predictions_large_wiki = trainer_large.predict(validation_features_wiki) 


  validation_features_wiki_for_biobert_base = Wiki_QUAD.map(
    lambda x: prepare_validation_features(x, tokenizer_biobert_base), batched=True, remove_columns=Wiki_QUAD.column_names
    )
  raw_predictions_bio_base_wiki = trainer_biobert_base.predict(validation_features_wiki_for_biobert_base)
  

  validation_features_wiki_for_biobert_large = Wiki_QUAD.map(
    lambda x: prepare_validation_features(x, tokenizer_biobert_large), batched=True, remove_columns=Wiki_QUAD.column_names
    )
  raw_predictions_bio_large_wiki = trainer_biobert_large.predict(validation_features_wiki_for_biobert_large)


  validation_features_wiki.set_format(type=validation_features_wiki.format["type"],  
                                      columns=list(validation_features_wiki.features.keys()))
  validation_features_wiki_for_biobert_base.set_format(type=validation_features_wiki_for_biobert_base.format["type"],  
                                                       columns=list(validation_features_wiki_for_biobert_base.features.keys()))
  validation_features_wiki_for_biobert_large.set_format(type=validation_features_wiki_for_biobert_large.format["type"],  
                                                        columns=list(validation_features_wiki_for_biobert_large.features.keys()))

  examples_wiki = Wiki_QUAD
  features_wiki = validation_features_wiki
  features_wiki_for_biobert_base = validation_features_wiki_for_biobert_base
  features_wiki_for_biobert_large = validation_features_wiki_for_biobert_large

  example_wiki_id_to_index = {k: i for i, k in enumerate(examples_wiki["id"])}
  example_wiki_id_to_index_bio_base = {k: i for i, k in enumerate(examples_wiki["id"])}
  example_wiki_id_to_index_bio_large  = {k: i for i, k in enumerate(examples_wiki["id"])}

  features_per_example_wiki = collections.defaultdict(list)
  features_per_example_wiki_bio_base = collections.defaultdict(list)
  features_per_example_wiki_bio_large = collections.defaultdict(list)

  for i, feature in enumerate(features_wiki):
    features_per_example_wiki[example_wiki_id_to_index[feature["example_id"]]].append(i)
  for i, feature in enumerate(features_wiki_for_biobert_base):
    features_per_example_wiki_bio_base[example_wiki_id_to_index_bio_base[feature["example_id"]]].append(i)
  for i, feature in enumerate(features_wiki_for_biobert_large):
    features_per_example_wiki_bio_large[example_wiki_id_to_index_bio_large[feature["example_id"]]].append(i)


  final_predictions2_wiki = postprocess_qa_predictions(Wiki_QUAD, validation_features_wiki, raw_predictions2_wiki.predictions)
  final_predictions_large_wiki = postprocess_qa_predictions(Wiki_QUAD, validation_features_wiki, raw_predictions_large_wiki.predictions)
  final_predictions_bio_base_wiki = postprocess_qa_predictions(Wiki_QUAD, validation_features_wiki_for_biobert_base, raw_predictions_bio_base_wiki.predictions)
  final_predictions_bio_large_wiki = postprocess_qa_predictions(Wiki_QUAD, validation_features_wiki_for_biobert_large, raw_predictions_bio_large_wiki.predictions)

  metric = load_metric("squad")

  formatted_predictions2 = [{"id": k, "prediction_text": v} for k, v in final_predictions2_wiki.items()]
  formatted_predictions_large = [{"id": k, "prediction_text": v} for k, v in final_predictions_large_wiki.items()]
  formatted_predictions_bio_base = [{"id": k, "prediction_text": v} for k, v in final_predictions_bio_base_wiki.items()]
  formatted_predictions_bio_large = [{"id": k, "prediction_text": v} for k, v in final_predictions_bio_large_wiki.items()]

  references = [{"id": ex["id"], "answers": ex["answers"]} for ex in Wiki_QUAD]

  evalu_2 = metric.compute(predictions=formatted_predictions2, references=references)
  evalu_large = metric.compute(predictions=formatted_predictions_large, references=references)
  evalu_bio_base = metric.compute(predictions=formatted_predictions_bio_base, references=references)
  evalu_bio_large = metric.compute(predictions=formatted_predictions_bio_large, references=references)

  f1_model_2 , exact_model_2 =  evalu_2['f1'], evalu_2['exact_match']
  f1_model_large, exact_model_large = evalu_large['f1'], evalu_large['exact_match']
  f1_model_bio_base, exact_model_bio_base = evalu_bio_base['f1'], evalu_bio_base['exact_match']
  f1_model_bio_large, exact_model_bio_large = evalu_bio_large['f1'], evalu_bio_large['exact_match']

  f1_result.loc[q] = [Qu_pairs[q][0]+"<symptom>"+Qu_pairs[q][1], f1_model_2, f1_model_large,f1_model_bio_base,f1_model_bio_large] 
  exact_result.loc[q] = [Qu_pairs[q][0]+"<symptom>"+Qu_pairs[q][1], exact_model_2, exact_model_large,exact_model_bio_base, exact_model_bio_large] 
  local_model_2_answer.loc[q] = [Qu_pairs[q][0]+"<symptom>"+Qu_pairs[q][1]] + [formatted_predictions2[j]['prediction_text'] for j in range(len(formatted_predictions2))]
  model_large_answer.loc[q] = [Qu_pairs[q][0]+"<symptom>"+Qu_pairs[q][1]] + [formatted_predictions_large[j]['prediction_text'] for j in range(len(formatted_predictions_large))]
  biobert_base_answer.loc[q] = [Qu_pairs[q][0]+"<symptom>"+Qu_pairs[q][1]] + [formatted_predictions_bio_base[j]['prediction_text'] for j in range(len(formatted_predictions_bio_base))]
  biobert_large_answer.loc[q] = [Qu_pairs[q][0]+"<symptom>"+Qu_pairs[q][1]] + [formatted_predictions_bio_large[j]['prediction_text'] for j in range(len(formatted_predictions_bio_large))]

"""## 6.3 Calculate all scores (for single model)
### Notice: change model and corresponding tokenizer
"""

all_score_start = np.zeros((343, 384)).reshape(343,384,1)  
all_score_end = np.zeros((343, 384)).reshape(343,384,1)    

for q in range(len(Qu_pairs)):  
  Wiki_QUAD = generate_Wiki_QUAD(Qu_pairs[q])  
  
  validation_features_wiki = Wiki_QUAD.map(
    lambda x: prepare_validation_features(x, tokenizer),  #tokenizer_biobert_base, tokenizer_biobert_large
    batched=True,
    remove_columns=Wiki_QUAD.column_names)
  
  raw_predictions2_wiki = trainer2.predict(validation_features_wiki) 
  #raw_predictions_large_wiki = trainer_large.predict(validation_features_wiki) 
  #raw_predictions_biobert_base_wiki = trainer_biobert_base.predict(validation_features_wiki) 
  #raw_predictions_biobert_large_wiki = trainer_biobert_large.predict(validation_features_wiki) 

  all_score_start = np.concatenate((all_score_start, raw_predictions2_wiki.predictions[0].reshape(343,384,1)), axis=2)
  all_score_end = np.concatenate((all_score_end, raw_predictions2_wiki.predictions[1].reshape(343,384,1)), axis=2)
all_score_start = all_score_start[:,:,1:] 
all_score_end = all_score_end[:,:,1:]

"""## 6.4 MSQ Approach

### 6.4.1 normalized scores
"""

from sklearn.preprocessing import normalize
all_score_start_norm =  np.empty_like(all_score_start)
all_score_end_norm =  np.empty_like(all_score_end)
for i in range(all_score_start.shape[0]):
  for j in range(all_score_start.shape[2]):
    all_score_start_norm[i,:,j] = (all_score_start[i,:,j]- all_score_start[i,:,j].min())/(all_score_start[i,:,j].max()-all_score_start[i,:,j].min())
    all_score_start_norm[i,:,j] = all_score_start_norm[i,:,j]/all_score_start_norm[i,:,j].sum()
    all_score_end_norm[i,:,j] = (all_score_end[i,:,j]- all_score_end[i,:,j].min())/(all_score_end[i,:,j].max()-all_score_end[i,:,j].min())
    all_score_end_norm[i,:,j] = all_score_end_norm[i,:,j]/all_score_end_norm[i,:,j].sum()

"""### 6.4.2 Word frequency scores and its normlized score



"""

from scipy.special import softmax
def Freq_score_QC(Q_list, C_list):
  Ques_Context_Fre_score = np.ones((len(C_list), len(Q_list)))*0.001  # initialize 
  for c in range(len(C_list)): 
    for q in range(len(Q_list)):
      question_word = Q_list[q].lower().split(' ')
      Ques_Context_Fre_score[c,q] += np.mean([C_list[c].lower().split(' ').count(qw) for qw in question_word])
  return Ques_Context_Fre_score

Ques_Context_Fre_score = Freq_score_QC([Qu_pairs[q][0]+Qu_pairs[q][1] for q in range(len(Qu_pairs))],  Wiki_QUAD['context'])

#all_score_start_norm_fs
all_score_start_norm_fs = (Ques_Context_Fre_score.reshape(-1,1) * all_score_start_norm.transpose(0,2,1).reshape(-1,LL)).reshape(all_score_start_norm.shape[0], all_score_start_norm.shape[2],all_score_start_norm.shape[1]).transpose(0,2,1)
all_score_end_norm_fs = (Ques_Context_Fre_score.reshape(-1,1) * all_score_end_norm.transpose(0,2,1).reshape(-1,LL)).reshape(all_score_end_norm.shape[0], all_score_end_norm.shape[2],all_score_end_norm.shape[1]).transpose(0,2,1)

"""### 6.4.3 SVD"""

import matplotlib.pyplot as plt
import numpy as np

U_s,S_s,V_s = np.linalg.svd(all_score_start_norm_fs) 
U_e,S_e,V_e = np.linalg.svd(all_score_end_norm_fs) 

svd_score_start = np.zeros((9,) + all_score_start.shape)
svd_score_end = np.zeros((9,) + all_score_start.shape) 
svd_res_score_start = np.zeros((9,) + all_score_start.shape) 
svd_res_score_end = np.zeros((9,) + all_score_start.shape)
for r in range(1,10):
  for i in range(U_s.shape[0]):
    for j in range(r):
      svd_score_start[r-1,i,:,:] += S_s[i,j] * np.matmul(U_s[i,:,j:j+1], V_s[i,j:j+1,:]) 
      svd_score_end[r-1,i,:,:]   += S_e[i,j] * np.matmul(U_e[i,:,j:j+1], V_e[i,j:j+1,:])  
  svd_res_score_start[r-1,:,:,:] =  all_score_start_norm - svd_score_start[r-1,i,:,:] 
  svd_res_score_end[r-1,:,:,:]   =  all_score_end_norm - svd_score_end[r-1,i,:,:] 

#norm is better: 
svd_score_start_norm =  (svd_score_start - svd_score_start.min(axis=2,keepdims=True))/(svd_score_start - svd_score_start.min(axis=2,keepdims=True)).sum(axis=2,keepdims=True)
svd_score_end_norm =  (svd_score_end - svd_score_end.min(axis=2,keepdims=True))/(svd_score_end - svd_score_end.min(axis=2,keepdims=True)).sum(axis=2,keepdims=True)

svd_results_f1 = [] 
svd_results_exact = []
for r in range(1,10):
  svd_results = calculate_wiki_ex_f1(np.max(svd_score_start_norm[r-1,:,:,:], axis=2), np.max(svd_score_end_norm[r-1,:,:,:], axis=2), 
                                     Qu_pairs_padded[0],     #Qu_pairs_padded[i] did NOT influence the result
                                     tokenizer = tokenizer)  #tokenizer_biobert_base
  svd_results_f1.append(svd_results['f1'])
  svd_results_exact.append(svd_results['exact_match'])
print(max(svd_results_f1),  'r =',  np.argmax(svd_results_f1)+1)
print(max(svd_results_exact), 'r =', np.argmax(svd_results_exact)+1)
plt.plot(range(1,10), svd_results_f1, 'ro-')
plt.plot(range(1,10), svd_results_exact, 'bo-')

"""## 6.5 Final prediction"""

r=1
final_start_score = np.max(svd_score_start_norm[r-1,:,:,:], axis=2)
final_end_score = np.max(svd_score_end_norm[r-1,:,:,:], axis=2)
validation_features_wiki_large = Wiki_QUAD.map(
    lambda x: prepare_validation_features(x, tokenizer = tokenizer),
    batched=True,
    remove_columns=Wiki_QUAD.column_names
)
final_predictions_wiki = postprocess_qa_predictions(Wiki_QUAD, validation_features_wiki_large, (final_start_score, final_end_score))
formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions_wiki.items()]
formatted_predictions_text = [formatted_predictions[j]['prediction_text'] for j in range(len(formatted_predictions))]
final_prediction_text = pd.DataFrame(columns = ['final prediction text'])
for q in range(len(formatted_predictions_text)):
  final_prediction_text.loc[q] = formatted_predictions_text[q]
final_prediction_text