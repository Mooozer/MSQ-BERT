# !/usr/bin/env python
# coding: utf-8

from tqdm.auto import tqdm
import numpy as np
def postprocess_qa_predictions(examples, features, raw_predictions, tokenizer, 
                               n_best_size = 20, max_answer_length = 30):
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["new_id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions = collections.OrderedDict()
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    for example_index, example in enumerate(tqdm(examples)):
        feature_indices = features_per_example[example_index]
        valid_answers = []
        context = example["context"]
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]

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
        predictions[example["new_id"]] = best_answer["text"]
        
    return predictions


import collections
def calculate_ex_f1(start_score, end_score, QuAD, tokenizer = tokenizer):
    '''
    input: start_score,  end_score,  
           QuAD (each id should correspond to a row of start_score/end_score), original QuAD dataset with new_id
           tokenizer. 
    '''
    validation_features = QuAD.map(lambda x: prepare_validation_features(x, tokenizer),
                                             batched=True, remove_columns=QuAD.column_names)
    validation_features.set_format(type=validation_features.format["type"],  
                                        columns=list(validation_features.features.keys()))
    predictions_predictions =  (start_score, end_score)

    final_predictions = postprocess_qa_predictions(QuAD, validation_features, predictions_predictions, tokenizer = tokenizer)
    metric = load_metric("squad")
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
    references = [{"id": ex["new_id"], "answers": ex["answers"]} for ex in QuAD]
    
    return metric.compute(predictions=formatted_predictions, references=references), formatted_predictions, references


from string import punctuation
def word_matching_scores(string_list, groundtruth_list):
    '''
    string_list: predicted answer(string) list
    groundtruth_list: groundtruth answer (string) list of list (multiple answers for each piece of data)
    output: scalar, mean of matching scores = mean(matching words /total words of groundtruth answer)
    '''
    if len(string_list) != len(groundtruth_list):
        return('error: two lists should have the same length')
    match_scores = []
    for i in range(len(string_list)):
        string1, string2_list = string_list[i], groundtruth_list[i]
        matching_words = [set(string1.strip(punctuation).split()).intersection(set(string2.strip(punctuation).split())) for string2 in string2_list]
        groundtruth_words = [set(string2.strip(punctuation).split()) for string2 in string2_list]
        match_score = [len(i)/len(j) for i in matching_words for j in groundtruth_words]
        match_scores.append(max(match_score))
    return(np.mean(match_scores)*100)




from difflib import SequenceMatcher
def string_matching_scores(string_list, groundtruth_list):
    '''
    string_list: predicted answer(string) list
    groundtruth_list: groundtruth answer (string) list of list (multiple answers for each piece of data)
    output: scalar, mean of string matching scores = mean(matching string length/groundtruth answer length)
    '''
    if len(string_list) != len(groundtruth_list):
        return('error: two lists should have the same length')
    match_scores = []
    for i in range(len(string_list)):
        string1, string2_list = string_list[i], groundtruth_list[i]
        match_size_ratio = [SequenceMatcher(None, string1, string2).find_longest_match(0,len(string1),0,len(string2)).size/len(string2) 
                            for string2 in string2_list]
        match_scores.append(max(match_size_ratio))
    return(np.mean(match_scores)*100)



import Levenshtein
def Levenshtein_similarity(string_list, groundtruth_list):
    '''
    string_list: predicted answer(string) list
    groundtruth_list: groundtruth answer (string) list of list (multiple answers for each piece of data)
    output: scalar, mean of Levenshtein similarity
    '''
    if len(string_list) != len(groundtruth_list):
        return('error: two lists should have the same length')
    Levenshtein_similarity = []
    for i in range(len(string_list)):
        string1, string2_list = string_list[i], groundtruth_list[i]
        Levenshtein_ratio = [Levenshtein.ratio(string1, string2) for string2 in string2_list]
        Levenshtein_similarity.append(max(Levenshtein_ratio))
    return(np.mean(Levenshtein_similarity)*100)


