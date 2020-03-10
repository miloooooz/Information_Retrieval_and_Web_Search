import numpy as np
import xgboost as xgb
from math import log
import spacy
import re
import pickle


def disambiguate_mentions(train_mentions, train_labels, dev_mentions, men_docs, parsed_entity_pages):

    '''
    :param train_mentions: key: mention_id, value: {doc_title: values, mention: token span, offset: position,
    length: length, candidate_entities: each entity candidate corresponds to key in parsed_candidate_entities}
    :param train_labels: key: mention_id, value: {doc_title: values, mention: token span, label: mention's ground truth entity label}
    :param dev_mentions: tokens
    :param men_docs: key: document title, value: document text
    :param parsed_entity_pages: key: entity name, value: text corresponding to entity's description (id, token ,lemma,
     pos-tag, entity-tag)
    :return: {mid: "Entity_label"}
    '''
    doc_term, wiki_term, doc_tf, doc_idf = preprocessing(men_docs=men_docs, parsed_entity_pages=parsed_entity_pages)
    train_doc_tf_idf, train_candidate_tf_idf = doc_tfidf(mentions=train_mentions, wiki_term=wiki_term, doc_tf=doc_tf,
                                                         doc_idf=doc_idf)
    test_doc_tf_idf, test_candidate_tf_idf = doc_tfidf(mentions=dev_mentions, wiki_term=wiki_term, doc_tf=doc_tf,
                                                         doc_idf=doc_idf)
    train_missing = missing_words(wiki_words=wiki_term, doc_words=doc_term,mentions=train_mentions)
    train_len_diff = length_diff(mentions=train_mentions)
    train_mention_candidate_sim = mention_candidate_words_sim(mentions=train_mentions)
    test_missing = missing_words(wiki_words=wiki_term, doc_words=doc_term,mentions=dev_mentions)
    test_mention_candidate_sim = mention_candidate_words_sim(mentions=dev_mentions)
    test_len_diff = length_diff(mentions=dev_mentions)

    param = {'objective': 'rank:pairwise', 'max_depth': 7, 'eta': 0.05, 'lambda': 100,
             'min_child_weight': 0.01, 'subsample': 0.5}
    model = train(params=param, labels=train_labels, mentions=train_mentions, candidate_tfidf=train_candidate_tf_idf,
                  len_diff=train_len_diff, estimator=2500, missing=train_missing, similarity = train_mention_candidate_sim, doc_tfidf = train_doc_tf_idf)
    prediction = test(model=model, mentions=dev_mentions, candidate_tfidf = test_candidate_tf_idf, missing=test_missing,
                      len_diff=test_len_diff, similarity = test_mention_candidate_sim, doc_tfidf=test_doc_tf_idf)
    candidates = dict()
    final_entities = dict()
    iterate = 0
    for men_id in dev_mentions:
        candidates[men_id] = dev_mentions[men_id]['candidate_entities']
        pred = prediction[iterate:iterate + len(candidates[men_id])]
        for i in range(len(candidates[men_id])):
            if pred[i] == max(pred):
                final_entities[men_id] = dev_mentions[men_id]['candidate_entities'][i]
        iterate += len(candidates[men_id])
    return final_entities


def preprocessing(men_docs, parsed_entity_pages):
    # Preprocess the documents into tokens regardless of punctuations, stopwords and spaces
    # Count the document frequency for each token and candidates in parsed_entity_pages
    # Filter out some appropriate descrptions for each candidate entity with part-of-the-speech tag
    # calculate the TF and IDF for each document
    doc_tf = dict()
    doc_idf = dict()
    nlp = spacy.load('en_core_web_sm')
    doc_term = dict()
    wiki_term = dict()
    for title in men_docs:
        doc = nlp(men_docs[title])
        doc_token = [token.lemma_.lower() for token in doc
                     if (not token.is_space and not token.is_punct and not token.is_stop)]
        doc_term[title] = doc_token
        for token in doc_token:
            if token in doc_tf:
                if title in doc_tf[token]:
                    doc_tf[token][title] += 1
                else:
                    doc_tf[token][title] = 1
            else:
                doc_tf[token] = {title: 1}
    for word in doc_tf:
        doc_idf[word] = 1 + log(len(men_docs) / (1 + len(doc_tf[word])))
    for candidates in parsed_entity_pages:
        words = [word[2].lower() for word in parsed_entity_pages[candidates]
                 if (not word[3] == 'DET' and not word[3] == 'ADV' and not word[3] == 'ADJ')]
        wiki_term[candidates] = words
    return doc_term, wiki_term, doc_tf, doc_idf


def doc_tfidf(wiki_term, doc_tf, doc_idf, mentions):
    '''tf_idf of pased entity description in documents'''
    # for each mentioned word, find out its candidate entities and calculate the tf-idf score for each candidate words
    # In my case, I only used the previous 25 descriptions for each entity and calculate the tfidf for each candidate entity
    # and the tf-idf for the wikipedia page description for certain candidate entity
    tfidf = dict()
    candidate_tfidf = dict()
    for men_id in mentions:
        doc_id = mentions[men_id]['doc_title']
        candidates = mentions[men_id]['candidate_entities']
        for entity in candidates:
            candidate_tf_idf = 0
            candidates_word = re.sub("[^0-9a-zA-Z]+", " ", entity).split()
            cur_tf_idf = 0
            for word in candidates_word:
                if word in doc_tf and doc_id in doc_tf[word]:
                    candidate_tf_idf += (1 + log(1 + log(doc_tf[word][doc_id]))) * doc_idf[word]
            if men_id in candidate_tfidf:
                candidate_tfidf[men_id][entity] = candidate_tf_idf
            else:
                candidate_tfidf[men_id] = {entity: candidate_tf_idf}
            for word in wiki_term[entity][:25]:  #[:round(0.2 * len(wiki_term[entity]))]:
                if word in doc_tf and doc_id in doc_tf[word]:
                    cur_tf_idf += (1 + log(1 + log(doc_tf[word][doc_id]))) * doc_idf[word]
            if men_id in tfidf:
                tfidf[men_id][entity] = cur_tf_idf
            else:
                tfidf[men_id] = {entity: cur_tf_idf}
    return tfidf, candidate_tfidf


def missing_words(wiki_words, doc_words, mentions):
    # count the number of missing words that appear in document but not in wikipedia description for certain candidates
    # return as dictionary of percentage
    # since if the candidate word is closely related to the mentioned word in the document, then the words in the document
    # should appear in its wikipedia description as much as possible
    missing = dict()
    for men_id in mentions:
        candidates = mentions[men_id]['candidate_entities']
        doc_id = mentions[men_id]['doc_title']
        doc_terms = doc_words[doc_id]
        for entity in candidates:
            count = 0
            for word in doc_terms:
                if word not in wiki_words[entity]:
                    count += 1
            if men_id in missing:
                missing[men_id][entity] = count/len(doc_terms)
            else:
                missing[men_id] = {entity: count/len(doc_terms)}
    return missing

def length_diff(mentions):
    # calculate the length difference between the candidate words and the mentioned words
    # return in the form of percentage
    len_diff = dict()
    for men_id in mentions:
        candidates = mentions[men_id]['candidate_entities']
        length = mentions[men_id]['length']
        for entity in candidates:
            candidates_word = re.sub("[^0-9a-zA-Z]+", " ", entity)
            if men_id not in len_diff:
                len_diff[men_id] = {entity: (abs(length-len(candidates_word))/length)}
            else:
                len_diff[men_id][entity] = (abs(length - len(candidates_word)) / length)
    return len_diff


def mention_candidate_words_sim(mentions):
    # calculate the simularity between the mentioned word and its candidate entities
    diff = dict()
    for men_id in mentions:
        mention_word = re.sub("[^0-9a-zA-Z]+", " ", mentions[men_id]['mention'])
        candidates = mentions[men_id]['candidate_entities']
        for entity in candidates:
            cur = 0
            same = 0
            candidates_word = re.sub("[^0-9a-zA-Z]+", " ", entity)
            for place in candidates_word:
                if place in mention_word[cur:]:
                    same += 1
                    cur = mention_word.index(place) + 1
            if men_id in diff:
                diff[men_id][entity] = same/len(entity)
            else:
                diff[men_id] = {entity: same / len(entity)}
    return diff


def train(params, mentions, labels, candidate_tfidf, doc_tfidf, len_diff, missing, estimator, similarity):
    # train the model with doc_tfidf (the tf-idf for wikipedia description of candidates in documents), len_diff,
    # missing_words and similarity
    candidates = dict()
    candidate_nb = 0
    train_group = []
    for men_id in mentions:
        candidates[men_id] = mentions[men_id]['candidate_entities']
        candidate_nb += len(candidates[men_id])
        train_group.append(len(candidates[men_id]))
    train_x = np.zeros((candidate_nb, 4))  # row nb = candidate * mention
    train_y = np.zeros(candidate_nb)

    mentions_words = [k for k, v in mentions.items()]

    cur_index = 0
    for men_id in range(len(mentions_words)):
        for entity_index in range(len(candidates[mentions_words[men_id]])):
            if labels[mentions_words[men_id]]['label'] == candidates[mentions_words[men_id]][entity_index]:
                train_y[cur_index] = 1
            train_x[cur_index][0] = doc_tfidf[mentions_words[men_id]][candidates[mentions_words[men_id]][entity_index]]
            # train_x[cur_index][3] = candidate_tfidf[mentions_words[men_id]][candidates[mentions_words[men_id]][entity_index]]
            train_x[cur_index][1] = missing[mentions_words[men_id]][candidates[mentions_words[men_id]][entity_index]]
            train_x[cur_index][2] = len_diff[mentions_words[men_id]][candidates[mentions_words[men_id]][entity_index]]
            train_x[cur_index][3] = similarity[mentions_words[men_id]][candidates[mentions_words[men_id]][entity_index]]

            cur_index += 1

    dtrain = xgb.DMatrix(train_x, train_y)
    train_group = np.array(train_group)
    dtrain.set_group(train_group)
    model = xgb.train(params=params, dtrain=dtrain, num_boost_round=estimator)
    return model


def test(model, mentions, candidate_tfidf, doc_tfidf, len_diff, missing, similarity):
    candidates = dict()
    test_group = []
    candidate_nb = 0
    for men_id in mentions:
        candidates[men_id] = mentions[men_id]['candidate_entities']
        candidate_nb += len(candidates[men_id])
        test_group.append(len(candidates[men_id]))
    test_x = np.zeros((candidate_nb, 4))  # row nb = candidate * mention
    test_y = np.zeros(candidate_nb)

    mentions_words = [k for k, v in mentions.items()]
    test_group = []

    cur_index = 0
    for men_id in range(len(mentions_words)):
        for entity_index in range(len(candidates[mentions_words[men_id]])):
            test_x[cur_index][0] = doc_tfidf[mentions_words[men_id]][candidates[mentions_words[men_id]][entity_index]]
            # test_x[cur_index][3] = candidate_tfidf[mentions_words[men_id]][candidates[mentions_words[men_id]][entity_index]]
            test_x[cur_index][1] = missing[mentions_words[men_id]][candidates[mentions_words[men_id]][entity_index]]
            test_x[cur_index][2] = len_diff[mentions_words[men_id]][candidates[mentions_words[men_id]][entity_index]]
            test_x[cur_index][3] = similarity[mentions_words[men_id]][candidates[mentions_words[men_id]][entity_index]]

            cur_index += 1

    dtest = xgb.DMatrix(test_x, test_y)
    test_group = np.array(test_group)
    dtest.set_group(test_group)
    prediction = model.predict(dtest)
    return prediction






