import spacy
import itertools
from math import log

# Input format: Document(D) = {key: doc_id, value: document text}
#               Query(Q) = String of words
#               Dictionary of Entity(DoE) = {key: entity, value: entity_id}

class InvertedIndex:
    def __init__(self):
        self.tf_tokens = dict()
        self.tf_entities = dict()
        self.idf_tokens = dict()
        self.idf_entities = dict()

    def index_documents(self, documents):
        nlp = spacy.load('en_core_web_sm')
        for i in documents:
            doc = nlp(documents[i])
            ent = [ent.text for ent in doc.ents]        # extract entity from documents
            tokens = [token.text for token in doc if (not token.is_space and not token.is_stop and not token.is_punct)]
            # extract token and filter the stopwords, punctuations and stopwords
            should_remove = []
            # to remove single word entities from token lists, single word entities only be considered as entity, not tokens
            for en in ent:
                if len(en.split(" ")) == 1:
                    should_remove.append(en)
            for en in should_remove:
                if en in tokens:
                    tokens.remove(en)
            for t in tokens:            # count token number in each document -> tf
                if t in self.tf_tokens:
                    self.tf_tokens[t][i] = tokens.count(t)
                else:
                    self.tf_tokens[t] = {i: tokens.count(t)}
            for e in ent:
                if e in self.tf_entities:
                    self.tf_entities[e][i] = ent.count(e)
                else:
                    self.tf_entities[e] = {i: ent.count(e)}
        for t in self.tf_tokens:        # count total token number and inverse it -> idf
            self.idf_tokens[t] = 1 + log(len(documents) / (1 + len(self.tf_tokens[t])))
        for e in self.tf_entities:
            self.idf_entities[e] = 1 + log(len(documents) / (1 + len(self.tf_entities[e])))

    def split_query(self, Q, DoE):
        # Split query sentence into combinations of free keywords to see if there is any possible entities in Q
        # In this case we only consider the combinations in the increasing order of the query, which means that
        # the token order won't reverse
        result = []
        tokens = Q.strip().split(" ")
        token_combination = []
        for i in range(len(tokens) + 1):
            # enumerate the combination of tokens with different lengths
            token_combination.extend(itertools.combinations(tokens, i))

        probable_entities = set()
        for probable_entity in token_combination:
            if " ".join(probable_entity) in DoE:
                probable_entities.add(" ".join(probable_entity))

        probable_entities = list(probable_entities)

        entity_combination = []
        for i in range(len(probable_entities) + 1):
            # enumerate different combinations of entities
            entity_combination.extend(itertools.combinations(probable_entities, i))

        # to see whether the combination has more tokens than the query itself
        # add into the result list if the total token count in entity list is smaller than the token count in query
        for entity_subset in entity_combination:
            add = True
            entity_terms = " ".join(entity_subset).split()
            token_list = [i for i in tokens]
            for term in entity_terms:
                if entity_terms.count(term) > token_list.count(term):
                    add = False
                    break
            if add:
                for term in entity_terms:
                    token_list.remove(term)
                result.append({"tokens": token_list, "entities": list(entity_subset)})

        return result

    def max_score_query(self, query_splits, doc_id):
        # use the TF-IDF score calculated from Q1 to compute the query score and thus to give a combination of
        # tokens and entity lists with the maximum score in order to process with the query to the best performance
        score_entity = []
        score_token = []
        combined_score = []
        for i in range(len(query_splits)):
            cur_entity = 0
            cur_token = 0
            for en in query_splits[i]["entities"]:
                if en in self.tf_entities and doc_id in self.tf_entities[en]:
                    tf_idf_entity = (1 + log(self.tf_entities[en][doc_id])) * self.idf_entities[en]
                else:
                    tf_idf_entity = 0
                cur_entity += tf_idf_entity
            score_entity.append(cur_entity)

            for to in query_splits[i]["tokens"]:
                if to in self.tf_tokens and doc_id in self.tf_tokens[to]:
                    tf_idf_token = (1 + log(1 + log(self.tf_tokens[to][doc_id]))) * self.idf_tokens[to]
                else:
                    tf_idf_token = 0
                cur_token += tf_idf_token
            score_token.append(cur_token)

            combined_score.append(score_entity[i] + 0.4 * score_token[i])

        for i in range(len(combined_score)):
            if combined_score[i] == max(combined_score):
                result = (max(combined_score), query_splits[i])
                return result

