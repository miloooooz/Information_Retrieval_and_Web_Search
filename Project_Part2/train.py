## Import Necessary Modules...
import pickle
from pprint import pprint
import project_part2_redo as project_part2

## Read the data sets...

### Read the Training Data
train_file = './Data/train.pickle'
train_mentions = pickle.load(open(train_file, 'rb'))

### Read the Training Labels...
train_label_file = './Data/train_labels.pickle'
train_labels = pickle.load(open(train_label_file, 'rb'))

### Read the Dev Data... (For Final Evaluation, we will replace it with the Test Data)
dev_file = './Data/dev.pickle'
dev_mentions = pickle.load(open(dev_file, 'rb'))

### Read the Parsed Entity Candidate Pages...
fname = './Data/parsed_candidate_entities.pickle'
parsed_entity_pages = pickle.load(open(fname, 'rb'))

### Read the Mention docs...
mens_docs_file = "./Data/men_docs.pickle"
men_docs = pickle.load(open(mens_docs_file, 'rb'))

## Result of the model...
result = project_part2.disambiguate_mentions(train_mentions, train_labels, train_mentions, men_docs, parsed_entity_pages)

# result

## We will be using the following function to compute the accuracy...
def compute_accuracy(result, data_labels):
    assert set(list(result.keys())) - set(list(data_labels.keys())) == set()
    TP = 0.0
    for id_ in result.keys():
        if result[id_] == data_labels[id_]['label']:
            TP +=1
    assert len(result) == len(data_labels)
    return TP/len(result)

accuracy = compute_accuracy(result, train_labels)
print("Accuracy = ", accuracy)
