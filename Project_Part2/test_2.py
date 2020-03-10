## Import Necessary Modules...
import pickle
import project_part2 as project_part2

# #     ## We will be using the following function to compute the accuracy...
def compute_accuracy(result, data_labels):
    assert set(list(result.keys())) - set(list(data_labels.keys())) == set()
    TP = 0.0
    for id_ in result.keys():
        if result[id_] == data_labels[id_]['label']:
            TP +=1
    assert len(result) == len(data_labels)
    return TP/len(result)

def compute_mark(accuracy):
    if accuracy < 0.7:
        return 0
    elif accuracy <= 0.75:
        return 10*(20*accuracy-14)
    elif accuracy < 0.85:
        return 10*(5*accuracy-2.75)
    elif accuracy < 0.89:
        return 10*(12.5*accuracy-9.125)
    else:
        return 20

## Read the data sets...

### Read the Training Data
train_file = './Data2/train.pickle'
train_mentions = pickle.load(open(train_file, 'rb'))

### Read the Training Labels...
train_label_file = './Data2/train_labels.pickle'
train_labels = pickle.load(open(train_label_file, 'rb'))

### Read the Dev Data... (For Final Evaluation, we will replace it with the Test Data)
dev_file = './Data2/dev2.pickle'
dev_mentions = pickle.load(open(dev_file, 'rb'))

### Read the Parsed Entity Candidate Pages...
fname = './Data2/parsed_candidate_entities.pickle'
parsed_entity_pages = pickle.load(open(fname, 'rb'))

### Read the Mention docs...
mens_docs_file = "./Data2/men_docs.pickle"
men_docs = pickle.load(open(mens_docs_file, 'rb'))

### Read the Dev Labels... (For Final Evaluation, we will replace it with the Test Data)
dev_label_file = './Data2/dev2_labels.pickle'
dev_labels = pickle.load(open(dev_label_file, 'rb'))


# test_mention = train_mentions
# test_label = train_labels
test_mention = dev_mentions
test_label = dev_labels

## Result of the model...
result = project_part2.disambiguate_mentions(train_mentions, train_labels, test_mention, men_docs, parsed_entity_pages)

# # ## Here, we print out sample result of the model for illustration...
# for key in list(result)[:5]:
#     print('KEY: {} \t VAL: {}'.format(key,result[key]))


accuracy = compute_accuracy(result, test_label)
print(f"Accuracy = {accuracy*100:.3f}%")



#
#
#
# ### Read the Training Data
# train_file2 = './Data2/train.pickle'
# train_mentions2 = pickle.load(open(train_file2, 'rb'))
#
# ### Read the Training Labels...
# train_label_file2 = './Data2/train_labels.pickle'
# train_labels2 = pickle.load(open(train_label_file2, 'rb'))
#
# ### Read the Dev Data... (For Final Evaluation, we will replace it with the Test Data)
# dev_file2 = './Data2/dev2.pickle'
# dev_mentions2 = pickle.load(open(dev_file2, 'rb'))
#
# ### Read the Parsed Entity Candidate Pages...
# fname2 = './Data2/parsed_candidate_entities.pickle'
# parsed_entity_pages2 = pickle.load(open(fname2, 'rb'))
#
# ### Read the Mention docs...
# mens_docs_file2 = "./Data2/men_docs.pickle"
# men_docs2 = pickle.load(open(mens_docs_file2, 'rb'))
#
# ### Read the Dev Labels... (For Final Evaluation, we will replace it with the Test Data)
# dev_label_file2 = './Data2/dev2_labels.pickle'
# dev_labels2 = pickle.load(open(dev_label_file2, 'rb'))
#
# # test_mention2 = train_mentions2
# # test_label2 = train_labels2
# test_mention2 = dev_mentions2
# test_label2 = dev_labels2
#
# ## Result of the model...
# result2 = project_part2.disambiguate_mentions(train_mentions2, train_labels2, test_mention2, men_docs2, parsed_entity_pages2)
#
# # # ## Here, we print out sample result of the model for illustration...
# # for key in list(result2)[:5]:
# #     print('KEY: {} \t VAL: {}'.format(key,result2[key]))
#
# accuracy2 = compute_accuracy(result2, test_label2)
# print(f"Accuracy2 = {accuracy2*100:.3f}%")
# print(f"Accuracy difference = {abs(accuracy-accuracy2)*100:.3f}%")
# # print(f"Mark1 = {compute_mark(accuracy)}")
# # print(f"Mark2 = {compute_mark(accuracy2)}")
