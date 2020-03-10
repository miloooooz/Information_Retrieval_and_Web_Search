import pickle
import project_part1 as project_part1

with open('test_500docs.pickle', 'rb') as handle:
    documents = pickle.load(handle)
# documents = pickle.load(open(fname,"rb"))
#print(documents, end = '\n\n')

## Step- 1. Construct the index...
index = project_part1.InvertedIndex()
index.index_documents(documents)
## Test cases
Q = 'New York Times Trump travel'
DoE = {'New York Times':0, 'New York':1,'New York City':2}#, 'Trump':3}
doc_id = 3

## 2. Split the query...
query_splits = index.split_query(Q, DoE)

## 3. Compute the max-score...
result = index.max_score_query(query_splits, doc_id)
print(result)