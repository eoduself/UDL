# Do not forget to run git clone https://github.com/amazon-science/esci-data.git to download data

import pandas as pd
from tqdm import tqdm
import os
import json

# For NDCG, we set the gains of E, S, C, and I to 1.0, 0.1, 0.01, and 0.0, respectively
def label_value(x):
    if x == "E":
       return 1.0
    elif x == "S":
       return 0.1
    elif x == "C":
       return 0.01
    elif x == "I":
       return 0
    else:
       print("ERROR")


df_examples = pd.read_parquet('esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet')
df_products = pd.read_parquet('esci-data/shopping_queries_dataset/shopping_queries_dataset_products.parquet')
df_sources = pd.read_csv("esci-data/shopping_queries_dataset/shopping_queries_dataset_sources.csv")

df_examples_products = pd.merge(
    df_examples,
    df_products,
    how='left',
    left_on=['product_locale','product_id'],
    right_on=['product_locale', 'product_id']
)

df_task = df_examples_products[df_examples_products["small_version"] == 1]
en_task = df_task[df_task["product_locale"] == "us"]
en_task_test = en_task[en_task["split"] == "test"].fillna("")
en_task_test["esci_label"] = en_task_test["esci_label"].apply(label_value)

corpus = {}

for doc_id in tqdm(en_task_test["product_id"].unique()):
    sample = en_task_test[en_task_test["product_id"] == doc_id]
    corpus[doc_id] = {'title':list(sample['product_title'])[0], 'text': list(sample['product_description'])[0] + \
                      list(sample['product_bullet_point'])[0] + list(sample['product_brand'])[0] + \
                      list(sample['product_color'])[0]}
print('corpus len:', len(corpus))

queries = {}
q_id = "ma-amazon"
num = 0
for query in tqdm(en_task_test["query"].unique()):
  queries[q_id+str(num)] = query
  num += 1
print('queries len:', len(queries))

qrels = {}

for key in tqdm(queries.keys()):
  sample = en_task_test[en_task_test["query"]==queries[key]]
  for index, row in sample.iterrows():
    if key not in qrels.keys():
      qrels[key] = {row['product_id']: row['esci_label']}
    else:
      qrels[key].update({row['product_id']: row['esci_label']})
print('qrels len:', len(qrels))

if not os.path.exists("esci-data/ndcg"):
    os.mkdir("esci-data/ndcg")

with open("esci-data/ndcg/corpus.jsonl", "w") as output:
    json.dump(corpus, output)
with open("esci-data/ndcg/qrels.jsonl", "w") as output:
    json.dump(qrels, output)
with open("esci-data/ndcg/queries.jsonl", "w") as output:
    json.dump(queries, output)

print("Reassign Gains for Recall measurement.")

# For Recall, we set the gains of E, S, C, and I to 1.0, 0.0, 0.0, and 0.0, respectively

for key in qrels.keys():
  for key2 in qrels[key].keys():
    if qrels[key][key2] != 1.0:
      qrels[key][key2] = 0.0

if not os.path.exists("esci-data/recall"):
    os.mkdir("esci-data/recall")

with open("esci-data/recall/corpus.jsonl", "w") as output:
    json.dump(corpus, output)
with open("esci-data/recall/qrels.jsonl", "w") as output:
    json.dump(qrels, output)
with open("esci-data/recall/queries.jsonl", "w") as output:
    json.dump(queries, output)    
