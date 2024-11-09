from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from time import time
import json

def evaluation(dataset, data_path, model_save_path, batch_size=32, score_function="cos_sim"):
    record = {}
    if dataset != "ma-amazon":
        # Load document / queries / qrels of test set
        test_document, test_queries, test_qrels = GenericDataLoader(data_path).load(split="test")

        model = DRES(models.SentenceBERT(model_save_path), batch_size=batch_size)
        eval_retriever = EvaluateRetrieval(model, score_function=score_function)

        print("Size of test queries:",len(test_queries))
        print("Size of test qrles:",len(test_qrels))
        print("Size of test document:",len(test_document))

        #### Retrieve dense results (format of results is identical to qrels)
        start_time = time()
        results = eval_retriever.retrieve(test_document, test_queries)
        end_time = time()
        eval_retriever.k_values = [10, 100]

        print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))
        print("Retriever evaluation for k in: {}".format(eval_retriever.k_values))
        ndcg, _map, recall, precision = eval_retriever.evaluate(test_qrels, results, eval_retriever.k_values)

        mrr = eval_retriever.evaluate_custom(test_qrels, results, eval_retriever.k_values, metric="mrr")
        recall_cap = eval_retriever.evaluate_custom(test_qrels, results, eval_retriever.k_values, metric="r_cap")
        hole = eval_retriever.evaluate_custom(test_qrels, results, eval_retriever.k_values, metric="hole")


        list_per = ['NDCG@10','Recall@100']

        for key in ndcg.keys():
            if key in list_per:
                print(key, ndcg[key]*100)
                record[key] = ndcg[key]*100
        for key in recall.keys():
            if key in list_per:
                print(key, recall[key]*100)
                record[key] = recall[key]*100

    else:
        for test_data_loc in ["esci-data/recall/", "esci-data/ndcg/"]:
            # Load document / queries / qrels of test set
            with open(test_data_loc + "corpus.jsonl", "r") as file:
                for line in file:
                    test_document = json.loads(line)

            with open(test_data_loc + "queries.jsonl", "r") as file:
                for line in file:
                    test_queries = json.loads(line)

            with open(test_data_loc + "qrels.jsonl", "r") as file:
                for line in file:
                    test_qrels = json.loads(line)

            if "ndcg" in test_data_loc:
                ## gain score should be int
                for key in test_qrels.keys():
                    for key2 in test_qrels[key]:
                        test_qrels[key][key2] = int(100*test_qrels[key][key2])

            else:
                ## gain score should be int
                for key in test_qrels.keys():
                    for key2 in test_qrels[key]:
                        test_qrels[key][key2] = int(test_qrels[key][key2])

            model = DRES(models.SentenceBERT(model_save_path), batch_size=batch_size)
            eval_retriever = EvaluateRetrieval(model, score_function=score_function)

            print("Size of test queries:",len(test_queries))
            print("Size of test qrles:",len(test_qrels))
            print("Size of test document:",len(test_document))

            #### Retrieve dense results (format of results is identical to qrels)
            start_time = time()
            results = eval_retriever.retrieve(test_document, test_queries)
            end_time = time()
            eval_retriever.k_values = [50, 100, 500]

            print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))
            print("Retriever evaluation for k in: {}".format(eval_retriever.k_values))
            ndcg, _map, recall, precision = eval_retriever.evaluate(test_qrels, results, eval_retriever.k_values)

            if "recall" in test_data_loc:
                list_per = ['Recall@100', 'Recall@500']
                for key in recall.keys():
                    if key in list_per:
                        print(key, recall[key]*100)
                        record[key] = recall[key]*100
            else:
                list_per = ['NDCG@50']
                for key in ndcg.keys():
                    if key in list_per:
                        print(key, ndcg[key]*100)
                        record[key] = ndcg[key]*100
                        
    ## Results are saved in the following file
    with open("Result-"+dataset+".jsonl", "w") as output:
        json.dump(record, output)
