from beir import util
from beir.datasets.data_loader import GenericDataLoader
import json

def load_dataset(dataset, doc_size=None):
    print("Dataset:", dataset)
    
    if dataset != "ma-amazon":

        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        data_path = util.download_and_unzip(url, 'datasets')

        document, queries, qrels = GenericDataLoader(data_path).load(split="test")

        ori_document = document

        # Detect the language of a dataset
        if dataset == "vihealthqa":
            lang = "Vietnamese"
        elif dataset == "germanquad":
            lang = "German"
        else:
            lang = "English"

    else:
        data_path = "esci-data/recall/" ### Please check the saved location of ma-amazon dataset if you meet the FileNotFoundError
        with open(data_path + "corpus.jsonl", "r") as file:
            for line in file:
                document = json.loads(line)

        with open(data_path + "queries.jsonl", "r") as file:
            for line in file:
                queries = json.loads(line)

        with open(data_path + "qrels.jsonl", "r") as file:
            for line in file:
                qrels = json.loads(line)

        lang = "English"

    print("Language of dataset:", lang)

    # Here, we reduce the size of documents (500) during generating the synthetic queries.
    # This helps to reproduce the results and ideas quickly.
    # You are welcome to run on the whole dataset as well but it may take time.
    if doc_size != None:
        loc = 0
        new_document = {}
        for key, val in document.items():
            new_document[key] = val
            loc += 1
            if loc == doc_size:
                break

        document = new_document
    print("Size of Document:", len(document))
    print("Size of Queries:", len(queries))
    print("Size of Qrels:", len(qrels))

    # Merge the title and text in each document
    key_loc = []
    document_title_text = []

    for key in document.keys():
        document_title_text.append(document[key]['title'] + '. ' + document[key]['text'])
        key_loc.append(key)

    return document, queries, qrels, document_title_text, key_loc, lang, data_path