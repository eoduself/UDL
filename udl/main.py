import json
from load_dataset import load_dataset
from udl import decision_model, decision_score, link_documents
from gen_queries import gen_queries
from fine_tune import fine_tune
from evaluation import evaluation

def main():
    ## load config
    with open("udl/config.jsonl", "r") as file:
        parameter = json.load(file)

    ## load dataset    
    document, queries, qrels, document_title_text, key_loc, lang, data_path = load_dataset(parameter["dataset"], doc_size=50)

    ## decision of similarity model
    embed_array, model_choice = decision_model(document_title_text, lang, gamma=parameter["gamma"])
    print("Embedding array size:", embed_array.shape)

    ## decision of similarity score
    doc_type = decision_score(document_title_text, lang)
    print("Document type:", doc_type)

    ## link documents
    linked_document, new_cor_key_loc = link_documents(document, key_loc, embed_array, model_choice, doc_type, delta=parameter["delta"])

    ## generate synthetic queries
    train_document, syn_queries, syn_qrels = gen_queries(document, linked_document, new_cor_key_loc, lang, parameter["dataset"], data_path, 
            query_aug=parameter["query_aug"], ques_per_passage=parameter["ques_per_passage"])

    ## fine-tune retrieval model
    model_save_path = fine_tune(parameter["retrieval_model_name"], parameter["dataset"], train_document, syn_queries, syn_qrels)

    ## evaluate retrieval model
    evaluation(parameter["dataset"], data_path, model_save_path)

if __name__ == "__main__":
    main()
