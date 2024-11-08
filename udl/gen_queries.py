from beir.datasets.data_loader import GenericDataLoader
from beir.generation import QueryGenerator as QGen
from beir.generation.models import QGenModel
from time import time
from tqdm import tqdm
import re
import pandas as pd


def gen_queries(ori_document, document, new_cor_key_loc, 
                lang, dataset, data_path, query_aug="qgen", ques_per_passage=1, batch_size=4, max_new_tokens=32):
    print("Type of Query Augmentation:", query_aug)
    prefix = "gen"
    start_time = time()

    if query_aug == "qgen":
        if lang == "Vietnamese":
            model_path = "doc2query/msmarco-vietnamese-mt5-base-v1"
        elif lang == "German":
            model_path = "doc2query/msmarco-german-mt5-base-v1"
        else:
            model_path = "BeIR/query-gen-msmarco-t5-base-v1"

        print("Model for QGen:", model_path)

        generator = QGen(model=QGenModel(model_path))
        generator.generate(document, output_dir=data_path, ques_per_passage=ques_per_passage, prefix=prefix, batch_size=batch_size)
        if dataset != "ma-amazon":
            # Load the synthetic queries, qrels and original document
            document, gen_queries, gen_qrels = GenericDataLoader(data_path, prefix=prefix).load(split="train")
        else:
            # Load the synthetic queries, qrels and original document
            _, gen_queries, gen_qrels = GenericDataLoader(data_path, prefix=prefix).load(split="train")        

        for key in gen_qrels.keys():
            for key2 in gen_qrels[key]:
                if key2 in new_cor_key_loc.keys():
                    gen_qrels[key] = {new_cor_key_loc[key2][0]: 1, new_cor_key_loc[key2][1]: 1} # Link Query-Documents

    elif query_aug == "summarization":
        from transformers import PegasusForConditionalGeneration, PegasusTokenizer
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"

        gen_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum").to(device)
        tok = PegasusTokenizer.from_pretrained("google/pegasus-xsum")

        gen_queries = {}
        gen_qrels = {}
        num = 0
        denote_q = "NEW_QUERY-"+str(num)


        for key in tqdm(document.keys()):
            input = document[key]['text']

            if len(tok(input, truncation=True, padding="longest", return_tensors="pt")['input_ids'][0]) > tok.max_len_single_sentence:
                batch = tok(input, truncation=True, padding="longest", return_tensors="pt")
                batch = torch.tensor([batch['input_ids'][0][:tok.max_len_single_sentence].tolist()]).to(device)

            else:
                batch = tok(input, truncation=True, padding="longest", return_tensors="pt").to(device)
                batch = batch['input_ids']

            generated_ids = gen_model.generate(batch, num_return_sequences=ques_per_passage, do_sample=True)

            for i in range(ques_per_passage):
                gen_queries[denote_q] = tok.batch_decode(generated_ids, skip_special_tokens=True)[i]

                if key in new_cor_key_loc.keys():
                    gen_qrels[denote_q] = {new_cor_key_loc[key][0]: 1, new_cor_key_loc[key][1]: 1} # Link Query-Documents
                else:
                    gen_qrels[denote_q] = {key: 1}

                num += 1
                denote_q = "NEW_QUERY-"+str(num)

        document = ori_document


    elif query_aug == "open_llama":
        from transformers import LlamaTokenizer, LlamaForCausalLM
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        gen_model = LlamaForCausalLM.from_pretrained("openlm-research/open_llama_3b_v2").to(device)
        tok = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b_v2")

        gen_queries = {}
        gen_qrels = {}
        num = 0
        denote_q = "NEW_QUERY-"+str(num)

        for key in tqdm(document.keys()):
            merge_title_txt = document[key]['title'] + '. ' + document[key]['text']
            prompt = 'Q: ' + merge_title_txt + '\nA:'

            input_ids = tok(prompt, return_tensors="pt").input_ids

            if len(input_ids[0]) > tok.max_len_single_sentence:
                chunk = torch.tensor([input_ids[0][:tok.max_len_single_sentence].tolist()]).to(device)
            else:
                chunk = input_ids.to(device)

            for _ in range(ques_per_passage):
                generation_output = gen_model.generate(input_ids=chunk, do_sample=True, max_new_tokens=max_new_tokens)
                sample = tok.decode(generation_output[0][len(input_ids[0]):])
                gen_queries[denote_q] = sample[:sample.find('\n')]

                if key in new_cor_key_loc.keys():
                    gen_qrels[denote_q] = {new_cor_key_loc[key][0]: 1, new_cor_key_loc[key][1]: 1} # Link Query-Documents
                else:
                    gen_qrels[denote_q] = {key: 1}

                num += 1
                denote_q = "NEW_QUERY-"+str(num)

        document = ori_document

    elif query_aug == "flan":

        from transformers import T5Tokenizer, T5ForConditionalGeneration
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"


        tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
        gen_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(device)

        gen_queries = {}
        gen_qrels = {}
        num = 0
        denote_q = "NEW_QUERY-"+str(num)

        for key in tqdm(document.keys()):
            merge_title_txt = document[key]['title'] + '. ' + document[key]['text']

            input = '<generate_query> paragraph: ' + merge_title_txt

            if len(tok(input, return_tensors="pt")['input_ids'][0]) > tok.max_len_single_sentence:
                batch = tok(input, return_tensors="pt")
                chunk = torch.tensor([batch['input_ids'][0][:tok.max_len_single_sentence].tolist()]).to(device)
                generated_ids = gen_model.generate(chunk, num_return_sequences=ques_per_passage, do_sample=True)


            else:
                batch = tok(input, return_tensors="pt").to(device)
                generated_ids = gen_model.generate(batch["input_ids"], num_return_sequences=ques_per_passage, do_sample=True)

            for i in range(ques_per_passage):
                gen_queries[denote_q] = tok.batch_decode(generated_ids, skip_special_tokens=True)[i]

            if key in new_cor_key_loc.keys():
                gen_qrels[denote_q] = {new_cor_key_loc[key][0]: 1, new_cor_key_loc[key][1]: 1}

            else:
                gen_qrels[denote_q] = {key: 1}

            num += 1
            denote_q = "NEW_QUERY-"+str(num)

        document = ori_document

    elif query_aug == "crop":
        from sentence_splitter import split_text_into_sentences
        import random

        gen_queries = {}
        gen_qrels = {}
        num = 0
        denote_q = "NEW_QUERY-"+str(num)

        for key in document.keys():
            if document[key]['text'] != '':
                split_txt = split_text_into_sentences(document[key]['text'], language="en")

                query_pos= random.choices(range(len(split_txt)), k=ques_per_passage)

                for loc in query_pos:
                    gen_queries[denote_q] = split_txt[loc]

                    if key in new_cor_key_loc.keys():
                        gen_qrels[denote_q] = {new_cor_key_loc[key][0]: 1, new_cor_key_loc[key][1]: 1}

                    else:
                        gen_qrels[denote_q] = {key: 1}

                    num += 1
                    denote_q = "NEW_QUERY-"+str(num)

        document = ori_document

    elif query_aug == "rm3":
        import pyterrier as pt
        pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"]) # If you call multiple time, comment this out

        docs = [{'docno':key, 'text':document[key]['text'], 'title':document[key]['title']} for key in document.keys()]
        indexer = pt.index.IterDictIndexer('./index', meta={'docno':33})
        indexref = indexer.index(docs, fields=('text', 'title', 'docno'))
        index = pt.IndexFactory.of(indexref)
        bm25 = pt.BatchRetrieve(index, wmodel="BM25")
        rm3_pipe = bm25 >> pt.rewrite.RM3(index) >> bm25

        gen_queries = {}
        gen_qrels = {}
        num = 0
        denote_q = "NEW_QUERY-"+str(num)

        for doc in tqdm(docs):
            tmp_query = re.sub('[^A-Za-z0-9 ]+', '', doc['text'])
            rm3_qgen = rm3_pipe.transform(pd.DataFrame({"qid": [denote_q], "query": [tmp_query]}))

            syn_query = []
            tmp_query = rm3_qgen["query"]
            if len(tmp_query) !=0: # not empty
                tmp_query = tmp_query.unique()[0]
                for val in tmp_query.split("^")[1:-1]:
                    syn_query.append(val.split()[-1])
                syn_query = ' '.join(syn_query)

                for rank in range(ques_per_passage):
                    gen_queries[denote_q] = syn_query
                    gen_qrels[denote_q] = {rm3_qgen["docno"][rank]: 1}
                    num += 1

                    denote_q = "NEW_QUERY-"+str(num)

        document = ori_document
        
    else:
        print("Query Augmentation is not properly defined.")

    print()
    print("Time computation for Query augmentation:", int(time()-start_time), "seconds")
    print("Size of synthetic queries:",len(gen_qrels))
    print("Size of synthetic qrles:",len(gen_queries))
    print("Size of original document:",len(document))


    return document, gen_queries, gen_qrels