# Universal Document Linking for Enhancing the Zero-Shot Retrieval 

> [!NOTE]
> We plan to share retrieval models fine-tuned using the Universal Document Linking (UDL) method.

Thank you for visiting our repository for *Link, Synthesize, Retrieve: Universal Document Linking for Zero-Shot Information Retrieval*, presented at EMNLP 2024.

This repository includes two code versions:

- Notebook: Provides an overview of the entire algorithm flow, making it easy to understand and run end-to-end.
- Script: Offers a modular version, ideal for deeper exploration and customization of each component.

We hope you find these resources helpful!

## Summary of Steps:

   (A) Download NER Models – Run this step once to set up the necessary NER models.

   (B) Set Parameters to do the experiments - Configure parameters for the experiments.
   
   (C) Download Dataset - Acquire the required dataset.
      
   (D) Decision of Similarity Model - Choose the similarity model to extract document embedding.
   
   (E) Decision of Similarity Score - Define the similarity score to link.
   
   (F) Link Documents - Link documents based on the chosen similarity model and similarity score.
   
   (G) Generate Synthetic Queries - Create synthetic queries for training.
   
   (H) Fine-tune Retrieval Model - Train the retrieval model with the documents and synthetic queries.
   
   (I) Evaluate Retrieval Model - Assess the performance of the retrieval model.

## Setup:
   (A) Create a Virtual Environment: 
   ```
   conda create -n [YOUR_VENV_NAME] python=3.10
   ```
 
   (B) Install Required Packages:
   ```
   pip install -r requirements.txt
   ```

   (C) Download NER Models:

   You can try different NER models. Here are some suggested installations:
 
   ```
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_scibert-0.5.1.tar.gz

   python -m spacy download en_core_web_trf
   ```
   
   (D) Additional Setup Steps:
   - Transformers Library: To avoid potential errors when calculating similarity scores, use the following transformers version:
   ```
   pip install transformers==4.30.2    
   ```  
   - SGPT Model: For Muennighoff/SGPT-125M-weightedmean-msmarco, install this specific version:
   ```
   pip install --upgrade git+https://github.com/Muennighoff/sentence-transformers.git@sgpt_poolings_specb
   ```
   - Open LLaMA Model: Install the following:
   ```
   pip install protobuf==3.20
   ```
   - RM3: Install python-terrier for RM3 support:
   ```
   pip install python-terrier
   ```   
   To clear multiple calls of RM3, delete the index folder:   
   ```
   rm index -r
   ```      
   - MA-Amazon Dataset: Clone the repository to download the dataset:
   ```
   git clone https://github.com/amazon-science/esci-data.git 
   ```      

## Runbook:
All the following files are located inside `udl` folder

   (A) Define the experimental parameters inside `config.jsonl`:
   ```
   - dataset: Specify a dataset that you want to try: nfcorpus, scifact, arguana, scidocs, climate-fever, trec-covid, quora, germanquad, vihealthqa, ma-amazon
   - query_aug: Specify the query augmentation that you want to try: crop, rm3, summarization, flan, open_llama, qgen.
      Note that qgen is only one tested on Vietnamese and German datasets.
   - ques_per_passage: Number of synthetic quries per document
   - retrieval_model_name: Specify a retrieval model that you want to try.
      For English, all-mpnet-base-v2, msmarco-distilbert-base-v3, Muennighoff/SGPT-125M-weightedmean-msmarco, sentence-transformers/distiluse-base-multilingual-cased-v2
      For German, svalabs/bi-electra-ms-marco-german-uncased, T-Systems-onsite/cross-en-de-roberta-sentence-transformer, sentence-transformers/distiluse-base-multilingual-cased-v2
      For Vietnamese, VoVanPhuc/sup-SimCSE-VietNamese-phobert-base, keepitreal/vietnamese-sbert, sentence-transformers/distiluse-base-multilingual-cased-v2    
   - gamma: gamma value for decision of similarity model
   - delta: delta value for decision of similarity score
   ```

   (B) Include python path:
   ```
   export PYTHONPATH=${PYTHONPATH}:/content/udl
   ```
   
   (C) Run the experiment: 
   ```
   python udl/main.py
   ```
   
   (D) Additional Notes:
   
   - MA-Amazon Dataset: Run the following to convert the MA-Amazon dataset to BEIR format. 
   ```
   python udl/ma_amazon_in_beir.py
   ```      
    

### Our implementation is highly influenced by https://github.com/beir-cellar/beir.

### If you use this code or build upon this idea, please cite the following paper:
```bibtext
@inproceedings{hwang-etal-2024-link,
    title = "Link, Synthesize, Retrieve: Universal Document Linking for Zero-Shot Information Retrieval",
    author = "Hwang, Dae Yon  and
      Taha, Bilal  and
      Pande, Harshit  and
      Nechaev, Yaroslav",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.1056",
    pages = "18971--18982",
    abstract = "Despite the recent advancements in information retrieval (IR), zero-shot IR remains a significant challenge, especially when dealing with new domains, languages, and newly-released use cases that lack historical query traffic from existing users. For such cases, it is common to use query augmentations followed by fine-tuning pre-trained models on the document data paired with synthetic queries. In this work, we propose a novel Universal Document Linking (UDL) algorithm, which links similar documents to enhance synthetic query generation across multiple datasets with different characteristics. UDL leverages entropy for the choice of similarity models and named entity recognition (NER) for the link decision of documents using similarity scores. Our empirical studies demonstrate the effectiveness and universality of the UDL across diverse datasets and IR models, surpassing state-of-the-art methods in zero-shot cases. The developed code for reproducibility is included in https://github.com/eoduself/UDL",
}
```
