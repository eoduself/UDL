# Universal Document Linking for Enhancing the Zero-Shot Retrieval 

> [!NOTE]
> We plan to share retrieval models fine-tuned using the Universal Document Linking (UDL) method.

Thank you for visiting our repository for *Link, Synthesize, Retrieve: Universal Document Linking for Zero-Shot Information Retrieval*, presented at EMNLP 2024.

This repository includes two code versions:

- Notebook: Provides an overview of the entire algorithm flow, making it easy to understand and run end-to-end.
- Script: Offers a modular version, ideal for deeper exploration and customization of each component.

We hope you find these resources helpful!

## The summary of steps is as follows:
   (A) Download NER Models. You only need to run this part once.

   (B) Set Parameters to do the experiments. 
   
   (C) Download Dataset. 
      
   (D) Decision of Similarity Model. 
   
   (E) Decision of Similarity Score. 
   
   (F) Link Documents based on (D) and (E). 
   
   (G) Generate Synthetic Queries. 
   
   (H) Fine-tune Retrieval Model. 
   
   (I) Evaluate Retrieval Model. 

## Setup:
   (A) Create virtual environment. 
   ```
   conda create -n [YOUR_VENV_NAME] python=3.10
   ```
 
   (B) Install the required packages.
   ```
   pip install -r requirements.txt
   ```

   (C) Download NER Models. You are welcome to try the other NER models. 
   ```
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_scibert-0.5.1.tar.gz

   python -m spacy download en_core_web_trf
   ```
   
   (D) Some extra setups:
   - You may need to upgrade transformers to avoid the error during decision of similarity score   
   ```
   pip install transformers==4.30.2    
   ```  
   - For Muennighoff/SGPT-125M-weightedmean-msmarco, install the following
   ```
   pip install --upgrade git+https://github.com/Muennighoff/sentence-transformers.git@sgpt_poolings_specb
   ```
   - For open_llama, install the following
   ```
   pip install protobuf==3.20
   ```
   - For rm3, install the following
   ```
   pip install python-terrier
   ```
   - For multiple calling of rm3, remove index folder
   ```
   rm index -r
   ```      
   - For MA-Amazon, run following command to download data
   ```
   git clone https://github.com/amazon-science/esci-data.git to download data
   ```      

## Runbook:
   (A) 
   
   (B) 
   
   (C) 
   
   (D) 
   
   (E) 
   
   (F) 

- If you want to test on MA-Amazon dataset, please run all lines in *MA_Amazon_in_BEIR.ipynb* or *python udl/ma_amazon_in_beir.py* first. This code is for converting the MA-Amazon dataset to BEIR format. 

- Our implementation is highly influenced by https://github.com/beir-cellar/beir.


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
