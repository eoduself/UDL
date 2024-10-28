# Universal Document Linking for Enhancing the Zero-Shot Retrieval 

$${\color{red}Script based codes will be available soon.}$$

* We reduce the size of the documents to 500 during generating the synthetic queries. This helps to reproduce the results quickly. You are welcome to run on the whole dataset as well but it may take time.

* Here, we show the runbook based on Jupyter Notebook (*Universal_Document_Linking_Code.ipynb*). 

* Please read the comments inside the code carefully which we tried our best to describe all the details. 

* If you want to test on MA-Amazon dataset, please run all lines in *MA_Amazon_in_BEIR.ipynb* first. This code is for converting the MA-Amazon dataset to BEIR format. 

* Setup: 

   (A) conda create -n udl python=3.10; conda activate udl
      
   (B) pip install -r requirements.txt

* The summary of steps is as follows:

   (A) Download NER Models. You only need to run this part once.

   (B) Set Parameters to do the experiments. 
   
   (C) Download Dataset. 
      
   (D) Decision of Similarity Model. 
   
   (E) Decision of Similarity Score. 
   
   (F) Link Documents based on (D) and (E). 
   
   (G) Generate Synthetic Queries. 
   
   (H) Fine-tune Retrieval Model. 
   
   (I) Evaluate Retrieval Model. 

* Our implementation is highly influenced by https://github.com/beir-cellar/beir.


