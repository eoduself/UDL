from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import re
from scipy.stats import entropy
import spacy

def decision_model(document_title_text, lang, max_feature=36000, gamma=0.7):

    tr_idf_model  = TfidfVectorizer(use_idf=True)
    tf_idf_vector = tr_idf_model.fit_transform(document_title_text)

    over_size = max_feature # max_features in TF-IDF
    if tf_idf_vector.shape[1] > over_size:
        print('Original size: ', tf_idf_vector.shape[1])
        over_size = tf_idf_vector.shape[1]
        tr_idf_model  = TfidfVectorizer(use_idf=True, max_features = max_feature) # helpful to prevent the sparsity problem in TF-IDF
        tf_idf_vector = tr_idf_model.fit_transform(document_title_text)

    tf_idf_array = tf_idf_vector.toarray()

    # 1. Measure (# of terms with entropy > 1) /  (# of terms with entropy <= 1).
    # 2. If it is higher than gamma, considers Pre-trained LM for extracting document embedding. 
    #    If not, uses TF-IDF for extracting document embedding.

    shannon = entropy(tf_idf_array, base=2) * (max_feature / over_size)

    #### Replace nan to zero #######
    shannon[np.isnan(shannon)] = 0
    shannon_selection = np.sum(shannon > 1)  / np.sum(shannon < 1)

    model_choice = ""
    if shannon_selection > gamma:
        model_choice = "Pre-trained LM"
        del tf_idf_array
        del tf_idf_vector
        del tr_idf_model
    else:
        model_choice = "TF-IDF"

    print("Similarity Model Decision:", model_choice)

    if model_choice == "Pre-trained LM":
        if lang == "Vietnamese":
            entropy_model_path = 'keepitreal/vietnamese-sbert'
        elif lang == "German":
            entropy_model_path = 'bert-base-german-cased'
        else:
            entropy_model_path = 'all-mpnet-base-v2'

        model = SentenceTransformer(entropy_model_path)
        print("Pre-trained LM for similarity measurement:", entropy_model_path)

        transformer_list = []

        for sample in document_title_text:
            transformer_list.append(model.encode(sample))

        transformer_array = np.array(transformer_list)

        return transformer_array, model_choice
        
    else:
        return tf_idf_array, model_choice

def decision_score(keyword_document, lang):
    if lang in ["Vietnamese", "German"]:
        from googletrans import Translator
        translator = Translator()

        for i in range(len(keyword_document)):
            if lang == "Vietnamese":
                keyword_document[i] = translator.translate(keyword_document[i], src="vi", dest="en").text
            if lang == "German":
                keyword_document[i] = translator.translate(keyword_document[i], src="de", dest="en").text


    # spacy.require_gpu() # comment this, if you meet the GPU error
    standard = spacy.load('en_core_web_trf')
    prof = spacy.load('en_core_sci_scibert')
    vocab_standard = 50265 # size of vocab is shared in https://github.com/explosion/spacy-models/releases/tag/en_core_web_trf-3.7.3
    vocab_prof = 785000 # size of vocab is shared in https://allenai.github.io/scispacy/

    collect_no_filter = []

    loc = 0
    for sentence in tqdm(keyword_document):
        sentence = re.sub("[^A-Z]", " ", sentence , 0, re.IGNORECASE)
        doc = standard(sentence)
        std_num = 0
        for ent in doc.ents:
            std_num += 1

        doc = prof(sentence)
        prof_num = 0
        for ent in doc.ents:
            prof_num += 1

        collect_no_filter.append([std_num, prof_num])

    val_std = 0
    val_prof = 0
    for num in tqdm(collect_no_filter):
        if num[1] != 0:
            val_std += num[0]
            val_prof += num[1]

    val_std /= len(collect_no_filter)
    val_prof /= len(collect_no_filter)

    if (val_std * vocab_prof) > (val_prof * vocab_standard):
        doc_type = "General"
    else:
        doc_type = "Specialized"

    print("Document is closed to", doc_type)

    return doc_type

def link_documents(document, key_loc, embed_array, model_choice, doc_type, delta=0.4):

    similarity = cosine_similarity(embed_array)
    print(f"{model_choice} is used for similarity model")

    sort_loc = np.argsort(similarity, axis=1)[:,-3:]
    val_loc = np.sort(similarity, axis=1)[:,-3:]

    high_similarity = []

    if doc_type == "General":
        key_thres = delta
    else:
        key_thres = 1 - delta

    print("Similarity score is", key_thres)

    for i in range(len(sort_loc)):
        for j in range(len(sort_loc[i])):
            if i != sort_loc[i][j] and val_loc[i][j] > key_thres:
                high_similarity.append([i, sort_loc[i][j]])

    # Generate a new document based on UDL. Concatenation is considered.
    num = 0
    denote = "NEW_DOCUMENT-"+str(num)
    new_cor_key_loc = {}

    for i in range(len(high_similarity)):
        document[denote] = {}

        if document[key_loc[high_similarity[i][0]]]['title'] != document[key_loc[high_similarity[i][1]]]['title']:
            document[denote]['title'] = document[key_loc[high_similarity[i][0]]]['title'] + ' ' + document[key_loc[high_similarity[i][1]]]['title']
        else:
            document[denote]['title'] = document[key_loc[high_similarity[i][0]]]['title']

        if document[key_loc[high_similarity[i][0]]]['text'] != document[key_loc[high_similarity[i][1]]]['text']:
            document[denote]['text'] = document[key_loc[high_similarity[i][0]]]['text'] + ' ' + document[key_loc[high_similarity[i][1]]]['text']
        else:
            document[denote]['text'] = document[key_loc[high_similarity[i][0]]]['text']

        num += 1
        new_cor_key_loc[denote] = [key_loc[high_similarity[i][0]], key_loc[high_similarity[i][1]]]
        denote = "NEW_DOCUMENT-"+str(num)

    print("Total size of documents after universal document linking:", len(document))

    return document, new_cor_key_loc
