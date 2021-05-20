
from elasticsearch  import Elasticsearch
from elasticsearch.helpers import scan
from tqdm import tqdm
from pprint import pprint
from bs4 import BeautifulSoup
import re, nltk
from difflib import SequenceMatcher
import pickle
import json
import os
import math

bioclean_mod    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').replace("-", ' ').strip().lower()).split()

stopwords       = nltk.corpus.stopwords.words("english")
elk_ip          = '192.168.188.80'

def clean_start_end(word):
    word = re.sub(r'(^\W+)', r'\1 ', word)
    word = re.sub(r'(\W+$)', r' \1', word)
    word = re.sub(r'\s+', ' ', word)
    return word.strip()

def tokenize(text):
    ret = []
    for token in nltk.tokenize.word_tokenize(text):
        ret.extend(clean_start_end(token).split())
    return ret

def get_first_n(question, n):
    question    = bioclean_mod(question)
    question    = [t for t in question if t not in stopwords]
    question    = ' '.join(question)
    ################################################
    doc_index   = 'natural_questions_0_1'
    es          = Elasticsearch(['{}:9200'.format(elk_ip)], verify_certs=True, timeout=300, max_retries=10, retry_on_timeout=True)
    ################################################
    bod         = {"size": n, "query": {"match": {"paragraph_text": question}}}
    res         = es.search(index=doc_index, body=bod, request_timeout=120)
    return res['hits']['hits']

def idf_val(w, idf, max_idf):
    if w in idf:
        return idf[w]
    return max_idf

def get_all_quests(dataset):
    ################################################
    questions_index = 'natural_questions_q_0_1'
    questions_map   = "natural_questions_q_map_0_1"
    es              = Elasticsearch(['{}:9200'.format(elk_ip)], verify_certs=True, timeout=300, max_retries=10, retry_on_timeout=True)
    if(dataset is None):
        bod = {}
    else:
        bod             = {"query": {"bool": {"must": [{"term": {"dataset": dataset}}]}}}
    items           = scan(es, query=bod, index=questions_index, doc_type=questions_map)
    total           = es.count(index=questions_index, body=bod)['count']
    ################################################
    return items, total

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def load_idfs_from_df(df_path):
    print('Loading IDF tables')
    with open(df_path, 'rb') as f:
        df = pickle.load(f)
    N   = 2684631
    idf = dict(
        [
            (
                item[0],
                math.log((N*1.0) / (1.0*item[1]))
            )
            for item in df.items()
        ]
    )
    ##############
    max_idf = 0.0
    for w in idf:
        if idf[w] > max_idf:
            max_idf = idf[w]
    ##############
    print('Loaded idf tables with max idf {}'.format(max_idf))
    return idf, max_idf

training7b_train_json = {'questions': []}
bm25_top100_train_pkl = {'queries': []}
bm25_docset_train_pkl = {}

#############################################################################

all_quests,         total_quests        = get_all_quests(None)

#############################################################################

print(total_quests)

#############################################################################

pbar                                    = tqdm(all_quests, total=total_quests)
my_counter, zero_count                  = 0, 0

#############################################################################

for quest in pbar:
    quest           = quest['_source']
    qtext           = quest['question']
    short_answer    = quest['short_answer']
    long_answer     = BeautifulSoup(quest['long_answer'], 'lxml').text.strip()
    ####################
    if ('<table>' in quest['long_answer'].lower()):
        continue
    else:
        # DONE
        bm25_100_datum  = {
            'num_rel'               : 0,
            'num_rel_ret'           : 0,
            'num_ret'               : 100,
            'query_id'              : str(quest["example_id"]),
            'query_text'            : quest['question'],
            'relevant_documents'    : [],
            'retrieved_documents'   : []
        }
        #######################
        q_data          = {
            "id"            : str(quest["example_id"]),
            "body"          : quest['question'],
            "documents"     : [],
            "snippets"      : []
        }
        #######################
        qtoks           = tokenize(qtext)
        all_retr_docs   = get_first_n(qtext, 100)
        ##########################################
        kept_docs   = {}
        rank        = 0
        for ret_doc in all_retr_docs:
            rank += 1
            ##################################################################
            kept_docs[ret_doc['_id']]   = {
                u'pmid'             : ret_doc['_id'],
                u'abstractText'     : ret_doc['_source']['paragraph_text'],
                u'title'            : ret_doc['_source']['document_title']
            }
            ##################################################################
            paragraph_text              = ' '.join(tokenize(ret_doc['_source']['paragraph_text']))
            ############################################
            is_relevant = False
            if(short_answer in ret_doc['_source']['paragraph_text']):
                similarity = similar(paragraph_text, long_answer)
                if(similarity > 0.8 ):
                    # DOC IS RELEVANT
                    is_relevant                     = True
                    bm25_100_datum['num_rel']       += 1
                    bm25_100_datum['num_rel_ret']   += 1
                    bm25_100_datum['relevant_documents'].append(ret_doc['_id'])
                    q_data['documents'].append("http://www.ncbi.nlm.nih.gov/pubmed/{}".format(ret_doc['_id']))
                    q_data['snippets'].append(
                        {
                            "offsetInBeginSection"      : ret_doc['_source']['paragraph_text'].index(short_answer), # 131,
                            "offsetInEndSection"        : ret_doc['_source']['paragraph_text'].index(short_answer)+len(short_answer), # 358,
                            "text"                      : short_answer,
                            "beginSection"              : "abstract",
                            "document"                  : "http://www.ncbi.nlm.nih.gov/pubmed/{}".format(ret_doc['_id']),
                            "endSection"                : "abstract"
                        }
                    )
                else:
                    # DOC IS NOT RELEVANT
                    pass
            else:
                # DOC IS NOT RELEVANT
                pass
            ############################################
            bm25_100_datum['retrieved_documents'].append({
                    u'bm25_score'       : ret_doc['_score'],
                    u'doc_id'           : ret_doc['_id'],
                    u'is_relevant'      : is_relevant,
                    u'norm_bm25_score'  : -1.0,
                    u'rank'             : rank
                })
            ############################################
        ##########################################
        if(bm25_100_datum['num_rel_ret']==0):
            zero_count += 1
        else:
            # KEEP IT IN THE DATASET
            # update docs
            bm25_docset_train_pkl.update(kept_docs)
            # update bm25
            bm25_top100_train_pkl['queries'].append(bm25_100_datum)
            # update questions
            training7b_train_json['questions'].append(q_data)

#############################################################################

bioasq7_data    = training7b_train_json
train_data      = bm25_top100_train_pkl
train_docs      = bm25_docset_train_pkl

########################################################
# SPLIT: train | dev | test : 0.8 | 0.1 | 0.1
########################################################

dev_from    = int(len(train_data['queries']) * 0.8)
dev_to      = int(len(train_data['queries']) * 0.9)

dev_data    = {'queries': train_data['queries'][dev_from:dev_to]}
test_data   = {'queries': train_data['queries'][dev_to:]}
train_data  = {'queries': train_data['queries'][:dev_from]}

########################################################

odir        = '../data/'

with open(os.path.join(odir, 'NQ_training7b.train.dev.test.json'), 'w') as f:
    f.write(json.dumps(training7b_train_json, indent=4, sort_keys=True))

pickle.dump(bm25_docset_train_pkl,  open(os.path.join(odir, 'NQ_bioasq7_bm25_docset_top100.train.dev.test.pkl'),  'wb'))

pickle.dump(train_data,             open(os.path.join(odir, 'NQ_bioasq7_bm25_top100.train.pkl'), 'wb'))
pickle.dump(dev_data,               open(os.path.join(odir, 'NQ_bioasq7_bm25_top100.dev.pkl'),   'wb'))
pickle.dump(test_data,              open(os.path.join(odir, 'NQ_bioasq7_bm25_top100.test.pkl'),  'wb'))







