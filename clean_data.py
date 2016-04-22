import os
import pandas as pd       
from text_processing import do_clean, do_stem, fixspell
import csv
import cPickle

from config import input_prefix, clean_input_prefix 

product_desc = pd.read_csv("%s/product_descriptions.csv"%(input_prefix), encoding='ISO-8859-1')
train = pd.read_csv("%s/train.csv"%(input_prefix), encoding='ISO-8859-1')
test = pd.read_csv("%s/test.csv"%(input_prefix), encoding='ISO-8859-1')
"""
train['label'] = train['relevance'].apply(lambda x: 1 if x < 1.5 else 2 if x < 2.5 else 3)

relevance_header = ['id', 'relevance', 'label']
train[relevance_header].to_csv('%s/relevance.csv'%(clean_input_prefix ), index=False, quoting=csv.QUOTE_NONNUMERIC)


train['query'] = train['search_term'].apply(lambda x : fixspell(x))
test['query'] = test['search_term'].apply(lambda x : fixspell(x))

data_header = ["id","product_uid","product_title","query", 'product_description']

train =  pd.merge(train, product_desc, how='left', on='product_uid')
test =  pd.merge(test, product_desc, how='left', on='product_uid')

train = train.fillna("")
test = test.fillna("")

train[data_header].to_csv('%s/train.csv'%(clean_input_prefix ),  index=False, quoting=csv.QUOTE_NONNUMERIC, encoding='utf-8')
test[data_header].to_csv('%s/test.csv'%(clean_input_prefix ) , index=False, quoting=csv.QUOTE_NONNUMERIC, encoding = 'utf-8')
"""
test_ids = test['id']
with open('%s/test_ids.pkl'%(clean_input_prefix), "wb") as f:
	cPickle.dump(test_ids, f, -1)
