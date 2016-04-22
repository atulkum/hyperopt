import pandas as pd
import cPickle
import os
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from config import feature_prefix, clean_input_prefix, n_run, n_fold

stemmer = PorterStemmer()
token_pattern = r"(?u)\b\w\w+\b"

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (stemmer.stem(w) for w in analyzer(doc))

with open("%s/cv_kfold.pkl"%(clean_input_prefix), "rb") as f:
	skf = cPickle.load(f)

train = pd.read_csv('%s/train.csv'%(clean_input_prefix))

feature_key = 'tfidf50'

is_first = True
for run in range(n_run):
	for fold, (validInd, trainInd) in enumerate(skf[run]):
		if is_first:	
			tfv = StemmedTfidfVectorizer(max_features=50,
                                 strip_accents='unicode', analyzer='word', token_pattern=token_pattern,
                                 ngram_range=(1,3), use_idf=1, 
                                 stop_words = None, norm='l2')
			tf_idf_feat = tfv.fit_transform(train["product_title"] + train["query"] + train["product_description"])
			
			is_first = False

		train_feat = tf_idf_feat[trainInd]
		test_feat = tf_idf_feat[validInd]

		with open("%s/cv/%s_train_feat_run%dfold%d.pkl"%(feature_prefix, feature_key, run, fold), "wb") as f:
			cPickle.dump(train_feat, f, -1)
		
		with open("%s/cv/%s_test_feat_run%dfold%d.pkl"%(feature_prefix, feature_key, run, fold), "wb") as f:
			cPickle.dump(test_feat, f, -1)

test = pd.read_csv('%s/test.csv'%(clean_input_prefix))
all = pd.concat([train, test])
n_train = len(train)

tfv = StemmedTfidfVectorizer(max_features=500,
	strip_accents='unicode', analyzer='word', token_pattern=token_pattern,
	ngram_range=(1,3), use_idf=1, 
	stop_words = None, norm='l2')

tf_idf_feat = tfv.fit_transform(all["product_title"] + all["query"] + all["product_description"])

train_feat = tf_idf_feat[:n_train]
test_feat = tf_idf_feat[n_train:]

with open("%s/all/%s_train_feat_all.pkl"%(feature_prefix, feature_key), "wb") as f:
	cPickle.dump(train_feat, f, -1)
		
with open("%s/all/%s_test_feat_all.pkl"%(feature_prefix,feature_key), "wb") as f:
	cPickle.dump(test_feat, f, -1)
