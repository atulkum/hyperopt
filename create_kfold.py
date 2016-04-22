import pandas as pd
from sklearn.cross_validation import StratifiedKFold
import cPickle
import os
from config import clean_input_prefix, n_run, n_fold

relevance = pd.read_csv('%s/relevance.csv'%(clean_input_prefix))

skf = [0]*n_run
for run in range(n_run):
	seed = 2016 + 1000*(run +1)
	skf[run] = StratifiedKFold(relevance['label'] , n_folds=n_fold, shuffle=True, random_state=seed)

with open('%s/cv_kfold.pkl'%(clean_input_prefix), "wb") as f:
	cPickle.dump(skf, f, -1)

param_spaces = {}
int_feat = []

with open('%s/param_spaces.pkl'%(clean_input_prefix), "wb") as f:
	cPickle.dump(param_spaces, f, -1)
with open('%s/int_feat.pkl'%(clean_input_prefix), "wb") as f:
	cPickle.dump(int_feat, f, -1)
