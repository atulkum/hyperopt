from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import cPickle
import pandas as pd
import sys
import csv
import numpy as np
import time
from models import train_predict, loss
from sklearn.cross_validation import StratifiedKFold
from pymongo import MongoClient
from bson.json_util import dumps
from bson.json_util import loads

######load common variables to use#####
from hyper_params import int_feat,param_spaces 

n_run = 3
n_folds = 3

skf = [0]*n_run
print 'Runs %i Fold %i'%(n_run, n_folds)

db = None
cv_scores = None

#####################################

global trial_counter
global log_handler

trial_counter = 0

def hyperopt_wrapper(param, feat_key, model_name, train):
    global trial_counter
    global log_handler
    trial_counter += 1

    # convert integer feat
    for f in int_feat:
        if param.has_key(f):
            param[f] = int(param[f])

    print("------------------------------------------------------------")
    print "Trial %d" % trial_counter

    print("        Model")
    print("              %s_%s" % (feat_key, model_name))
    print("        Param")
    for k,v in sorted(param.items()):
        print("              %s: %s" % (k,v))

    ## evaluate performance
    loss_cv_mean, loss_cv_std = hyperopt_obj(param, feat_key, model_name, trial_counter, train)
	
    param['model_name'] = model_name
    param['feat_key'] = feat_key
    param['timestamp'] = int(time.time())   
    param['trial_counter'] = trial_counter
    param['trial_counter'] = trial_counter
    param['loss_cv_mean'] = loss_cv_mean
    param['loss_cv_std'] = loss_cv_std

    #publish to mongodb

    cv_scores.insert_one(param)

    return {'loss': -loss_cv_mean, 'attachments': {'std': loss_cv_std}, 'status': STATUS_OK}

def hyperopt_obj(param, feat_key, model_name, trial_counter, train):
    loss_cv = np.zeros((n_run, n_folds), dtype=float)

    for run in range(n_run):
	for fold, (trainInd, validInd) in enumerate(skf[run]):
            rng = np.random.RandomState(2016 + 1000 * (run+1) + n_folds * (fold+1))
	
	    X_train = train.iloc[trainInd]
	    X_test = train.iloc[validInd]
	
            y_train = y[trainInd]
            y_test = y[validInd]
	
	    pred_test = train_predict(X_train, X_test, y_train, y_test, model_name, param)
	    loss_test = loss(y_test, pred_test)
            loss_cv[run,fold] = loss_test
	    print("Run %i Fold %i  cv Loss:  %.6f" % (run, fold, loss_test))

    loss_cv_mean = np.mean(loss_cv)
    loss_cv_std = np.std(loss_cv)
    print("              Mean: %.6f" % loss_cv_mean)
    print("              Std: %.6f" % loss_cv_std)
   
    return loss_cv_mean, loss_cv_std


if __name__ == "__main__":
	ofst = 1
	isCV = True #boolean(sys.argv[ofst]); ofst += 1
	db_name = 'opt_db' #sys.argv[ofst]; ofst += 1
	collection_name ='cust_satis' #sys.argv[ofst]; ofst += 1
	model_name = 'clf_skl_lr' #sys.argv[ofst]; ofst += 1
	feat_key = 'simple' #sys.argv[ofst]; ofst += 1
	train_file = '../custsat/input/train.csv' #sys.argv[ofst]; ofst += 1
	if isCV:
		max_evals = 10 #int(sys.argv[ofst]); ofst += 1
		if len(sys.argv) >= ofst:
			n_run =3 #int(sys.argv[ofst]); ofst += 1
			n_folds =3 #int(sys.argv[ofst]); ofst += 1
	else:
		test_file = '../custsat/input/test.csv' #sys.argv[ofst]; ofst += 1
		sub_dir = '../custsat/submissions' #sys.argv[ofst]; ofst += 1

	client = MongoClient()
	db = client[db_name]
	cv_scores = db[collection_name]

	print 'Mongo Client Initialized DB: %s, Collection: %s'%(db_name, collection_name)

	train = pd.read_csv(train_file, index_col = False)
	y = train['TARGET']
	train.drop('TARGET', axis=1, inplace=True)
	train.drop('ID', axis=1, inplace=True)

	if isCV:
		print '-------- corss validation -------'
	   
		for run in range(n_run):
			seed = 2016 + 1000*(run +1)
			skf[run] = StratifiedKFold(y , n_folds=n_folds, shuffle=True, random_state=seed)

		print("************************************************************")
		print("Search for the best params")
		param_space = param_spaces[model_name]
		trials = Trials()
		objective = lambda p: hyperopt_wrapper(p, feat_key, model_name, train)
		best_params = fmin(objective, param_space, algo=tpe.suggest,
				   trials=trials, max_evals=max_evals)
		for f in int_feat:
		    if best_params.has_key(f):
			best_params[f] = int(best_params[f])
		print("************************************************************")
		print("Best params")
		for k,v in best_params.items():
		    print "        %s: %s" % (k,v)
		trial_losses = -np.asarray(trials.losses(), dtype=float)
		best_loss_mean = max(trial_losses)
		ind = np.where(trial_losses == best_loss_mean)[0][0]
		best_loss_std = trials.trial_attachments(trials.trials[ind])['std']
		print("Loss stats")
		print("        Mean: %.6f\n        Std: %.6f" % (best_loss_mean, best_loss_std))

	else:
		print '-------- generating submission -------'
		
		test = pd.read_csv(test_file, index_col = False)
		test_ids = test['ID']
		test.drop('ID', axis=1, inplace=True)

		
		best_params = loads(dumps(cv_scores.find({'model_name':model_id, 'feat_key':feat_key}).sort([('loss_cv_mean', -1)]).limit(1)))[0]
		
		print("Best params")
		for k,v in best_params.items():
		    print "        %s: %s" % (k,v)

	    	pred_test = train_predict(train, test, y, None, model_name, best_params)
		output = pd.DataFrame({"ID": test_ids, "TARGET": pred_test})
	   	subm_path = '%s/subm_%d.csv'%(sub_dir,int(time.time()))
    		output.to_csv(subm_path, index=False)

