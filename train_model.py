from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import cPickle
import pandas as pd
from config import log_prefix, model_prefix, feature_prefix, clean_input_prefix, n_run, n_fold
import sys
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import Ridge, Lasso, LassoLars, ElasticNet

######load common variables to use#####
with open('%s/int_feat.pkl'%(clean_input_prefix), "rb") as f:
	int_feat = cPickle.load(f)
with open("%s/cv_kfold.pkl"%(clean_input_prefix), "rb") as f:
        skf = cPickle.load(f)
with open('%s/param_spaces.pkl'%(clean_input_prefix), "rb") as f:
	param_spaces = cPickle.load(f)

relevance = pd.read_csv('%s/relevance.csv'%(clean_input_prefix))
y = relevance['relevance']
valid_ids = relevance['id']

with open('%s/test_ids.pkl'%(clean_input_prefix), "rb") as f:
	test_ids = cPickle.load(f)
#####################################
global trial_counter
global log_handler

trial_counter = 0

def hyperopt_wrapper(param, feat_key, model_name):
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
    loss_cv_mean, loss_cv_std = hyperopt_obj(param, feat_key, model_name, trial_counter)

    var_to_log = [
        "%d" % trial_counter,
        "%.6f" % loss_cv_mean,
        "%.6f" % loss_cv_std
    ]
    for k,v in sorted(param.items()):
        var_to_log.append("%s" % v)
    writer.writerow(var_to_log)
    log_handler.flush()

    return {'loss': -loss_cv_mean, 'attachments': {'std': loss_cv_std}, 'status': STATUS_OK}

def train_predict(X_train, X_test, y_train, model_name, param):
	if model_name == 'reg_skl_svr':
		X_train, X_test = X_train.toarray(), X_test.toarray()
		scaler = StandardScaler()
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.transform(X_test)
		svr = SVR(C=param['C'], gamma=param['gamma'], epsilon=param['epsilon'], degree=param['degree'], kernel=param['kernel'])
		svr.fit(X_train, y_train)
		pred_test = svr.predict(X_test)
	elif model_name == "clf_skl_lr":
		lr = LogisticRegression(penalty="l2", dual=True, tol=1e-5,
                                            C=param['C'], fit_intercept=True, intercept_scaling=1.0,
                                            class_weight='auto', random_state=param['random_state'])
                lr.fit(X_train, y_train)
		pred_test = lr.predict(X_test)
	elif model_name == "reg_skl_ridge":
                ridge = Ridge(alpha=param["alpha"], normalize=True)
                ridge.fit(X_train, y_train)
                pred_test = ridge.predict(X_test)

	return pred_test

def mse(y_true, y_pred):
	return mean_squared_error(y_true, y_pred)**0.5

def hyperopt_obj(param, feat_key, model_name, trial_counter):
    loss_cv = np.zeros((n_run, n_fold), dtype=float)

    for run in range(n_run):
	for fold, (validInd, trainInd) in enumerate(skf[run]):
            rng = np.random.RandomState(2016 + 1000 * (run+1) + 10 * (fold+1))
            
	    with open("%s/cv/%s_train_feat_run%dfold%d.pkl"%(feature_prefix, feat_key, run, fold), "rb") as f:
                        X_train = cPickle.load(f)

	    with open("%s/cv/%s_test_feat_run%dfold%d.pkl"%(feature_prefix, feat_key, run, fold), "rb") as f:
                        X_test = cPickle.load(f)
	
            y_train = y[trainInd]
            y_test = y[validInd]
            
	    pred_test = train_predict(X_train, X_test, y_train, model_name, param)
	    mse_test = mse(y_test, pred_test)
    	    
   	    output = pd.DataFrame({"id": valid_ids[validInd], "prediction": pred_test})
	    cv_out_path = "%s/cv/%s_%s_pred_run%dfold%d_trial%d.pkl"%(model_prefix, feat_key, model_name, run, fold, trial_counter)
            output.to_csv(cv_out_path , index=False)

            loss_cv[run,fold] = mse_test

    loss_cv_mean = np.mean(loss_cv)
    loss_cv_std = np.std(loss_cv)
    print("              Mean: %.6f" % loss_cv_mean)
    print("              Std: %.6f" % loss_cv_std)


    ###### store test pred for ensemble
    with open("%s/all/%s_train_feat_all.pkl"%(feature_prefix, feat_key), "rb") as f:
	X_train_all = cPickle.load(f)
    with open("%s/all/%s_test_feat_all.pkl"%(feature_prefix, feat_key), "rb") as f:
        X_test_all = cPickle.load(f)
    pred_test = train_predict(X_train_all, X_test_all, y, model_name, param)
    output = pd.DataFrame({"id": test_ids, "prediction": pred_test})
    subm_path = "%s/all/%s_%s_pred_all_trail%d.pkl"%(model_prefix, feat_key, model_name, trial_counter)
    output.to_csv(subm_path, index=False)
    
    return loss_cv_mean, loss_cv_std


if __name__ == "__main__":
	model_name = sys.argv[1]
	feat_key = sys.argv[2]

	param_space = param_spaces[model_name]

	log_file = "%s/%s_%s_hyperopt.log" % (log_prefix, feat_key, model_name )
	log_handler = open( log_file, 'wb' )
	writer = csv.writer( log_handler )
	headers = [ 'trial_counter', 'loss_mean', 'loss_std' ]
	for k,v in sorted(param_space.items()):
	    headers.append(k)
	writer.writerow( headers )
	log_handler.flush()

	print("************************************************************")
	print("Search for the best params")
	trials = Trials()
	objective = lambda p: hyperopt_wrapper(p, feat_key, model_name)
	best_params = fmin(objective, param_space, algo=tpe.suggest,
			   trials=trials, max_evals=param_space["max_evals"])
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





