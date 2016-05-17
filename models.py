#import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
import numpy as np

def train_predict(X_train, X_test, y_train, y_test,  model_name, param):
	
	if model_name == 'clf_xgb_tree':
		'''
		if y_test:
        		dtest_base = xgb.DMatrix(X_test, label=y_test)
		else:
        		dtest_base = xgb.DMatrix(X_test)
			
                dtrain_base = xgb.DMatrix(X_train, label=y_train)
		watchlist = []
                #watchlist  = [(dtrain_base, 'train'), (dtest_base, 'valid')]
		bst = xgb.train(param, dtrain_base, param['num_round'], watchlist, feval=xgb_loss)
                pred_test = bst.predict(dtest_base)
		'''
		print 'no xgboost' 
	elif model_name == "clf_skl_lr":
                lr = LogisticRegression(penalty="l2", dual=True, tol=1e-5,
                                            C=param['C'], fit_intercept=True, intercept_scaling=1.0,
                                            random_state=param['random_state'])
                lr.fit(X_train, y_train)
                pred_test = lr.predict(X_test)
        elif model_name == 'clf_skl_etr':
		etr = ExtraTreesClassifier(n_estimators=int(param['n_estimators']),
                                              max_features=param['max_features'],
                                              n_jobs=param['n_jobs'],
                                              random_state=param['random_state'])
                etr.fit(X_train, y_train)
                pred_test = etr.predict_proba(X_test)[:,1]

        elif model_name == 'clf_skl_rf':
		rf = RandomForestClassifier(n_estimators=param['n_estimators'],
                                               max_features=param['max_features'],
                                               n_jobs=param['n_jobs'],
                                               random_state=param['random_state'])
                rf.fit(X_train, y_train)
                pred_test = rf.predict_proba(X_test)[:,1]

        elif model_name == 'clf_skl_gbm':
		gbm = GradientBoostingClassifier(n_estimators=param['n_estimators'],
                                                    max_features=param['max_features'],
                                                    learning_rate=param['learning_rate'],
                                                    max_depth=param['max_depth'],
                                                    subsample=param['subsample'],
                                                    random_state=param['random_state'])

                gbm.fit(X_train, y_train)
                pred_test = gbm.predict_proba(X_test)[:,1]

	return pred_test

#def xgb_loss(preds, dtrain):
#    	labels = dtrain.get_label()
#    	roc = loss(labels, preds)
#	return 'roc', float(roc)
	
def loss(y_true, y_pred):
	fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
        roc_auc = metrics.auc(fpr, tpr)
	return roc_auc
