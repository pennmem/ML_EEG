from hyperopt import hp, fmin, STATUS_OK
from sklearn.linear_model import LogisticRegression  # L2
from sklearn.ensemble import RandomForestClassifier as RF  # random forests
from sklearn import svm # svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneGroupOut # leave one group out
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from ml_models import*
import pandas as pd 
import numpy as np
import hyperopt

class HyperOptGeneric: 
    
    '''A class that return the optimal set of hyperparameters for a machine learning algorithm using Sequential Model-Based Optimization
        ----------
        Attributes: 
            X: features
            y: labels
            learner_name: the name of the ML algo (L1,L2,SVM,RF,Xgboost)
            search_method: 
            ind_params: individual-specific for the learner
            n_folds: number of cross-validation folds, default to 5
    '''
    
    def __init__(self, X, y, learner_name, search_method, n_folds = 5, session = None):
        self.X = X 
        self.y = y 
        self.learner = None
        self.space4classifier = None
        self.learner_name = learner_name
        self.search_method = search_method 
        self.ind_params = None
        self.n_folds = n_folds
        self.session = session
        self.best = 0.0
        
        
    def get_cv_score(self, params): 
        
        # create classifier
        
        if self.session is not None: 
            logo = LeaveOneGroupOut() 
            cv_generator = logo.split(self.X,self.y,self.session)  # split by session
        else: 
            skf = StratifiedKFold(n_splits = self.n_folds)  # create stratified folds
            cv_generator = skf.get_n_splits(self.X,self.y) # create generator for these folds        
        clf = self.get_classifier(params)
    
        cross_scores = cross_val_score(clf, self.X, self.y, cv = cv_generator, scoring = 'roc_auc')  # compute cross-validation score 
        
        return cross_scores.mean()  # return average score across folds
    
    
    # create an objective function for evaluating the CV performance
    def objective(self, params): 
        
        score = self.get_cv_score(params)  # auc 
        if score > self.best: 
            self.best = score
            print('new best:', self.best, params)
        
        return({'loss': -score, 'status':STATUS_OK})  # want to optimize score or minimize -score

    # generate classifier based on leaner_name and their corresponding search space
    def get_classifier(self, params):    
        
        if self.learner_name == 'L1':
            self.ind_params = {'class_weight':'balanced', 'solver':'liblinear', 'penalty':'l1'}
            joint_params = self.ind_params.copy()
            joint_params.update(params)
            print(joint_params)
            clf = LogisticRegression(**joint_params)
            self.space4classifier ={'C': hp.loguniform('C',-10,10)}

        if self.learner_name == 'L2': 
            self.ind_params = {'class_weight':'balanced', 'solver':'liblinear', 'penalty':'l2'}
            joint_params = self.ind_params.copy()
            joint_params.update(params)
            print(joint_params)
            clf = LogisticRegression(**joint_params)
            self.space4classifier ={'C': hp.loguniform('C',-5,5)}
        
        if self.learner_name == 'SVM': 
            
            n_obs = len(y)
            n_pos, n_neg = np.sum(y), n_obs- np.sum(y)
            pos_weight = n_obs/2.0/n_pos
            neg_weight = n_obs/2.0/n_neg
            
            self.ind_params =  {'class_weight': {0:neg_weight,1:pos_weight}}
            joint_params = self.ind_params.copy()
            joint_params.update(params)
            joint_params.update({'probability': True})
            clf = svm.SVC(**joint_params)
            
        if self.learner_name == 'RF': 
            self.ind_params = {'class_weight': 'balanced', 'n_jobs':20}
            joint_params = self.ind_params.copy()
            joint_params.update(params)
            clf = RF(**joint_params)
            
            
        if self.learner_name == 'XGB':
            
            n_obs = len(self.y)  
            n_pos, n_neg = np.sum(self.y), n_obs- np.sum(self.y)  # calculate weights for the pos/neg classes
            self.ind_params = {'objective': 'reg:logistic', 'scale_pos_weight':n_pos/n_neg*1.0}
            joint_params = self.ind_params.copy()
            joint_params.update(params)
            clf = xgb.XGBClassifier(**joint_params)
        
        return clf
    
    def get_space4classifier(self):
        if self.learner_name == 'L1':
            self.space4classifier ={'C': hp.loguniform('C',-5,5)}

        if self.learner_name == 'L2': 
            self.space4classifier ={'C': hp.loguniform('C',-5,5)}
        
        if self.learner_name == 'SVM': 
            
            self.space4classifier ={
              'C': hp.uniform('C', 0,20),
              'gamma': hp.uniform('gamma', 0,20),
              #'gamma': hp.choice('gamma', ['auto']),
              'kernel':hp.choice('kernel', ['rbf', 'poly', 'sigmoid']),
                'normalize': hp.choice('normalize', [0,1])}
              #'degree':hp.choice('degree', np.arange(3,6))}
            
        if self.learner_name == 'RF': 
            self.space4classifier ={
              'n_estimators': hp.choice('n_estimators', np.arange(200,2000,200)),
              'max_depth': hp.choice('max_depth', np.arange(1,50,step =3)),
              'min_samples_leaf': hp.choice('min_samples_leaf', np.arange(1,10,step = 2))}
        
        if self.learner_name == "XGB": 
            self.space4classifier ={
                  'n_estimators': hp.choice('n_estimators', np.arange(100,500, step = 100)),
                  'max_depth': hp.choice('max_depth', np.arange(1,10,1)),
                  'learning_rate': hp.choice('learning_rate',[0.01,0.1,1.0]),
                  'reg_lambda': hp.loguniform('reg_lambda',-5,5)}
#                   'min_child_weight':hp.choice('min_child_weight',np.arange(3.0,7.0, step =2.0))}
#                   'scale_pos_weight':hp.uniform('scale_pos_weight',1,10),
#                   'subsample': hp.uniform('subsample',0.7,1.0)}

        
            
    def optimize_hyperparams(self): 
        
        self.get_space4classifier()
        
        if self.search_method == 'rand': 
            best_params = fmin(self.objective, self.space4classifier, algo = hyperopt.rand.suggest, max_evals = 50, rstate = np.random.RandomState(100))
        if self.search_method == 'tpe': 
            best_params = fmin(self.objective, self.space4classifier, algo = hyperopt.tpe.suggest, max_evals = 50, rstate = np.random.RandomState(100))
        
        best_classifier = self.get_classifier(best_params)
        return {'best_params':best_params, 'best_classifier':best_classifier}