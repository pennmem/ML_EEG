# ML_EEG
Collection of machine learning algorithms for building memory classification. The main class implementation is 
in the ml_models.py. The HyperOptGeneric class takes in the following arguments: 
. X: features
. y: response
. learner_name: ['L1', 'L2', 'SVM', 'RF', 'XGB'] correponding to L1, L2 logistic regression, support vector machine, random forest, and 
extreme gradient boosting (XGBoost) machine. 
. search method: 'tpe' or 'random' using the hyperopt package. 
. session: labels for how to split into cross-validation fold. 

Return: 
The best-tuned model for the kind of classifier specified in learner name. 
