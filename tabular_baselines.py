
import pickle, random
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import  make_scorer, f1_score
 

from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from collections import Counter

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.metrics import classification_report_imbalanced

from imblearn.datasets import fetch_datasets


import warnings
warnings.filterwarnings('ignore')

def unbalanced_models(model_name):

    if model_name =='SVM':
        #Cs = [0.01, 0.1, 1, 10, 100]
        kernels=['rbf', 'linear']
        model = SVC(gamma='scale', random_state=0)
        parameters = { 'kernel': kernels}
    if model_name =='GaussianNB':
        var_smooth = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        model =  GaussianNB()
        parameters = {'var_smoothing': var_smooth}
    if model_name =='LogR':
        Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        model = LogisticRegression(random_state=0) 
        parameters = {'C': Cs}

    return model, parameters


def models_oversampling(oversample_type, model_name):
    
    if model_name =='SVM':
        Cs = [0.01, 0.1, 1, 10, 100]
        kernels=['rbf', 'linear']
        model = SVC(gamma='scale', random_state=0, class_weight="balanced")
        if oversample_type == 'smote':
            pipeline = Pipeline([('sampling', SMOTE(random_state=3, k_neighbors=1)),
                    ('classification', model)])
        if oversample_type == 'borderlinesmote':
            pipeline = Pipeline([('sampling', BorderlineSMOTE(random_state=3, k_neighbors=1)),
                    ('classification', model)])
        if oversample_type == 'oversampler':
            pipeline = Pipeline([('sampling', RandomOverSampler(random_state=3)),
                    ('classification', model)])
        parameters = {'classification__C': Cs, 'classification__kernel': kernels}
  
    if model_name =='GaussianNB':
        var_smooth = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]       
        model = GaussianNB()
        if oversample_type == 'smote':
            pipeline = Pipeline([('sampling', SMOTE(random_state=3, k_neighbors=1)),
                    ('classification', model)])
        if oversample_type == 'borderlinesmote':
            pipeline = Pipeline([('sampling', BorderlineSMOTE(random_state=3, k_neighbors=1)),
                    ('classification', model)])
        if oversample_type == 'oversampler':
            pipeline = Pipeline([('sampling', RandomOverSampler(random_state=3)),
                    ('classification', model)])
        parameters = {'classification__var_smoothing': var_smooth,
                    }           
    if model_name =='LogR':
        Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        model = LogisticRegression(random_state=0,  class_weight="balanced")
        if oversample_type == 'smote':
            pipeline = Pipeline([
                ('sampling', SMOTE(random_state=3, k_neighbors=1)),
            ('classification', model)])
        if oversample_type == 'borderlinesmote':
            pipeline = Pipeline([
                ('sampling', BorderlineSMOTE(random_state=3, k_neighbors=1)),
            ('classification', model)])
        if oversample_type == 'oversampler':
            pipeline = Pipeline([('sampling', RandomOverSampler(random_state=3)),
                    ('classification', model)])
        parameters = {'classification__C': Cs}
                      
    return pipeline, parameters
            

def split_dataset(name, X, y, pct=0.8):

    if name == 'webpage':
        '''combined = list(zip(X, y))
        random.shuffle(combined)
        X, y = zip(*combined)
        X, y =np.array(X), np.array(y)
        pickle.dump([X, y], open('pickles/webpage_dataset.p', 'wb'))
        '''
        X, y = pickle.load(open('pickles/webpage_dataset.p', 'rb'))        

    train_lim = int(pct*len(X))
    X_train = X[:train_lim]
    X_test = X[train_lim:]
    y_train = y[:train_lim]
    print(Counter(y_train))
    y_test = y[train_lim:]

    print('shapes', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return  X_train, y_train, X_test, y_test


def get_preds(X_test, y_test, best_model):
    '''make predictions for test set & print classif report'''   

    y_pred = best_model.predict(X_test)  
    print('classification report')
    print(classification_report_imbalanced(y_test, y_pred))
    
        
def train_eval_classifier_unbal(name, X, y):
    '''trains LogR/NB/SVM without considering the class imbalance
    '''
    
    X_train, y_train, X_test, y_test = split_dataset(name, X, y)
    models = ['SVM', 'GaussianNB', 'LogR']
    models = ['SVM']
    scorer = make_scorer(f1_score, average='macro')

    for classif in models:
        print(classif)
        model, params = unbalanced_models(classif)
        
        '''use RepeatedStratifiedKFold s.t. each fold of the cross-validation split  
        has the same class distribution as the original dataset'''
        cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=1)
        clf = GridSearchCV(model, params, cv= cv, verbose=1, 
                           scoring=scorer
                           )
        best_model = clf.fit(X_train, y_train)
        get_preds(X_test, y_test, best_model)
    

def train_eval_classifier_smote(name, X, y):
    '''trains LogR/NB/SVM on training set using SMOTE for class imbalance'''
    
    X_train, y_train, X_test, y_test = split_dataset(name, X, y)
    #models = ['GaussianNB', 'LogR']
    models = [ 'SVM']
    
    scorer = make_scorer(f1_score, average='macro')

    for classif in models:
        types_of_oversampling = ['smote', 'borderlinesmote', 'oversampler']
        print('classifier', classif)
        for oversample_type in types_of_oversampling:
            print('type of oversampling:', oversample_type)
            model, params = models_oversampling(oversample_type, classif)
        
        '''use RepeatedStratifiedKFold s.t. each fold of the cross-validation split  
        has the same class distribution as the original dataset'''
        cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=1)
        clf = GridSearchCV(model, params, cv= cv, verbose=1, 
                           scoring=scorer
                           )
        best_model = clf.fit(X_train, y_train)
        get_preds(X_test, y_test, best_model)
    



datasets = ['arrhythmia','mammography', 'webpage' ]
for name in datasets:
    print('dataset:', name)
    dataset = fetch_datasets()[name]
    # Access the features (X) and labels (y)
    X = dataset.data
    y = dataset.target
    print('counting instances of each class', Counter(y))
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)

    train_eval_classifier_unbal(name, X, y)
    train_eval_classifier_smote(name, X, y)



