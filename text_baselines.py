import re, pickle
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, make_scorer

from sklearn import preprocessing
#from sklearn.pipeline import Pipeline

from collections import Counter

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.metrics import classification_report_imbalanced, geometric_mean_score


from twokenize import tokenize
from load_sent_dicts import load_dicts
from features_sent import generate_sentim_features

import warnings
warnings.filterwarnings('ignore')


def read_data():

    df1 = pd.read_csv('resources/annotated_tweets.csv')
    df1 = df1[(df1.sentiment!='irr')] #exclude irrelevants
    print('tweets are:',len(df1))
    
    texts= df1.text.values
    labels = df1.sentiment.values
    dates = df1.date.values
        
    new_texts, new_labels, new_dates = [], [], []
    for i in range(len(texts)):
        new_texts.append(texts[i])
        new_labels.append(labels[i])
        new_dates.append(dates[i])
    t, l, d = np.array(new_texts), np.array(new_labels, dtype='<U8'), np.array(new_dates)
    
    return [t, l, d]
    

def preprocess_text(text):
    '''basic preprocessing'''
    
    url_regex = re.compile(r"http\S+")
    mention_regex = re.compile(r"(@\w+)+")
    punct_regex = re.compile(r'[^\w]+', re.UNICODE)
    newText = ''
    
    text = text.lower()
    wordList = tokenize(text)
    for t in wordList:
        newText = newText + t + ' '
    newText = newText.strip() # remove leading & trailing whitespaces 

    # remove urls, mentions, punctuation
    result = url_regex.sub(' urlink ', newText)
    result = mention_regex.sub(' usrmention ', result)
    result = punct_regex.sub(' ', result)
    result = ' '.join(result.split())  #join into 1 string

    return result.strip()    
            

def models_oversampling(oversample_type, model_name):
    '''returns the model and its parameters for gridsearch after applying
    oversampling techniques'''
    
    if model_name =='SVM':
        Cs = [0.01, 0.1, 1, 10, 100]
        kernels=['rbf', 'linear']
        model = SVC(gamma='scale', random_state=0, class_weight="balanced")
        if oversample_type == 'smote':
            pipeline = Pipeline([('sampling', SMOTE(random_state=3,
                            sampling_strategy={'pos':418})),
                    ('classification', model)])
        if oversample_type == 'borderlinesmote':
            pipeline = Pipeline([('sampling', BorderlineSMOTE(random_state=3, 
                                            sampling_strategy={'pos':418})),
                    ('classification', model)])
        if oversample_type == 'oversampler':
            pipeline = Pipeline([('sampling', RandomOverSampler(random_state=3,
                                sampling_strategy={'pos':128})),
                    ('classification', model)])
        parameters = {'classification__C': Cs, 'classification__kernel': kernels}
    if model_name == 'RF':
        num_estimators = [100, 200, 300]
        model = RandomForestClassifier(random_state=0, class_weight="balanced")
        if oversample_type == 'smote':
            pipeline = Pipeline([
                ('sampling', SMOTE(random_state=3,
                                   sampling_strategy={'pos':128})),
            ('classification', model)])
        if oversample_type == 'borderlinesmote':
            pipeline = Pipeline([
                ('sampling', BorderlineSMOTE(random_state=3,
                                sampling_strategy={'pos':128})),
            ('classification', model)])
        if oversample_type == 'oversampler':
            pipeline = Pipeline([('sampling', RandomOverSampler(random_state=3,
                    sampling_strategy={'pos':128})),
                    ('classification', model)])
        parameters = {'classification__n_estimators': num_estimators}
  
    if model_name =='GaussianNB':
        var_smooth = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]       
        model = GaussianNB()
        if oversample_type == 'smote':
            pipeline = Pipeline([('sampling', SMOTE(random_state=3, 
                                    sampling_strategy={'pos':318})),
                    ('classification', model)])
        if oversample_type == 'borderlinesmote':
            pipeline = Pipeline([('sampling', BorderlineSMOTE(random_state=3, 
                                sampling_strategy={'pos':128})),
                    ('classification', model)])
        if oversample_type == 'oversampler':
            pipeline = Pipeline([('sampling', RandomOverSampler(random_state=3,
                    sampling_strategy={'pos':418})),
                    ('classification', model)])
        parameters = {'classification__var_smoothing': var_smooth,
                    }           
    if model_name =='LogR':
        Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        model = LogisticRegression(random_state=0,  class_weight="balanced")
        if oversample_type == 'smote':
            pipeline = Pipeline([
                ('sampling', SMOTE(random_state=3,
                                   sampling_strategy={'pos':418})),
            ('classification', model)])
        if oversample_type == 'borderlinesmote':
            pipeline = Pipeline([
                ('sampling', BorderlineSMOTE(random_state=3,
                                sampling_strategy={'pos':418})),
            ('classification', model)])
        if oversample_type == 'oversampler':
            pipeline = Pipeline([('sampling', RandomOverSampler(random_state=3,
                    sampling_strategy={'pos':418})),
                    ('classification', model)])
        parameters = {'classification__C': Cs}
                      
    return pipeline, parameters
            

def create_tweets_dataset(): 
    '''makes the dataset of tweets'''
    
    dataset = read_data() # list of arrays (texts and labels from the dataset)

    texts, labels = [], []   
    for i in range(len(dataset[0])): #dataset[0]: texts; dataset[1]: labels
        texts.append(preprocess_text(dataset[0][i]))
        labels.append(dataset[1][i])
    
    lexicons_uni, embeds, lexicons_bi = load_dicts() #load lexicons & embeddings
    #extract the features
    l, e = generate_sentim_features(texts, lexicons_uni, embeds)

    l, e = np.nan_to_num(l), np.nan_to_num(e)
    l = np.reshape(l, (-1,1))
    X = np.concatenate((l,e), axis=1)
    
    y = np.array(labels)
    print('shape of X', X.shape, 'shape of y', y.shape)
    
    return X, np.array(labels), texts


def split_dataset(X, y, texts, pct=0.8):
    '''splits the dataset in train/ test set for the 
    classification task'''

    train_lim = int(pct*len(X))
    X_train = X[:train_lim]
    X_test = X[train_lim:]
    y_train = y[:train_lim]
    y_test = y[train_lim:]
    texts_test = texts[train_lim:]
    pickle.dump(texts_test, open('pickles/sentim_text_test.p', 'wb'))

    print('shapes', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return  X_train, y_train, X_test, y_test


def unbalanced_models(model_name):
    '''returns the model and its parameters for gridsearch'''

    if model_name =='SVM':
        Cs = [0.01, 0.1, 1, 10, 100]
        kernels=['rbf', 'linear']
        model = SVC(gamma='scale', random_state=0)
        parameters = {'C': Cs, 'kernel': kernels}
    if model_name =='GaussianNB':
        var_smooth = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        model =  GaussianNB()
        parameters = {'var_smoothing': var_smooth}
    if model_name =='LogR':
        Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        model = LogisticRegression(random_state=0) 
        parameters = {'C': Cs}
    if model_name == 'RF':
        num_estimators = [100, 200, 300]
        model =  RandomForestClassifier(random_state=0)
        parameters = {'n_estimators': num_estimators}

    return model, parameters


def train_eval_classifier_unbal(X_train, y_train, X_test, y_test):
    '''trains LogR/NB/SVM without considering the class imbalance
    '''
    
    models = ['GaussianNB', 'LogR', 'SVM']
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
        y_pred = best_model.predict(X_test)  
        print('classification report')
        print(classification_report_imbalanced(y_test, y_pred))
        gmean = geometric_mean_score(y_test, y_pred, pos_label='pos')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        print(gmean, f1_macro)

        if classif == 'GaussianNB':
            text_test_set = pickle.load(open('pickles/sentim_text_test.p', 'rb'))
            text_test_set = np.array(text_test_set)
            indices = np.where(np.logical_and(y_pred == 'pos', y_test == 'pos'))

            print(text_test_set[indices])


def train_eval_classifier(X_train, y_train, X_test, y_test):
    '''trains LogR/RF/SVM on training set after oversampling
    & evaluates on test set'''
    
    models = ['SVM', 'GaussianNB', 'LogR']
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
            y_pred = best_model.predict(X_test)  
            #print('classification report')
            print(classification_report_imbalanced(y_test, y_pred))
            #gmean = geometric_mean_score(y_test, y_pred)
            gmean = geometric_mean_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            print(gmean, f1_macro)


def baseline_text_main():

    #create the tabular dataset
    '''embed, labels, texts = create_tweets_dataset()
    with open('pickles/tweets_embeddings.p', 'wb') as file:
        pickle.dump(embed, file)
    with open('pickles/tweets_labels.p', 'wb') as file:
        pickle.dump(labels, file)'''

    X = pickle.load(open('pickles/tweets_embeddings.p', 'rb'))
    y = pickle.load(open('pickles/tweets_labels.p', 'rb'))
    from collections import Counter
    print(Counter(y))
    #split the dataset
    X_train, y_train, X_test, y_test = split_dataset(X, y, texts)
    #train classifiers without oversampling
    train_eval_classifier_unbal(X_train, y_train, X_test, y_test)
    #train classifiers with SMOTE/ Borderline SMOTE/ Random oversampling
    train_eval_classifier(X_train, y_train, X_test, y_test)


baseline_text_main()
