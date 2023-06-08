

seed_value= 0
import numpy as np
import tensorflow as tf
import os, random, pickle
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

from keras import backend as K
tf.config.threading.set_intra_op_parallelism_threads(1)

from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, RepeatVector
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers

from keras.regularizers import l2
from keras.models import Model, load_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, f1_score

import h5py

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer, average_precision_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from imblearn.metrics import classification_report_imbalanced, geometric_mean_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler

from collections import Counter


def get_ae_dataset():
    '''returns tensors of train& dev set of the data 
    that will be used for training the autoencoder'''

    X = pickle.load(open('pickles/tweets_embeddings.p', 'rb')) 
    X_train = X[:int(0.7*len(X))]
    X_val = X[int(0.7*len(X)):int(0.8*len(X))]

    labels = pickle.load(open('pickles/tweets_labels.p', 'rb'))
    labels_train = labels[:int(0.7*len(labels))]
    X_pos_tr = X_train[labels_train == 'pos']
    pickle.dump(X_pos_tr, open('pickles/pos_tweets_train.p', 'wb'))

    return X_train, X_val


def split_train_test_dataset(X, y, pct=0.8):
    '''splits the dataset in train/ test set for the 
    classification task'''

    train_lim = int(pct*len(X))
    X_train = X[:train_lim]
    X_test = X[train_lim:]
    y_train = y[:train_lim]
    y_test = y[train_lim:]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
   
    return  X_train, y_train, X_test, y_test

 
def train_ffae():
    '''trains a feedforward autoencoder'''

    X_train, X_val = get_ae_dataset()
    
    model = Sequential()
    model.add(Dense(256, input_dim=X_train.shape[1], kernel_regularizer=l2(0.1),
                    activation='tanh'))       
           
    model.add(Dense(X_train.shape[1])) 
    
    adam = optimizers.Adam(lr=0.0001)
   
    model.compile(loss='mean_squared_error', metrics=['mse'], optimizer = adam)
    filepath = 'models/best_ffae_tweets.h5'
    
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, 
                                 save_best_only=True)    
    early_stopping_monitor = EarlyStopping(patience=5)
    
    history = model.fit(X_train, X_train,\
              batch_size=16,epochs=300,shuffle=True,verbose=2,\
              validation_data=(X_val, X_val), 
              callbacks=[early_stopping_monitor,checkpoint])

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Plot the loss and validation loss
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def sentim_models(model_name):
    '''returns the model and its parameters for gridsearch'''
    
    if model_name =='SVM':
        Cs = [ 0.01, 0.1, 1, 10, 100]
        kernels=['rbf', 'linear']        
        model = SVC(gamma='scale', random_state=0,class_weight="balanced",
                     max_iter=100000)
        parameters = {'C': Cs, 'kernel': kernels}
    if model_name =='GaussianNB':
        model =  GaussianNB()     
        var_smooth = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        parameters = {'var_smoothing': var_smooth}
    if model_name == 'LogR':
        model = LogisticRegression(random_state=0,  class_weight="balanced",
                                    max_iter=5000) 
        Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        parameters = {'C': Cs}

    return model, parameters
            


def train_eval_classifier_ff_ae(X_train, y_train):
    '''trains LogR/NB/SVM on training set & saves the models to pickles.
    '''
    
    models = ['SVM', 'GaussianNB', 'LogR']
    scorer = make_scorer(f1_score, average='macro')

    for classif in models:
        print('classifier', classif)
        model, params = sentim_models(classif)
        
        '''use RepeatedStratifiedKFold s.t. each fold of the cross-validation split  
        has the same class distribution as the original dataset'''
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
        clf = GridSearchCV(model, params, cv= cv, verbose=1, 
                           scoring=scorer
                           )
        best_model = clf.fit(X_train, y_train)
        print(clf.best_estimator_)
        y_pred = best_model.predict(X_test)  
        print('classification report')
        print(classification_report_imbalanced(y_test, y_pred))


X = pickle.load(open('pickles/tweets_embeddings.p', 'rb'))
y = pickle.load(open('pickles/tweets_labels.p', 'rb'))

#train Feedforward AE  
train_ffae()
filepath = 'models/best_ffae_tweets.h5'               
model = load_model(filepath)
pos_tweets_train = pickle.load(open('pickles/pos_tweets_train.p', 'rb'))


#generate data for the minority class
synthetic_data = []
for i in range (1):
    preds = model.predict(pos_tweets_train)
    print(preds.shape)
    synthetic_data.append(preds)
print(len(synthetic_data))

#compute metrics to evaluate the reconstruction
for instance_set in synthetic_data: 
    mse = mean_squared_error(pos_tweets_train, instance_set)
    print("MSE:", mse)

    r2 = r2_score(pos_tweets_train, instance_set)
    print("R-squared:", r2)

#add the generated data to the initial training set
X_train, y_train, X_test, y_test = split_train_test_dataset(X, y, pct=0.8)
X_train_ff_ae = list(X_train) 
for instance_set in synthetic_data:
    X_train_ff_ae.extend(instance_set)
y_train_ff_ae = list(y_train)  
y_train_ff_ae.extend(['pos'] * (1 * len(synthetic_data[0])))
print(len(X_train_ff_ae), len(y_train_ff_ae))
combined = list(zip(X_train_ff_ae, y_train_ff_ae))
random.shuffle(combined)
X_train_ff_ae, y_train_ff_ae = zip(*combined)
X_train_ff_ae, y_train_ff_ae =np.array(X_train_ff_ae), np.array(y_train_ff_ae)

#scale the data
scaler = MinMaxScaler()
X_train_ff_ae = scaler.fit_transform(X_train_ff_ae)
X_test = scaler.transform(X_test)

#train and evaluate the classifiers
train_eval_classifier_ff_ae(X_train_ff_ae, y_train_ff_ae)






