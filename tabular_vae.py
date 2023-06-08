import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse

from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle, random

import matplotlib.pyplot as plt
import h5py

from keras.models import Model, load_model
from keras import optimizers

from sklearn.metrics import mean_squared_error, r2_score,\
        mean_absolute_percentage_error, f1_score, make_scorer

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from keras.regularizers import l1, l2

from imblearn.datasets import fetch_datasets
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler


from imblearn.metrics import classification_report_imbalanced, geometric_mean_score

from collections import Counter


def get_data(name):
    '''Access the features (X) and labels (y)'''

    arrhythmia = fetch_datasets()[name]
    X = arrhythmia.data
    y = arrhythmia.target
    print(Counter(y))
    # Print the shape of X and y
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)
    zero_count = X.size - np.count_nonzero(X)
    print("Percentage of zeros:", zero_count/X.size)

    return X, y


def get_ae_dataset(X,y):
    '''returns arrays of train& dev set of the data 
    that will be used for training the autoencoder'''

    X_train = X[:int(0.7*len(X))]
    X_val = X[int(0.7*len(X)):int(0.8*len(X))]
    y_train = y[:int(0.7*len(X))]
    y_val = y[int(0.7*len(X)):int(0.8*len(X))]
    X_train_minority = X_train[y_train == 1]
    X_val_minority = X_val[y_val == 1]

    #return X_train_minority, X_val_minority
    return X_train, X_val


def split_train_test_dataset(name, X, y, pct=0.8):
    '''splits the dataset in train/ test set for the 
    classification task'''

    if name == 'webpage':
        X, y = pickle.load(open('pickles/webpage_dataset.p', 'rb'))        
    train_lim = int(pct*len(X))
    X_train = X[:train_lim]
    X_test = X[train_lim:]
    y_train = y[:train_lim]
    y_test = y[train_lim:]
    
    return  X_train, y_train, X_test, y_test


def sampling(args):
    # reparameterization trick
    # instead of sampling from Q(z|x), sample eps = N(0,I)
    # then x = x_mean + x_sigma*eps= x_mean + sqrt(e^(x_log_var))*eps = x_mean + e^(0.5 * x_log_var)*eps
    latent_dim, x_mean, x_log_var = args
    epsilon = K.random_normal(shape=(K.shape(x_mean)[0], latent_dim), mean=0.,
                              stddev=1.0)
    return x_mean + K.exp(0.5 * x_log_var) * epsilon # (e^a)^b=e^ab


def train_vae_arrhythmia(latent_dim, X, y):
    '''trains a feedforward autoencoder of 1 encoder & 1 decoder'''

    K.clear_session()
    original_dim = X.shape[1]
    intermediate_dim = 128

    # Variational autoencoder model
    inputs = Input(shape=(original_dim,))
    encoded = Dense(intermediate_dim, activation='tanh',)(inputs)
    x_mean = Dense(latent_dim, activation='tanh')(encoded)
    x_log_var = Dense(latent_dim, activation='tanh')(encoded) 

    x = Lambda(sampling, output_shape=(latent_dim,))([latent_dim, x_mean, x_log_var])

    decoded = Dense(intermediate_dim, activation='tanh')(x)
    decoded = Dense(original_dim)(decoded)
    vae = Model(inputs, decoded, name='vae')
    print (vae.summary())

    # Create the loss function and compile the model
    # The loss function as defined by paper Kingma
    reconstruction_loss = mse(inputs, decoded)
    kl_loss =  -0.5 * K.sum(1 + x_log_var - K.square(x_mean) -K.exp(x_log_var), axis=-1)
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    adam = optimizers.Adam(lr=0.0001)
    vae.compile(optimizer=adam)

    # encoder model (first part of the variotional autoencoder) 
    encoder = Model(inputs, [x_mean, x_log_var, x], name='encoder')

    # decoder model 
    encoded_input = Input(shape=(latent_dim,))
    decoder_layer = vae.layers[5](encoded_input)
    decoder_layer = vae.layers[6](decoder_layer)
    decoder = Model(encoded_input, decoder_layer, name='decoder')

    X_train, X_val = get_ae_dataset(X,y)

    filepath = 'models/best_vae_arrhythmia.h5'

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, 
                                    save_best_only=True)    
    early_stopping_monitor = EarlyStopping(patience=5)

    history = vae.fit(X_train, X_train,\
                batch_size=2,epochs=300,shuffle=True,verbose=2,\
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

    return encoder, decoder


def train_vae_mammography(latent_dim, X, y):
    '''trains a feedforward autoencoder of 1 encoder & 1 decoder'''

    K.clear_session()
    original_dim = X.shape[1]

    # Variational autoencoder model
    inputs = Input(shape=(original_dim,))
    x_mean = Dense(latent_dim, activation='tanh')(inputs)
    x_log_var = Dense(latent_dim, activation='tanh')(inputs) 

    x = Lambda(sampling, output_shape=(latent_dim,))([latent_dim, x_mean, x_log_var])

    decoded = Dense(original_dim)(x)
    vae = Model(inputs, decoded, name='vae')
    print (vae.summary())

    # Create the loss function and compile the model
    # The loss function as defined by paper Kingma
    reconstruction_loss = mse(inputs, decoded)
    kl_loss =  -0.5 * K.sum(1 + x_log_var - K.square(x_mean) -K.exp(x_log_var), axis=-1)
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    adam = optimizers.Adam(lr=0.0001)
    vae.compile(optimizer=adam)

    # encoder model (first part of the variotional autoencoder) 
    encoder = Model(inputs, [x_mean, x_log_var, x], name='encoder')

    # decoder model 
    encoded_input = Input(shape=(latent_dim,))
    decoder_layer = vae.layers[4](encoded_input)
    decoder = Model(encoded_input, decoder_layer, name='decoder')

    X_train, X_val = get_ae_dataset(X,y)

    filepath = 'models/best_vae_mammography.h5'

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, 
                                    save_best_only=True)    
    early_stopping_monitor = EarlyStopping(patience=10)

    history = vae.fit(X_train, None,\
                batch_size=8,epochs=300,shuffle=True,verbose=2,\
                validation_data=(X_val, None), 
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

    return encoder, decoder


def train_vae_webpage(latent_dim, X, y):
    '''trains a feedforward autoencoder of 1 encoder & 1 decoder'''

    K.clear_session()
    original_dim = X.shape[1]
    intermediate_dim = 64

    # Variational autoencoder model
    inputs = Input(shape=(original_dim,))
    encoded = Dense(intermediate_dim, activation='tanh', kernel_regularizer=l2(0.01))(inputs)
    x_mean = Dense(latent_dim, activation='tanh', kernel_regularizer=l2(0.01))(encoded)
    x_log_var = Dense(latent_dim, activation='tanh', kernel_regularizer=l2(0.01))(encoded) 

    x = Lambda(sampling, output_shape=(latent_dim,))([latent_dim, x_mean, x_log_var])

    decoded = Dense(intermediate_dim, activation='tanh', kernel_regularizer=l2(0.01))(x)
    decoded = Dense(original_dim)(decoded)
    vae = Model(inputs, decoded, name='vae')
    print (vae.summary())

    # Create the loss function and compile the model
    # The loss function as defined by paper Kingma
    reconstruction_loss = mse(inputs, decoded)
    kl_loss =  -0.5 * K.sum(1 + x_log_var - K.square(x_mean) -K.exp(x_log_var), axis=-1)
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    adam = optimizers.Adam(lr=0.0001)
    vae.compile(optimizer=adam)

    # encoder model (first part of the variotional autoencoder) 
    encoder = Model(inputs, [x_mean, x_log_var, x], name='encoder')

    # decoder model 
    encoded_input = Input(shape=(latent_dim,))
    decoder_layer = vae.layers[5](encoded_input)
    decoder_layer = vae.layers[6](decoder_layer)
    decoder = Model(encoded_input, decoder_layer, name='decoder')

    X_train, X_val = get_ae_dataset(X,y)

    filepath = 'models/best_vae_arrhythmia.h5'

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, 
                                    save_best_only=True)    
    early_stopping_monitor = EarlyStopping(patience=5)

    history = vae.fit(X_train, X_train,\
                batch_size=128,epochs=300,shuffle=True,verbose=2,\
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

    return encoder, decoder


def generate_synthetic_vae(name, X, y, X_train, y_train):
    '''generate data for the minority class using Variational autoencoder'''

    #train Variational AE  
    if name == 'arrhythmia':
        latent_dim = 128
        encoder, decoder = train_vae_arrhythmia(latent_dim, X,y)
    if name == 'mammography':      
        latent_dim = 4
        encoder, decoder = train_vae_mammography(latent_dim, X,y)    
    if name == 'webpage':      
        latent_dim = 128
        encoder, decoder = train_vae_webpage(latent_dim, X,y)

    #generate data for the minority class
    X_train_minority = X_train[y_train == 1]
    synthetic_data = []
    for i in range (1):    
        generated_data = []
        for instance in X_train_minority:
            encoded_instance = encoder.predict(instance.reshape(1,X_train_minority.shape[1]))[2]
            decoded_instance = decoder.predict(encoded_instance)
            generated_data.append(decoded_instance)
        generated_data = np.array(generated_data)
        generated_data = generated_data.reshape(len(X_train_minority), X_train_minority.shape[1])
        synthetic_data.append(generated_data)

    #compute metrics to evaluate the reconstruction
    for instance_set in synthetic_data:
        mse = mean_squared_error(X_train_minority, instance_set)
        print("MSE:", mse)

        r2 = r2_score(X_train_minority, instance_set)
        print("R-squared:", r2)

        mape = mean_absolute_percentage_error(X_train_minority, instance_set)
        print('MAPE',mape)

        return synthetic_data
    

def concat_train_data(synthetic_data, X_train, y_train, X_test):

    X_train_ff_ae = list(X_train) 
    print(len(X_train_ff_ae))
    for instance_set in synthetic_data:
        X_train_ff_ae.extend(instance_set)
    y_train_ff_ae = list(y_train)  
    y_train_ff_ae.extend([1] * (1 * len(synthetic_data[0])))
    print(len(X_train_ff_ae), len(y_train_ff_ae))
    combined = list(zip(X_train_ff_ae, y_train_ff_ae))
    random.shuffle(combined)
    X_train_ff_ae, y_train_ff_ae = zip(*combined)
    X_train_ff_ae, y_train_ff_ae =np.array(X_train_ff_ae), np.array(y_train_ff_ae)

    #scale the data
    scaler = MinMaxScaler()
    X_train_ff_ae = scaler.fit_transform(X_train_ff_ae)
    X_test = scaler.transform(X_test)

    return X_train_ff_ae, X_test, y_train_ff_ae


def classific_models(model_name):
    '''returns the model and its parameters for gridsearch'''
    
    if model_name =='SVM':
        Cs = [ 0.1, 1, 10]
        kernels=['rbf', 'linear']        
        model = SVC(gamma='scale', random_state=0,class_weight="balanced",
                     )
        parameters = {'kernel': kernels}
    if model_name =='GaussianNB':
        model =  GaussianNB()     
        var_smooth = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        parameters = {'var_smoothing': var_smooth}
    if model_name == 'LogR':
        model = LogisticRegression(random_state=0,  class_weight="balanced",
                                    max_iter=10000) 
        Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        parameters = {'C': Cs}

    return model, parameters


def models_oversampling(oversample_type, model_name):
    
    if model_name =='SVM':
        Cs = [0.01, 0.1, 1, 10, 100]
        kernels=['rbf', 'linear']
        model = SVC(gamma='scale', random_state=0, class_weight="balanced", max_iter=5000)
        if oversample_type == 'smote':
            pipeline = Pipeline([('sampling', SMOTE(random_state=3, k_neighbors=1)),
                    ('classification', model)])
        if oversample_type == 'borderlinesmote':
            pipeline = Pipeline([('sampling', BorderlineSMOTE(random_state=3, k_neighbors=1)),
                    ('classification', model)])
        if oversample_type == 'oversampler':
            pipeline = Pipeline([('sampling', RandomOverSampler(random_state=3)),
                    ('classification', model)])
        #parameters = {'classification__C': Cs, 'classification__kernel': kernels}
        parameters = {'classification__kernel': kernels}
  
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
        model = LogisticRegression(random_state=0,  class_weight="balanced", max_iter=5000)
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



def train_classifier_vae(name, X_train, y_train, X_test, y_test):
    '''trains LogR/NB/SVM on training set & saves the models to pickles.
    '''

    models = ['SVM','GaussianNB', 'LogR']
    #models = ['SVM']
    scorer = make_scorer(f1_score, average='macro')

    for classif in models:
        types_of_oversampling = ['smote', 'borderlinesmote', 'oversampler']
        types_of_oversampling = ['smote']
        print('classifier', classif)
        for oversample_type in types_of_oversampling:
            print('type of oversampling:', oversample_type)   
            model, params = models_oversampling(oversample_type, classif)
        #model, params = classific_models(classif)
        '''use RepeatedStratifiedKFold s.t. each fold of the cross-validation split  
        has the same class distribution as the original dataset'''
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
        clf = GridSearchCV(model, params, cv= cv, verbose=1, 
                           scoring=scorer
                           )
        best_model = clf.fit(X_train, y_train)
        print(clf.best_estimator_)
        y_pred = best_model.predict(X_test)  
        print(classification_report_imbalanced(y_test, y_pred))
        #pickle.dump(best_model, open('pickles/model_vae_'+str(name)+'_'+str(classif)+'.p', 'wb'))
      

def evaluate_vae(name, X_test, y_test):
    '''make predictions for test set for all models & print classification report'''   

    classifiers = ['SVM','GaussianNB', 'LogR']
    names = ['SVM', 'Gaussian Naive Bayes', 'Logistic Regression']
    '''classifiers = ['SVM']
    names = ['SVM']'''

    for clf_name, clf in zip(names, classifiers):
     
      print('classifier:', clf_name)  
      model = pickle.load(open('pickles/model_vae_'+str(name)+'_'+str(clf)+'.p', 'rb'))
      y_pred = model.predict(X_test)  
      print('classification report')
      print(classification_report_imbalanced(y_test, y_pred))

 


#get the datasets
names = ['arrhythmia', 'webpage', 'mammography']
names = ['arrhythmia']
for name in names:
    print('dataset', name)
    X, y = get_data(name)
    X_train, y_train, X_test, y_test = split_train_test_dataset(name, X, y, pct=0.8)
    #add the generated data to the initial training set
    synthetic_data = generate_synthetic_vae(name,X, y, X_train, y_train)
    #concatenate all training data
    X_train_vae, X_test, y_train_vae = concat_train_data(synthetic_data, X_train, y_train, X_test)
    #train the classifiers
    #train_classifier_vae(name,X_train_vae, y_train_vae, X_test, y_test)
    #evaluate the models 
    #evaluate_vae(name, X_test, y_test)


