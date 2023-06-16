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

from sklearn.metrics import mean_squared_error,r2_score,\
        mean_absolute_percentage_error, f1_score, make_scorer

from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from imblearn.metrics import classification_report_imbalanced, geometric_mean_score


def get_ae_dataset():
    '''returns tensors of train& dev set of the data 
    that will be used for training the autoencoder'''

    X = pickle.load(open('pickles/tweets_embeddings.p', 'rb')) 
    X_train = X[:int(0.7*len(X))]
    X_val = X[int(0.7*len(X)):int(0.8*len(X))]

    noise_pct = 0.05
    X_train_noise = X_train + (noise_pct*np.random.normal(loc=0.0, scale=1.0, size=X_train.shape))
    X_val_noise = X_val + (noise_pct*np.random.normal(loc=0.0, scale=1.0, size=X_val.shape))


    return X_train, X_train_noise, X_val, X_val_noise



def split_train_test_dataset(X, y, pct=0.8):
    '''splits the dataset in train/ test set for the 
    classification task'''

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
    x_mean, x_log_var = args
    epsilon = K.random_normal(shape=(K.shape(x_mean)[0], latent_dim), mean=0.,
                              stddev=1.0)
    return x_mean + K.exp(0.5 * x_log_var) * epsilon # (e^a)^b=e^ab


def train_vae(latent_dim):
    '''trains a variational autoencoder'''

    K.clear_session()

    X_train, X_train_noise, X_val, X_val_noise = get_ae_dataset()
    original_dim = X_train.shape[1]

    # Variational autoencoder model
    inputs = Input(shape=(original_dim,))
    x_mean = Dense(latent_dim, activation='tanh')(inputs)
    x_log_var = Dense(latent_dim, activation='tanh')(inputs) 

    x = Lambda(sampling, output_shape=(latent_dim,))([x_mean, x_log_var])

    decoded = Dense(original_dim)(x)
    vae = Model(inputs, decoded, name='vae')
    print (vae.summary())

    # Create the loss function and compile the model
    reconstruction_loss = mse(inputs, decoded)
    kl_loss =  -0.5 * K.sum(1 + x_log_var - K.square(x_mean) -K.exp(x_log_var), axis=-1)
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    adam = optimizers.Adam(lr=0.0001)
    vae.compile(optimizer=adam)

    # encoder model  
    encoder = Model(inputs, [x_mean, x_log_var, x], name='encoder')

    # decoder model 
    encoded_input = Input(shape=(latent_dim,))
    decoder_layer = vae.layers[4](encoded_input)
    decoder = Model(encoded_input, decoder_layer, name='decoder')

    filepath = 'models/best_vae.h5'

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, 
                                    save_best_only=True)    
    early_stopping_monitor = EarlyStopping(patience=5)

    history = vae.fit(X_train, X_train,\
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

    return encoder, decoder


def sentim_models(model_name):
    '''returns the model and its parameters for gridsearch'''
    
    if model_name =='SVM':
        Cs = [ 0.01, 0.1, 1, 10, 100]
        kernels=['rbf', 'linear']        
        model = SVC(gamma='scale', random_state=0, class_weight="balanced",
                    max_iter=10000)
        parameters = {'C': Cs, 'kernel': kernels}
    if model_name =='GaussianNB':  
        model =  GaussianNB()     
        var_smooth = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        parameters = {'var_smoothing': var_smooth}     
    if model_name =='LogR':
        model = LogisticRegression(random_state=0, class_weight="balanced", 
                                    max_iter=5000) 
        Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        parameters = {'C': Cs}
    if model_name =='RF':
        model =  RandomForestClassifier(random_state=0)
        num_estimators = [100, 200, 300]
        parameters = {'n_estimators': num_estimators}

    return model, parameters


def train_eval_classifier_vae(classif, X_train, y_train):
    '''trains LogR/NB/SVM on training set & evaluates them.
    '''
            
    scorer = make_scorer(f1_score, average='macro')
    model, params = sentim_models(classif)
    
    '''use RepeatedStratifiedKFold s.t. each fold of the cross-validation split  
    has the same class distribution as the original dataset'''
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=1)
    clf = GridSearchCV(model, params, cv= cv, verbose=1, 
                        scoring=scorer
                        )
    best_model = clf.fit(X_train, y_train)
    print(clf.best_estimator_)
    y_pred = best_model.predict(X_test)  
    print('classification report')
    print(classification_report_imbalanced(y_test, y_pred))
    gmean = geometric_mean_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')

    if classif == 'GaussianNB':
        text_test_set = pickle.load(open('pickles/sentim_text_test.p', 'rb'))
        text_test_set = np.array(text_test_set)
        indices = np.where(np.logical_and(y_pred != 'pos', y_test == 'pos'))
        print(text_test_set[indices])

    return gmean, f1_macro
    
 
def vae_main():
    '''main function for training VAE for oversampling, train & evaluate the classifiers'''


    X = pickle.load(open('pickles/tweets_embeddings.p', 'rb'))
    y = pickle.load(open('pickles/tweets_labels.p', 'rb'))
    X_train, y_train, X_test, y_test = split_train_test_dataset(X, y, pct=0.8)
    X_pos_tr = X_train[y_train == 'pos']
    print('number of training instances in pos class:', X_pos_tr.shape)

    #instances = [10, 30, 50, 100, 300]
    instances = [50]

    models = ['SVM', 'GaussianNB', 'LogR']

    for m in models:
        print(m)
        gmean_list, f1_macro_list = [], []
        for i in instances:
            #train Variational AE  
            encoder, decoder = train_vae(latent_dim)
            latent_samples = np.random.normal(size=(i, latent_dim))
            synthetic_data = decoder.predict(latent_samples)

            #add the generated data to the initial training set
            X_train_ff_ae = list(X_train)    
            X_train_ff_ae.extend(synthetic_data)
            y_train_ff_ae = list(y_train)  
            y_train_ff_ae.extend(['pos'] * len(synthetic_data))
            print(len(X_train_ff_ae), len(y_train_ff_ae))
            combined = list(zip(X_train_ff_ae, y_train_ff_ae))
            random.shuffle(combined)
            X_train_ff_ae, y_train_ff_ae = zip(*combined)
            X_train_ff_ae, y_train_ff_ae =np.array(X_train_ff_ae), np.array(y_train_ff_ae)

            #scale the data
            scaler = MinMaxScaler()
            X_train_ff_ae = scaler.fit_transform(X_train_ff_ae)
            X_test = scaler.transform(X_test)

            #train the classifiers and evaluate
            gmean, f1_macro = train_eval_classifier_vae(m, X_train_ff_ae, y_train_ff_ae)
            gmean_list.append(gmean)
            f1_macro_list.append(f1_macro)

        print('for model ', m, ',values of gmean are', gmean_list)
        print('for model ', m, ',values of f1 macro are', f1_macro_list)


latent_dim = 64
vae_main()
