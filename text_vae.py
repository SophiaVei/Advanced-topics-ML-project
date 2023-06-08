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

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from imblearn.metrics import classification_report_imbalanced



def get_ae_dataset():
    '''returns tensors of train& dev set of the data 
    that will be used for training the autoencoder'''

    X = pickle.load(open('pickles/tweets_embeddings.p', 'rb')) 
    X_train = X[:int(0.7*len(X))]
    X_val = X[int(0.7*len(X)):int(0.8*len(X))]

    return X_train, X_val



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
    original_dim = 300

    # Variational autoencoder model
    inputs = Input(shape=(original_dim,))
    x_mean = Dense(latent_dim, activation='tanh')(inputs)
    x_log_var = Dense(latent_dim, activation='tanh')(inputs) 

    x = Lambda(sampling, output_shape=(latent_dim,))([x_mean, x_log_var])

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

    X_train, X_val = get_ae_dataset()

    filepath = 'models/best_vae.h5'

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, 
                                    save_best_only=True)    
    early_stopping_monitor = EarlyStopping(patience=5)

    history = vae.fit(X_train, None,\
                batch_size=16,epochs=300,shuffle=True,verbose=2,\
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

    return model, parameters


def train_eval_classifier_vae(X_train, y_train):
    '''trains LogR/NB/SVM on training set & evaluate them.
    '''
    
    models = ['GaussianNB', 'LogR']
            
    scorer = make_scorer(f1_score, average='macro')

    for classif in models:
        print(classif)
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
     
      
 
latent_dim = 256
#train Variational AE  
encoder, decoder = train_vae(latent_dim)

filepath = 'models/best_vae.h5'               
model = load_model(filepath)

#generate data for the minority class
pos_tweets_train = pickle.load(open('pickles/pos_tweets_train.p', 'rb'))
synthetic_data = []
for i in range (1):    
    generated_data = []
    for instance in pos_tweets_train:
        encoded_instance = encoder.predict(instance.reshape(1,300))[2]
        decoded_instance = decoder.predict(encoded_instance)
        generated_data.append(decoded_instance)
    generated_data = np.array(generated_data)
    generated_data = generated_data.reshape(len(pos_tweets_train), 300)
    synthetic_data.append(generated_data)

#compute metrics to evaluate the reconstruction
for instance_set in synthetic_data:
    mse = mean_squared_error(pos_tweets_train, instance_set)
    print("MSE:", mse)

    r2 = r2_score(pos_tweets_train, instance_set)
    print("R-squared:", r2)

#add the generated data to the initial training set
X = pickle.load(open('pickles/tweets_embeddings.p', 'rb'))
y = pickle.load(open('pickles/tweets_labels.p', 'rb'))
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

#train the classifiers anf evaluate
train_eval_classifier_vae(X_train_ff_ae, y_train_ff_ae)




