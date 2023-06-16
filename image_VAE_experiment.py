import warnings
import numpy as np
from imblearn import keras
from keras.backend import binary_crossentropy
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Lambda, Conv2DTranspose, Conv2D
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")

#basic user input variables
generateNoOfImages = 1000 #requested amount of generated synthetic images
generateDigit = 4 #assumed minority class (0 to 9) of which synthetic data is requested
epochsCount = 300

latent_dim = 256
img_shape = (28, 28, 1)
batch_size = 32

def load_imbalanced_data():
    (X_train, y_train), (_, _) = mnist.load_data()
    X_train = X_train / 255.0  # Normalize pixel values to [0, 1]
    X_train = np.expand_dims(X_train, axis=3)
    y_train = y_train.reshape(-1, 1)

    # Create an imbalanced dataset by keeping only a few samples of a specific class (namely, 1)
    mask = (y_train == generateDigit)
    X_train_imbalanced = X_train[mask.squeeze()]
    return X_train_imbalanced


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


#alternate architecture used initially:

# def define_vae(latent_dim):
#     #encoder
#     inputs = Input(shape=img_shape)
#     x = Flatten()(inputs)
#     x = Dense(512, activation='relu')(x)
#     z_mean = Dense(latent_dim)(x)
#     z_log_var = Dense(latent_dim)(x)
#     z = Lambda(sampling)([z_mean, z_log_var])
#
#     encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
#
#     #decoder
#     latent_inputs = Input(shape=(latent_dim,))
#     x = Dense(512, activation='relu')(latent_inputs)
#     x = Dense(7 * 7 * 64, activation='relu')(x)
#     x = Reshape((7, 7, 64))(x)
#     x = Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu')(x)
#     decoded_outputs = Conv2DTranspose(1, kernel_size=3, strides=2, padding='same', activation='sigmoid')(x)
#
#     decoder = Model(latent_inputs, decoded_outputs, name='decoder')
#
#     outputs = decoder(encoder(inputs)[2])
#     vae = Model(inputs, outputs, name='vae')
#
#     #loss function
#     reconstruction_loss = K.mean(K.square(inputs - outputs), axis=[1, 2, 3])
#     kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
#     vae_loss = reconstruction_loss + kl_loss
#     vae.add_loss(K.mean(vae_loss))
#
#     return vae
from keras.losses import binary_crossentropy

def define_vae(latent_dim):
    #encoder
    inputs = Input(shape=img_shape)
    x = Flatten()(inputs)
    x = Dense(7 * 7 * 128, activation='relu')(x)
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)
    z = Lambda(sampling)([z_mean, z_log_var])

    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    #decoder
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(7 * 7 * 128, activation='relu')(latent_inputs)
    x = Reshape((7, 7, 128))(x)
    x = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    decoded_outputs = Conv2DTranspose(1, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)

    decoder = Model(latent_inputs, decoded_outputs, name='decoder')

    #VAE
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')

    #loss function
    inputs_flat = Flatten()(inputs)
    outputs_flat = Flatten()(outputs)
    reconstruction_loss = binary_crossentropy(inputs_flat, outputs_flat)
    kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = reconstruction_loss + kl_loss
    vae.add_loss(K.mean(vae_loss))

    return vae


def train_vae(epochs, batch_size, vae):
    X_train = load_imbalanced_data()

    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        vae.train_on_batch(imgs, None)

        print(f"Epoch: {epoch+1}/{epochs} - Loss: {vae.evaluate(imgs, None, verbose=0)}")

def generate_images(vae, latent_dim, num_images=generateNoOfImages):
    noise = np.random.normal(0, 1, (num_images, 28, 28, 1))
    gen_imgs = vae.predict(noise)

    fig, axs = plt.subplots(int(np.sqrt(num_images)), int(np.sqrt(num_images)))
    count = 0
    for i in range(int(np.sqrt(num_images))):
        for j in range(int(np.sqrt(num_images))):
            axs[i, j].imshow(gen_imgs[count, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            count += 1
    plt.show()
    return gen_imgs

#initialize VAE
vae = define_vae(latent_dim)
vae.compile(optimizer='adam')
train_vae(epochsCount, batch_size, vae)

#generate images
gen_images = generate_images(vae, latent_dim)

gen_images_squeezed = np.squeeze(gen_images, axis=-1)
y_test_new = np.full(generateNoOfImages, generateDigit)

import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#load MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#reshape and normalize the input features
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

# Split the dataset into training and testing sets
X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#train a classifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

#predict original test data
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)

print("Initial Evaluation Report:")
print(report)

X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_test = gen_images_squeezed
y_test = y_test_new

#reshape & normalize the input features
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#evaluate the synthetic data
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)

print("Evaluation report on the generated data:")
print(report)
