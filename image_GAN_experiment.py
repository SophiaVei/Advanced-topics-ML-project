import warnings
import numpy as np
from keras.callbacks import LearningRateScheduler
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, UpSampling2D, Conv2D, ZeroPadding2D, Conv2DTranspose
from keras.layers import BatchNormalization, Activation, LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")

#basic variables
generateNoOfImages = 1000 #requested amount of generated synthetic images
generateDigit = 7 #assumed minority class (0 to 9) of which synthetic data is requested
epochsCount = 300


def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=100):
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))
    return LearningRateScheduler(schedule)

def load_imbalanced_data():
    (X_train, y_train), (_, _) = mnist.load_data()
    X_train = X_train / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)
    y_train = y_train.reshape(-1, 1)

    #declare the minority class here
    #generate data of a specific class (=generateDigit)
    mask = (y_train == generateDigit)
    X_train_imbalanced = X_train[mask.squeeze()]
    return X_train_imbalanced

def define_generator(latent_dim):
    model = Sequential()
    model.add(Dense(7 * 7 * 256, activation="relu", input_dim=latent_dim))
    model.add(Reshape((7, 7, 256)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2DTranspose(1, kernel_size=3, strides=1, padding='same', activation='tanh'))

    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img)

def define_discriminator(img_shape):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=img_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)

def train_gan(epochs, batch_size, latent_dim, generator, discriminator, combined):
    X_train = load_imbalanced_data()
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    #scheduler = step_decay_schedule(initial_lr=0.0002, decay_factor=0.75, step_size=100)

    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = combined.train_on_batch(noise, valid)

        print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}] [G loss: {g_loss}]")


def plot_generated_images(generator, latent_dim, num_images=generateNoOfImages):
    noise = np.random.normal(0, 1, (num_images, latent_dim))
    gen_imgs = generator.predict(noise)

    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(int(np.sqrt(num_images)), int(np.sqrt(num_images)))
    count = 0
    for i in range(int(np.sqrt(num_images))):
        for j in range(int(np.sqrt(num_images))):
            axs[i, j].imshow(gen_imgs[count, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            count += 1
    plt.show()
    return gen_imgs

latent_dim = 256
img_shape = (28, 28, 1)

optimizer = Adam(0.0002, 0.5)

generator = define_generator(latent_dim)
discriminator = define_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

z = Input(shape=(latent_dim,))
img = generator(z)

discriminator.trainable = False

valid = discriminator(img)

combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

train_gan(epochs=epochsCount, batch_size=32, latent_dim=latent_dim, generator=generator, discriminator=discriminator, combined=combined)
gen_images = plot_generated_images(generator, latent_dim)

gen_images_squeezed = np.squeeze(gen_images, axis=-1)
y_test_new = np.full(generateNoOfImages, generateDigit)

import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#load MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape and normalize the input features
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0


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

#reshape and normalize the input features
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#evaluate the synthetic data
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)

print("Evaluation report on the generated data:")
print(report)