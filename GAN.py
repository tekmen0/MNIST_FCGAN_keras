#based on: https://medium.com/analytics-vidhya/implementing-a-gan-in-keras-d6c36bc6ab5f

from keras.optimizers import Adam
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import os

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU

from keras.layers import Input
from keras.models import Model

np.random.seed(10)

noise_dim = 100

batch_size = 16
steps_per_epoch = 3750
epochs = 10

save_path = 'fcgan-images'

img_rows, img_cols, channels = 28, 28, 1

optimizer = Adam(0.0002, 0.5)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
print(type(x_train))
x_train = x_train.reshape(-1, img_rows*img_cols*channels)
print(type(x_train))

if not os.path.isdir(save_path):
    os.mkdir(save_path)

#Define models
def create_generator():
    generator = Sequential()

    generator.add(Dense(256, input_dim=noise_dim))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(img_rows * img_cols * channels, activation='tanh'))

    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return generator

def create_discriminator():
    discriminator = Sequential()

    discriminator.add(Dense(1024, input_dim=img_rows * img_cols * channels))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(1, activation='sigmoid'))

    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return discriminator

discriminator = create_discriminator()
generator = create_generator()

discriminator.trainable = False

gan_input = Input(shape=(noise_dim,))  #most probably after comma, it assumes dimension is 1
fake_image = generator(gan_input)

gan_output = discriminator(fake_image)

gan = Model(gan_input, gan_output)
gan.compile(loss="binary_crossentropy", optimizer = optimizer)

#train
for epoch in range(epochs):
    for batch in range(steps_per_epoch):
        #each loop processes 32 samples (2xbatch_size) at single propagation
        noise = np.random.normal(0, 1, size=(batch_size, noise_dim))
        #note that first, predicting with initial values
        fake_x = generator.predict(noise)
        real_x = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

        x = np.concatenate((real_x, fake_x))

        disc_y = np.zeros(2*batch_size)
        disc_y[:batch_size] = 0.9
        #Single gradient update over one batch of samples with 32 train set.
        d_loss = discriminator.train_on_batch(x, disc_y)

        #how loss function works for generative models?
        #because it gan produces values between 0 and 1,
        #its loss func works the same as normal classifier
        y_gen = np.ones(batch_size)
        g_loss = gan.train_on_batch(noise, y_gen)

    #weights_d = [list(w.reshape((-1))) for w in discriminator.get_weights()]
    print('\n'+'#'*60)
    print(f'D WEIGHTS: {discriminator.get_weights().shape()}')
    print(f'GAN WEIGHTS: {gan.get_weights().shape()}')
    print()
    print(f'Epoch: {epoch} \t Discriminator Loss: {d_loss} \t\t Generator Loss: {g_loss}')


def show_images(noise):
    generated_images = generator.predict(noise)
    plt.figure(figsize=(10,10))

    for i, image in enumerate(generated_images):
        plt.subplot(10, 10, i+1)
        plt.imshow(image.reshape((img_rows, img_cols, channels)), cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

noise = np.random.normal(0, 1, size=(100, noise_dim))
show_images(noise)




