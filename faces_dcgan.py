from __future__ import print_function, division

import os
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import datetime
from keras.layers import Input, Dense, Reshape, Flatten, Conv2DTranspose, BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from PIL import Image
from glob import glob


class DCGAN():
    def __init__(self):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        optimizer = Adam(lr=0.00002, beta_1=0.5)
        optimizer_gen = Adam(lr=0.0002, beta_1=0.5)
        self.discriminator = self.build_discriminator()
        if START_EPOCH is not None:
            self.discriminator.load_weights(LOAD_WEIGHTS_PATH + 'faces_d_' + str(START_EPOCH) + '.h5')
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.generator = self.build_generator()
        if START_EPOCH is not None:
            self.generator.load_weights(LOAD_WEIGHTS_PATH + 'faces_g_' + str(START_EPOCH) + '.h5')
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        self.discriminator.trainable = False
        valid = self.discriminator(img)
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer_gen)

    def build_generator(self):
        model = Sequential()
        model.add(Dense(8 * 8 * 1024, activation="linear", input_dim=self.latent_dim))
        model.add(Reshape((8, 8, 1024)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(filters=512, kernel_size=[5, 5], strides=[2, 2], kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), padding="same"))
        model.add(BatchNormalization(epsilon=0.00005, trainable=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(filters=256, kernel_size=[5, 5], strides=[2, 2], kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), padding="same"))
        model.add(BatchNormalization(epsilon=0.00005, trainable=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(filters=128, kernel_size=[5, 5], strides=[2, 2], kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), padding="same"))
        model.add(BatchNormalization(epsilon=0.00005, trainable=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(filters=64, kernel_size=[5, 5], strides=[2, 2], kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), padding="same"))
        model.add(BatchNormalization(epsilon=0.00005, trainable=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(filters=self.channels, kernel_size=[5, 5], strides=[1, 1], kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), padding="same"))
        model.add(Activation("tanh"))
        print("Generator:")
        model.summary()
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=[5, 5], strides=[2, 2], kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), input_shape=self.img_shape, padding="same"))
        model.add(BatchNormalization(epsilon=0.00005, trainable=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(filters=128, kernel_size=[5, 5], strides=[2, 2], kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), padding="same"))
        model.add(BatchNormalization(epsilon=0.00005, trainable=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(filters=256, kernel_size=[5, 5], strides=[2, 2], kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), padding="same"))
        model.add(BatchNormalization(epsilon=0.00005, trainable=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(filters=512, kernel_size=[5, 5], strides=[1, 1], kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), padding="same"))
        model.add(BatchNormalization(epsilon=0.00005, trainable=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(filters=1024, kernel_size=[5, 5], strides=[2, 2], kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), padding="same"))
        model.add(BatchNormalization(epsilon=0.00005, trainable=True))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dense(1, activation='linear'))
        model.add(Activation("sigmoid"))
        print("Discriminator:")
        model.summary()
        img = Input(shape=self.img_shape)
        validity = model(img)
        return Model(img, validity)

    def get_batches(self, data, batch_size):
        batches = []
        for i in range(int(data.shape[0] // batch_size)):
            batch = data[i * batch_size:(i + 1) * batch_size]
            augmented_images = []
            for img in batch:
                image = Image.fromarray(img)
                if random.choice([True, False]):
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                augmented_images.append(np.asarray(image))
            batch = np.asarray(augmented_images)
            normalized_batch = (batch / 127.5) - 1.0
            batches.append(normalized_batch)
        return batches

    def train(self, epochs, batch_size=64):
        x_train = np.asarray([np.asarray(Image.open(file).resize((self.img_rows, self.img_cols))) for file in glob(INPUT_DATA_DIR + '*')])
        valid = np.ones((batch_size, 1)) * random.uniform(0.9, 1.0)
        fake = np.zeros((batch_size, 1))
        epoch_n = 0
        d_losses = []
        g_losses = []
        for epoch in range(epochs):
            epoch_n += 1
            start_time = time.time()
            mini_epoch_n = 0
            for imgs in self.get_batches(x_train, batch_size):
                mini_epoch_n += 1
                noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
                gen_imgs = self.generator.predict(noise)
                self.discriminator.trainable = True
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                self.discriminator.trainable = False
                g_loss = self.combined.train_on_batch(noise, valid)
                d_losses.append(d_loss)
                g_losses.append(g_loss)
                print("Batch " + str(mini_epoch_n) + " in epoch " + str(epoch_n) + " with D: " + str(d_loss) + " G: " + str(g_loss) + " finished in " + str(time.time() - start_time))

            print("Epoch " + str(epoch_n) + " finished in " + str(time.time() - start_time))
            self.generator.save_weights(SAVE_PATH + 'faces_g_' + str(epoch_n) + '.h5')
            self.discriminator.save_weights(SAVE_PATH + 'faces_d_' + str(epoch_n) + '.h5')
            self.save_imgs(epoch_n)
            plt.plot(d_losses, label='Discriminator', alpha=0.6)
            plt.plot(g_losses, label='Generator', alpha=0.6)
            plt.title("Losses")
            plt.legend()
            plt.savefig(OUTPUT_DIR + "losses_" + str(epoch_n) + ".png")
            plt.close()

    def show_samples(self, sample_images, name, epoch):
        figure, axes = plt.subplots(1, len(sample_images), figsize=(128, 128))
        for index, axis in enumerate(axes):
            axis.axis('off')
            image_array = sample_images[index]
            axis.imshow(image_array)
            image = Image.fromarray(image_array)
            image.save(name + "faces_" + str(epoch) + "_" + str(index) + ".png")
        plt.close()

    def save_imgs(self, epoch):
        r = 5
        noise = np.random.uniform(-1, 1, (r, self.latent_dim))
        samples = self.generator.predict(noise)
        sample_images = [((sample + 1.0) * 127.5).astype(np.uint8) for sample in samples]
        self.show_samples(sample_images, OUTPUT_DIR, epoch)


# Path to project folder
BASE_PATH = "E:/PycharmProjects/Faces-DCGAN"
# Path to folder with checkpoints and which epoch to load
START_EPOCH = None
LOAD_WEIGHTS_PATH = BASE_PATH + '/models/2020-01-26_20-56-42/'


DATASET_LIST_PATH = BASE_PATH + "/input/100k.txt"
INPUT_DATA_DIR = BASE_PATH + "/input/100k/100k/"
OUTPUT_DIR = BASE_PATH + '/output/{date:%Y-%m-%d_%H-%M-%S}/'.format(date=datetime.datetime.now())
SAVE_PATH = BASE_PATH + '/models/{date:%Y-%m-%d_%H-%M-%S}/'.format(date=datetime.datetime.now())

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
DATASET = [INPUT_DATA_DIR + str(line).rstrip() for line in open(DATASET_LIST_PATH,"r")]
DATASET_SIZE = len(DATASET)

print ("Input size: " + str(DATASET_SIZE))

dcgan = DCGAN()
dcgan.train(epochs=200, batch_size=64)
