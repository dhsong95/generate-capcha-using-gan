from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import BatchNormalization, Dense, Flatten,\
    LeakyReLU, Reshape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

from utils import load_capcha


class GANNetwork:
    def __init__(self):
        self.noise_size = 256
        self.image_height = 50
        self.image_width = 200
        self.image_channel = 1

        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()

        self.optimizer = Adam()

        self.discriminator.compile(
            optimizer=self.optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        self.generator.compile(
            optimizer=self.optimizer,
            loss='binary_crossentropy'
        )

        self.discriminator.trainable = False
        noise = keras.Input(self.noise_size)
        generated_image = self.generator(noise)
        validity = self.discriminator(generated_image)
        self.combined = Model(noise, validity)

        self.combined.summary()

        self.combined.compile(
            optimizer=self.optimizer,
            loss='binary_crossentropy'
        )

    def build_discriminator(self):
        model = Sequential()
        model.add(Flatten(input_shape=(
            self.image_height, self.image_width, self.image_channel
        )))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))

        image = keras.Input(shape=(
            self.image_height, self.image_width, self.image_channel
        ))
        validity = model(image)

        model.summary()

        return Model(image, validity)

    def build_generator(self):
        model = Sequential()
        model.add(Dense(512, input_shape=(self.noise_size,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(
            self.image_height * self.image_width * self.image_channel,
            activation='tanh')
        )
        model.add(Reshape(
            (self.image_height, self.image_width, self.image_channel)
        ))

        noise = keras.Input(shape=(self.noise_size))
        image = model(noise)

        model.summary()

        return Model(noise, image)

    def train(self, epochs, batch_size, save_interval):
        dataset = load_capcha()
        dataset = np.expand_dims(dataset, -1)
        dataset = (dataset.astype(np.float) - 127.5) / 127.5

        half_batch_size = batch_size // 2

        for epoch in range(epochs):
            indices = np.random.randint(0, dataset.shape[0], half_batch_size)
            images = dataset[indices]

            noise = np.random.normal(0, 1, (half_batch_size, self.noise_size))
            generated_images = self.generator.predict(noise)

            d_real_loss = self.discriminator.train_on_batch(
                images, np.ones((half_batch_size, 1))
            )
            d_fake_loss = self.discriminator.train_on_batch(
                generated_images, np.zeros((half_batch_size, 1))
            )

            d_loss = 0.5 * np.add(d_real_loss, d_fake_loss)

            noise = np.random.normal(0, 1, (batch_size, self.noise_size))
            g_loss = self.combined.train_on_batch(
                noise, np.ones((batch_size, 1))
            )

            print(
                f'{epoch:8}\t',
                f'[D loss: {d_loss[0]:.4f}, acc: {d_loss[1]:.2%}]\t',
                f'[G loss: {g_loss:.4f}]'
            )

            if (epoch % save_interval == 0) or (epoch == epochs - 1):
                self.save_images(epoch)

    def save_images(self, epoch):
        r, c = 5, 5

        noise = np.random.normal(0, 1, size=(r*c, self.noise_size))
        generated_images = self.generator.predict(noise)

        fig, axes = plt.subplots(nrows=r, ncols=c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axes[i, j].imshow(generated_images[cnt, :, :, 0], cmap='gray')
                axes[i, j].axis('off')
                cnt += 1
        fig.savefig('images/capcha_{}.png'.format(epoch))
        plt.close()
