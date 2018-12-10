from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Lambda ,Reshape
from keras.models import Model
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras.losses import mse, binary_crossentropy
import numpy as np
import os
from keras.utils import np_utils
import matplotlib.pyplot as plt

from keras import backend as K

def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")

    else:
        print("---  There is this folder!  ---")

#建立儲存結果資料夾
now_path = os.path.abspath('.')
mkdir(now_path+"\\log")
log_path = now_path + "\\log"

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

yTr_embedding = np_utils.to_categorical(y_train)
yTe_embedding = np_utils.to_categorical(y_test)


def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(latent_dim,))
    return z_mean + K.exp(z_log_sigma) * epsilon


def vae_loss(input_img, output_img):
    xent_loss = binary_crossentropy(input_img, output_img)
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    return xent_loss + kl_loss


latent_dim = 32
batch_size = 50

inputs= Input(shape=(28, 28, 1), name='encoder_input')  # adapt this if using `channels_first` image data format
x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
h = Dense(64, activation='relu')(x)


# Variational Part
z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_sigma])

encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
encoder.summary()
# at this point the representation is (4, 4, 8) i.e. 128-dimensional
latent_inputs = Input(shape=(latent_dim,))
x = Dense(64, activation='relu')(latent_inputs)
x = Dense(128, activation='relu')(x)
x = Reshape((4, 4, 8))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='VAE')
vae.summary()


vae.compile(optimizer='adadelta', loss=vae_loss)

print("------Start Training------")
vae.fit(x_train, x_train, epochs=10, batch_size=batch_size, shuffle=True, validation_data=(x_test, x_test))

vae.save(log_path + "/autoencoder.h5")

#decoded_imgs = autoencoder.predict(x_test)


n = 10
plt.figure(figsize=(20, 4))
for i in range(1,n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()