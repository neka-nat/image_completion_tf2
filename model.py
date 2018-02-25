import numpy as np
from keras.layers import Flatten, Activation, Conv2D, Conv3D, Conv3DTranspose, Dense, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
import keras.backend as K

def model_generator(input_shape=(5, 256, 256, 3)):
    """
    Architecture of the image completion network
    """
    rate = 4
    model = Sequential()
    model.add(Conv3D(64 / rate, kernel_size=5, strides=1, padding='same',
                     dilation_rate=(1, 1, 1), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv3D(128 / rate, kernel_size=3, strides=(1, 2, 2),
                     padding='same', dilation_rate=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(128 / rate, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv3D(256 / rate, kernel_size=3, strides=(1, 2, 2),
                     padding='same', dilation_rate=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(256 / rate, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(256 / rate, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv3D(256 / rate, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(256 / rate, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 4, 4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(256 / rate, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 8, 8)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(256 / rate, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 16, 16)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv3D(256 / rate, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(256 / rate, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv3DTranspose(128 / rate, kernel_size=4, strides=(1, 2, 2),
                              padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(128 / rate, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv3DTranspose(64 / rate, kernel_size=4, strides=(1, 2, 2),
                              padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(32 / rate, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv3D(3, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    return model


def model_discriminator(global_shape=(5, 256, 256, 3), local_shape=(5, 128, 128, 3)):
    g_img = Input(shape=global_shape)
    l_img = Input(shape=local_shape)
    rate = 4

    # Local Discriminator
    x_l = Conv3D(64 / rate, kernel_size=5, strides=(1, 2, 2), padding='same')(l_img)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    x_l = Conv3D(128 / rate, kernel_size=5, strides=(1, 2, 2), padding='same')(x_l)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    x_l = Conv3D(256 / rate, kernel_size=5, strides=(1, 2, 2), padding='same')(x_l)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    x_l = Conv3D(512 / rate, kernel_size=5, strides=(1, 2, 2), padding='same')(x_l)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    x_l = Conv3D(512 / rate, kernel_size=5, strides=(1, 2, 2), padding='same')(x_l)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    x_l = Flatten()(x_l)
    x_l = Dense(1024, activation='relu')(x_l)

    # Global Discriminator
    x_g = Conv3D(64 / rate, kernel_size=5, strides=(1, 2, 2), padding='same')(g_img)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Conv3D(128 / rate, kernel_size=5, strides=(1, 2, 2), padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Conv3D(256 / rate, kernel_size=5, strides=(1, 2, 2), padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Conv3D(512 / rate, kernel_size=5, strides=(1, 2, 2), padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Conv3D(512 / rate, kernel_size=5, strides=(1, 2, 2), padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Conv3D(512 / rate, kernel_size=5, strides=(1, 2, 2), padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Flatten()(x_g)
    x_g = Dense(1024, activation='relu')(x_g)

    x = concatenate([x_l, x_g])
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=[g_img, l_img], outputs=x)

if __name__ == "__main__":
    from keras.utils import plot_model
    generator = model_generator()
    generator.summary()
    plot_model(generator, to_file='generator.png', show_shapes=True)
    discriminator = model_discriminator()
    discriminator.summary()
    plot_model(discriminator, to_file='discriminator.png', show_shapes=True)
