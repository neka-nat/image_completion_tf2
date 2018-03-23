import numpy as np
from keras.layers import Reshape, Lambda, Flatten, Activation, Conv2D, Conv2DTranspose, Dense, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
import keras.backend as K
import tensorflow as tf

def model_generator(input_shape=(256, 256, 3)):
    """
    Architecture of the image completion network
    """
    model = Sequential()
    model.add(Conv2D(64, kernel_size=5, strides=1, padding='same',
                     dilation_rate=(1, 1), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(128, kernel_size=3, strides=2,
                     padding='same', dilation_rate=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(256, kernel_size=3, strides=2,
                     padding='same', dilation_rate=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(4, 4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(8, 8)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(16, 16)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(128, kernel_size=4, strides=2,
                              padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(64, kernel_size=4, strides=2,
                              padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(3, kernel_size=3, strides=1,
                     padding='same', dilation_rate=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    return model


def model_discriminator(global_shape=(256, 256, 3), local_shape=(128, 128, 3)):
    def crop_image(img, crop):
        return tf.image.crop_to_bounding_box(img,
                                             crop[1],
                                             crop[0],
                                             crop[3] - crop[1],
                                             crop[2] - crop[0])

    in_pts = Input(shape=(4,), dtype='int32')
    cropping = Lambda(lambda x: K.map_fn(lambda y: crop_image(y[0], y[1]), elems=x, dtype=tf.float32),
                      output_shape=local_shape)
    g_img = Input(shape=global_shape)
    l_img = cropping([g_img, in_pts])

    # Local Discriminator
    x_l = Conv2D(64, kernel_size=5, strides=2, padding='same')(l_img)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    x_l = Conv2D(128, kernel_size=5, strides=2, padding='same')(x_l)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    x_l = Conv2D(256, kernel_size=5, strides=2, padding='same')(x_l)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    x_l = Conv2D(512, kernel_size=5, strides=2, padding='same')(x_l)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    x_l = Conv2D(512, kernel_size=5, strides=2, padding='same')(x_l)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)
    x_l = Flatten()(x_l)
    x_l = Dense(1024, activation='relu')(x_l)

    # Global Discriminator
    x_g = Conv2D(64, kernel_size=5, strides=2, padding='same')(g_img)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Conv2D(128, kernel_size=5, strides=2, padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Conv2D(256, kernel_size=5, strides=2, padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Conv2D(512, kernel_size=5, strides=2, padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Conv2D(512, kernel_size=5, strides=2, padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Conv2D(512, kernel_size=5, strides=2, padding='same')(x_g)
    x_g = BatchNormalization()(x_g)
    x_g = Activation('relu')(x_g)
    x_g = Flatten()(x_g)
    x_g = Dense(1024, activation='relu')(x_g)

    x = concatenate([x_l, x_g])
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=[g_img, in_pts], outputs=x)

if __name__ == "__main__":
    from keras.utils import plot_model
    generator = model_generator()
    generator.summary()
    plot_model(generator, to_file='generator.png', show_shapes=True)
    discriminator = model_discriminator()
    discriminator.summary()
    plot_model(discriminator, to_file='discriminator.png', show_shapes=True)
