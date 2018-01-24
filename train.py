import matplotlib as mpl

# This line allows mpl to run with no DISPLAY defined
mpl.use('Agg')

import numpy as np
import cv2
import os
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.layers import merge, Input
from keras.models import Model
import keras.backend as K
from model import model_generator, model_discriminator

class DataGenerator(object):
    def __init__(self, image_size, local_size):
        self.image_size = image_size
        self.local_size = local_size
        self.reset()

    def reset(self):
        self.images = []
        self.points = []
        self.masks = []

    def flow_from_directory(self, root_dir, batch_size, hole_min=24, hole_max=48):
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                img = cv2.imread(os.path.join(root, f))
                img = cv2.resize(img, self.image_size)[:, :, ::-1]
                self.images.append(img)

                x1 = np.random.randint(0, self.image_size[0] - self.local_size[0] + 1)
                y1 = np.random.randint(0, self.image_size[1] - self.local_size[1] + 1)
                x2, y2 = np.array([x1, y1]) + np.array(self.local_size)
                self.points.append([x1, y1, x2, y2])

                w, h = np.random.randint(hole_min, hole_max + 1, 2)
                p1 = x1 + np.random.randint(0, self.local_size[0] - w)
                q1 = y1 + np.random.randint(0, self.local_size[1] - h)
                p2 = p1 + w
                q2 = q1 + h

                m = np.zeros((self.image_size[0], self.image_size[1], 1), dtype=np.uint8)
                m[q1:q2 + 1, p1:p2 + 1] = 1
                self.masks.append(m)

                if len(self.images) == batch_size:
                    inputs = np.asarray(self.images, dtype=np.float32) / 255
                    points = np.asarray(self.points, dtype=np.int32)
                    masks = np.asarray(self.masks, dtype=np.float32)
                    self.reset()
                    yield inputs, points, masks

def example_gan(path="output", data_dir="data"):
    input_shape = (256, 256, 3)
    local_shape = (128, 128, 3)
    batch_size = 4
    n_epoch = 100

    train_datagen = DataGenerator(input_shape[:2], local_shape[:2])

    generator = model_generator(input_shape)
    discriminator = model_discriminator(input_shape, local_shape)
    optimizer = Adam(0.0002, 0.5)

    # build model
    org_img = Input(shape=input_shape)
    mask = Input(shape=(input_shape[0], input_shape[1], 1))

    in_img = merge([org_img, mask],
                   mode=lambda x: x[0] * (1 - x[1]),
                   output_shape=input_shape)
    imitation = generator(in_img)
    completion = merge([imitation, org_img, mask],
                       mode=lambda x: x[0] * x[2] + x[1] * (1 - x[2]),
                       output_shape=input_shape)
    cmp_model = Model([org_img, mask], completion)
    cmp_model.compile(loss='mse', 
                      optimizer=optimizer)

    discriminator.compile(loss='binary_crossentropy', 
                          optimizer=optimizer)
    for n in range(n_epoch):
        for inputs, points, masks in train_datagen.flow_from_directory(data_dir, batch_size):
            cmp_image = cmp_model.predict([inputs, masks])
            local = []
            local_cmp = []
            for i in range(batch_size):
                x1, y1, x2, y2 = points[i]
                local.append(inputs[i][y1:y2, x1:x2, :])
                local_cmp.append(cmp_image[i][y1:y2, x1:x2, :])

            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            # Train the discriminator
            d_loss_real = discriminator.train_on_batch([inputs, np.array(local)], valid)
            d_loss_fake = discriminator.train_on_batch([cmp_image, np.array(local_cmp)], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss = cmp_model.train_on_batch([inputs, masks], inputs)
        print("%d [D loss: %f] [G mse: %f]" % (n, d_loss, g_loss))

    # save model
    generator.save(os.path.join(path, "generator.h5"))
    discriminator.save(os.path.join(path, "discriminator.h5"))


def main():
    example_gan()


if __name__ == "__main__":
    main()
