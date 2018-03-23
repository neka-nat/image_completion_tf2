import imghdr
import numpy as np
import cv2
import os
from keras.optimizers import Adadelta
from keras.layers import merge, Input
from keras.models import Model
from keras.engine.topology import Container
from keras.utils import generic_utils
import keras.backend as K
import matplotlib.pyplot as plt
from model import model_generator, model_discriminator

class DataGenerator(object):
    def __init__(self, root_dir, image_size, local_size):
        self.image_size = image_size
        self.local_size = local_size
        self.reset()
        self.img_file_list = []
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                full_path = os.path.join(root, f)
                if imghdr.what(full_path) is None:
                    continue
                self.img_file_list.append(full_path)

    def __len__(self):
        return len(self.img_file_list)

    def reset(self):
        self.images = []
        self.points = []
        self.masks = []

    def flow(self, batch_size, hole_min=64, hole_max=128):
        np.random.shuffle(self.img_file_list)
        for f in self.img_file_list:
            img = cv2.imread(f)
            img = cv2.resize(img, self.image_size)[:, :, ::-1]
            self.images.append(img)

            x1 = np.random.randint(0, self.image_size[0] - self.local_size[0] + 1)
            y1 = np.random.randint(0, self.image_size[1] - self.local_size[1] + 1)
            x2, y2 = np.array([x1, y1]) + np.array(self.local_size)
            self.points.append([x1, y1, x2, y2])

            w, h = np.random.randint(hole_min, hole_max, 2)
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

def example_gan(result_dir="output", data_dir="data"):
    input_shape = (256, 256, 3)
    local_shape = (128, 128, 3)
    batch_size = 4
    n_epoch = 100
    tc = int(n_epoch * 0.18)
    td = int(n_epoch * 0.02)
    alpha = 0.0004

    train_datagen = DataGenerator(data_dir, input_shape[:2], local_shape[:2])

    generator = model_generator(input_shape)
    discriminator = model_discriminator(input_shape, local_shape)
    optimizer = Adadelta()

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
    cmp_container = Container([org_img, mask], completion)
    cmp_out = cmp_container([org_img, mask])
    cmp_model = Model([org_img, mask], cmp_out)
    cmp_model.compile(loss='mse',
                      optimizer=optimizer)
    cmp_model.summary()

    in_pts = Input(shape=(4,), dtype='int32')
    d_container = Container([org_img, in_pts], discriminator([org_img, in_pts]))
    d_model = Model([org_img, in_pts], d_container([org_img, in_pts]))
    d_model.compile(loss='binary_crossentropy', 
                    optimizer=optimizer)
    d_model.summary()

    d_container.trainable = False
    all_model = Model([org_img, mask, in_pts],
                      [cmp_out, d_container([cmp_out, in_pts])])
    all_model.compile(loss=['mse', 'binary_crossentropy'],
                      loss_weights=[1.0, alpha], optimizer=optimizer)
    all_model.summary()

    for n in range(n_epoch):
        progbar = generic_utils.Progbar(len(train_datagen))
        for inputs, points, masks in train_datagen.flow(batch_size):
            cmp_image = cmp_model.predict([inputs, masks])
            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            g_loss = 0.0
            d_loss = 0.0
            if n < tc:
                g_loss = cmp_model.train_on_batch([inputs, masks], inputs)
            else:
                d_loss_real = d_model.train_on_batch([inputs, points], valid)
                d_loss_fake = d_model.train_on_batch([cmp_image, points], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                if n >= tc + td:
                    g_loss = all_model.train_on_batch([inputs, masks, points],
                                                      [inputs, valid])
                    g_loss = g_loss[0] + alpha * g_loss[1]
            progbar.add(inputs.shape[0], values=[("D loss", d_loss), ("G mse", g_loss)])

        num_img = min(5, batch_size)
        fig, axs = plt.subplots(num_img, 3)
        for i in range(num_img):
            axs[i, 0].imshow(inputs[i] * (1 - masks[i]))
            axs[i, 0].axis('off')
            axs[i, 0].set_title('Input')
            axs[i, 1].imshow(cmp_image[i])
            axs[i, 1].axis('off')
            axs[i, 1].set_title('Output')
            axs[i, 2].imshow(inputs[i])
            axs[i, 2].axis('off')
            axs[i, 2].set_title('Ground Truth')
        fig.savefig(os.path.join(result_dir, "result_%d.png" % n))
        plt.close()
    # save model
    generator.save(os.path.join(result_dir, "generator.h5"))
    discriminator.save(os.path.join(result_dir, "discriminator.h5"))


def main():
    example_gan()


if __name__ == "__main__":
    main()
