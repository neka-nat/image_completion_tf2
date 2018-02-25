import numpy as np
import cv2
import os
from keras.optimizers import Adadelta
from keras.layers import merge, Input, Lambda
from keras.models import Model
from keras.engine.topology import Container
import keras.backend as K
import matplotlib.pyplot as plt
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

    def flow_from_directory(self, root_dir, batch_size):
        vd_file_list = []
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                full_path = os.path.join(root, f)
                if not full_path.endswith(('.mp4', '.mpg', '.mpeg')):
                    continue
                vd_file_list.append(full_path)

        np.random.shuffle(vd_file_list)
        for f in vd_file_list:
            imgs = []
            vidcap = cv2.VideoCapture(f)
            while True:
                success, image = vidcap.read()
                if not success:
                    break
                cnv_img = cv2.resize(image, self.image_size[1:])[:, :, ::-1]
                imgs.append(cnv_img)

            for _ in range(len(imgs) / self.image_size[0]):
                idxs = np.random.randint(0, len(imgs) - self.image_size[0], (batch_size,))
                for i in idxs:
                    self.images.append(imgs[i:i+self.image_size[0]])

                    pt1 = (0.5 * (np.array(self.image_size[1:]) - np.array(self.local_size[1:]))).astype(np.int32)
                    x2, y2 = pt1 + np.array(self.local_size[1:], dtype=np.int32)
                    self.points.append([pt1[0], pt1[1], x2, y2])
                    m = np.ones((self.image_size[0], self.image_size[1], self.image_size[2], 1), dtype=np.uint8)
                    m[:, pt1[1]:y2, pt1[0]:x2, :] = 0
                    self.masks.append(m)
                inputs = np.asarray(self.images, dtype=np.float32) / 255
                points = np.asarray(self.points, dtype=np.int32)
                masks = np.asarray(self.masks, dtype=np.float32)
                self.reset()
                yield inputs, points, masks

def example_gan(result_dir="output", data_dir="data"):
    input_shape = (5, 128, 128, 3)
    local_shape = (5, 64, 64, 3)
    batch_size = 4
    n_epoch = 100
    tc = int(n_epoch * 0.18)
    td = int(n_epoch * 0.02)
    alpha = 0.0004

    train_datagen = DataGenerator(input_shape[:3], local_shape[:3])

    generator = model_generator(input_shape)
    discriminator = model_discriminator(input_shape, local_shape)
    optimizer = Adadelta()

    # build model
    org_img = Input(shape=input_shape)
    mask = Input(shape=(input_shape[0], input_shape[1], input_shape[2], 1))

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

    local_img = Input(shape=local_shape)
    d_container = Container([org_img, local_img], discriminator([org_img, local_img]))
    d_model = Model([org_img, local_img], d_container([org_img, local_img]))
    d_model.compile(loss='binary_crossentropy', 
                    optimizer=optimizer)

    def random_cropping(x, x1, y1, x2, y2):
        out = []
        for idx in range(batch_size):
            out.append(x[idx, y1[idx]:y2[idx], x1[idx]:x2[idx], :])
        return K.stack(out, axis=0)
    cropping = Lambda(random_cropping, output_shape=local_shape)

    for n in range(n_epoch):
        for inputs, points, masks in train_datagen.flow_from_directory(data_dir, batch_size):
            cmp_image = cmp_model.predict([inputs, masks])
            local = []
            for i in range(batch_size):
                x1, y1, x2, y2 = points[i]
                local.append(inputs[i][:, y1:y2, x1:x2, :])

            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            g_loss = 0.0
            d_loss = 0.0
            if n < tc:
                g_loss = cmp_model.train_on_batch([inputs, masks], inputs)
            else:
                d_loss_real = d_model.train_on_batch([inputs, np.array(local)], valid)
                d_loss_fake = d_model.train_on_batch([cmp_image, np.array(local_cmp)], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                if n >= tc + td:
                    d_container.trainable = False
                    cropping.arguments = {'x1': points[:, 0], 'y1': points[:, 1],
                                          'x2': points[:, 2], 'y2': points[:, 3]}
                    all_model = Model([org_img, mask],
                                      [cmp_out, d_container([cmp_out, cropping(cmp_out)])])
                    all_model.compile(loss=['mse', 'binary_crossentropy'],
                                      loss_weights=[1.0, alpha], optimizer=optimizer)
                    g_loss = all_model.train_on_batch([inputs, masks],
                                                      [inputs, valid])

        print("%d [D loss: %e] [G mse: %e]" % (n, d_loss, g_loss))
        idx = np.random.randint(batch_size)
        fig, axs = plt.subplots(num_img, 3)
        for i in range(input_shape[0]):
            axs[i, 0].imshow(inputs[idx, i, ...] * (1 - masks[idx, i, ...]))
            axs[i, 0].axis('off')
            axs[i, 0].set_title('Input')
            axs[i, 1].imshow(cmp_image[idx, i, ...])
            axs[i, 1].axis('off')
            axs[i, 1].set_title('Output')
            axs[i, 2].imshow(inputs[idx, i, ...])
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
