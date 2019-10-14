import numpy as np
import cv2
import os
import tensorflow.keras.models
import matplotlib.pyplot as plt
from model import model_generator
from train import DataGenerator


def test_gan(model_path="output/generator.h5", result_dir="output", data_dir="test"):
    input_shape = (256, 256, 3)
    local_shape = (128, 128, 3)
    batch_size = 4

    test_datagen = DataGenerator(data_dir, input_shape[:2], local_shape[:2])
    cmp_model = tensorflow.keras.models.load_model(model_path, compile=False)
    cmp_model.summary()

    cnt = 0
    for inputs, points, masks in test_datagen.flow(batch_size):
        cmp_image = cmp_model.predict([inputs, masks])
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
        fig.savefig(os.path.join(result_dir, "result_test_%d.png" % cnt))
        plt.close()
        cnt += 1


def main():
    test_gan()


if __name__ == "__main__":
    main()
