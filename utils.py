import os

from cv2 import cv2
import numpy as np


def get_image_path():
    data_dir = './data'
    image_path = list()
    for filename in os.listdir(data_dir):
        filename = os.path.join(data_dir, filename)
        image_path.append(filename)
    return image_path


def load_capcha():
    image_path = get_image_path()
    N = len(image_path)

    images = np.zeros(shape=(N, 50, 200))
    for idx, filename in enumerate(image_path):
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        images[idx] = image
    return images


if __name__ == "__main__":
    load_capcha()
