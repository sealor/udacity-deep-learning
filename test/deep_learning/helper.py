import os
import pickle
import time

import matplotlib.pyplot as plt


class Helper:
    @staticmethod
    def load_pickle(*pickle_file_path):
        with open(os.path.join(*pickle_file_path), "rb") as file:
            return pickle.load(file)

    @staticmethod
    def save_pickle(pickle_file_name, data):
        with open(pickle_file_name, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def plot_image(img_array, timeout=2):
        plt.imshow(img_array, cmap="Greys_r")
        plt.show(block=False)
        time.sleep(timeout)
        plt.close()
