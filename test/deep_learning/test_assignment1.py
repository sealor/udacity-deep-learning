import hashlib
import os
import pickle
from unittest import TestCase

import numpy as np
from matplotlib import image
from sklearn.linear_model import LinearRegression

from deep_learning.task_1_notmnist import maybe_download, maybe_extract, maybe_pickle, maybe_merge, data_root, \
    image_size, num_classes, test_size, train_size, valid_size
from test.deep_learning.helper import Helper


class TestAssignment1(TestCase):
    def step1_download_and_extract(self):
        train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
        test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

        self.train_folders = maybe_extract(train_filename)
        self.test_folders = maybe_extract(test_filename)

    def test_problem1(self):
        self.step1_download_and_extract()

        picture_name = os.path.join(self.train_folders[0], "a2F6b28udHRm.png")
        img = image.imread(picture_name)

        Helper.plot_image(img)

    def step2_pickle_datasets(self):
        self.step1_download_and_extract()

        self.train_datasets = maybe_pickle(self.train_folders, 45000)
        self.test_datasets = maybe_pickle(self.test_folders, 1800)

    def test_problem2(self):
        self.step2_pickle_datasets()

        a_images = Helper.load_pickle(self.test_folders[0] + ".pickle")
        Helper.plot_image(a_images[0])

    def step3_prepare_datasets(self):
        self.step2_pickle_datasets()

        maybe_merge(self.train_datasets, self.test_datasets)

    def test_problem3(self):
        self.step3_prepare_datasets()

        datasets = Helper.load_pickle(data_root, "notMNIST.pickle")

        self.assertEqual((200000, 28, 28), datasets["train_dataset"].shape)
        self.assertEqual((200000,), datasets["train_labels"].shape)

        self.assertEqual((10000, 28, 28), datasets["valid_dataset"].shape)
        self.assertEqual((10000,), datasets["valid_labels"].shape)

        self.assertEqual((10000, 28, 28), datasets["test_dataset"].shape)
        self.assertEqual((10000,), datasets["test_labels"].shape)

    def test_problem4(self):
        pickle_file_name = os.path.join(data_root, 'notMNIST.pickle')
        statinfo = os.stat(pickle_file_name)

        self.assertAlmostEqual(690800512, statinfo.st_size, delta=1024)

    def step5_hash_datasets(self):
        self.step3_prepare_datasets()

        pickle_file_name = os.path.join(data_root, 'notMNIST-hashed.pickle')

        if not os.path.exists(pickle_file_name):
            datasets = Helper.load_pickle(data_root, "notMNIST.pickle")
            valid_dataset = {hashlib.md5(array.tobytes()).hexdigest() for array in datasets["valid_dataset"]}
            train_dataset = {hashlib.md5(array.tobytes()).hexdigest() for array in datasets["train_dataset"]}
            test_dataset = {hashlib.md5(array.tobytes()).hexdigest() for array in datasets["test_dataset"]}

            Helper.save_pickle(pickle_file_name, (valid_dataset, train_dataset, test_dataset))

    def test_problem5_check_for_duplicates(self):
        self.step5_hash_datasets()

        valid_dataset, train_dataset, test_dataset = Helper.load_pickle(data_root, 'notMNIST-hashed.pickle')

        self.assertNotEqual(set(), valid_dataset.intersection(train_dataset))
        self.assertNotEqual(set(), valid_dataset.intersection(test_dataset))
        self.assertNotEqual(set(), test_dataset.intersection(train_dataset))

    def step5_create_sanitized_datasets(self):
        self.step5_hash_datasets()

        class UniqueDataExtractor:
            def __init__(self, used_image_hashes=None, indices=None):
                self.used_image_hashes = used_image_hashes or set()
                self.indices = indices or num_classes * [0]

            def extract(self, letter_pickle_file_names, size):
                dataset = np.ndarray((size, image_size, image_size), dtype=np.float32)
                labels = np.ndarray(size, dtype=np.int32)

                dataset_position = 0
                for letter_number, letter_pickle_file_name in enumerate(letter_pickle_file_names):
                    with open(letter_pickle_file_name, "rb") as file:
                        letter_dataset = pickle.load(file)
                        np.random.shuffle(letter_dataset)

                        current_size = 0
                        while current_size < size // num_classes:
                            letter_image_array = letter_dataset[self.indices[letter_number]]
                            hash = hashlib.md5(letter_image_array.tobytes()).hexdigest()

                            if hash not in self.used_image_hashes:
                                self.used_image_hashes.add(hash)
                                dataset[dataset_position, :, :] = letter_image_array
                                labels[dataset_position] = letter_number
                                current_size += 1
                                dataset_position += 1
                            self.indices[letter_number] += 1

                return dataset, labels

        pickle_file_name = os.path.join(data_root, "notMNIST-sanitized.pickle")

        if not os.path.exists(pickle_file_name):
            test_data_extractor = UniqueDataExtractor()
            test_dataset, test_labels = test_data_extractor.extract(self.test_datasets, test_size)

            train_data_extractor = UniqueDataExtractor(test_data_extractor.used_image_hashes)
            train_dataset, train_labels = train_data_extractor.extract(self.train_datasets, train_size)
            valid_dataset, valid_labels = train_data_extractor.extract(self.train_datasets, valid_size)

            datasets = {
                'train_dataset': train_dataset,
                'train_labels': train_labels,
                'valid_dataset': valid_dataset,
                'valid_labels': valid_labels,
                'test_dataset': test_dataset,
                'test_labels': test_labels,
            }
            Helper.save_pickle(pickle_file_name, datasets)

    def test_problem5_create_sanitized_datasets(self):
        self.step5_create_sanitized_datasets()

        datasets = Helper.load_pickle(data_root, "notMNIST-sanitized.pickle")

        self.assertEqual((200000, 28, 28), datasets["train_dataset"].shape)
        self.assertEqual((200000,), datasets["train_labels"].shape)

        self.assertEqual((10000, 28, 28), datasets["valid_dataset"].shape)
        self.assertEqual((10000,), datasets["valid_labels"].shape)

        self.assertEqual((10000, 28, 28), datasets["test_dataset"].shape)
        self.assertEqual((10000,), datasets["test_labels"].shape)

    def step6_use_off_the_shelf_classifier(self):
        def reduce_first_dimension(dataset, divider=10):
            return dataset.take(range(0, dataset.shape[0], divider), axis=0)

        def reshape_to_2_dimensions(dataset):
            index_dim, x_dim, y_dim = dataset.shape
            return dataset.reshape((index_dim, x_dim * y_dim))

        datasets = Helper.load_pickle(data_root, "notMNIST-sanitized.pickle")

        self.train_dataset = reduce_first_dimension(reshape_to_2_dimensions(datasets["train_dataset"]))
        self.train_labels = reduce_first_dimension(datasets["train_labels"])

        self.test_dataset = reduce_first_dimension(reshape_to_2_dimensions(datasets["test_dataset"]))
        self.test_labels = reduce_first_dimension(datasets["test_labels"])

        self.regr = LinearRegression()
        self.regr.fit(self.train_dataset, self.train_labels)

    def test_problem6(self):
        self.step6_use_off_the_shelf_classifier()

        mean_squared_error = np.mean((self.regr.predict(self.test_dataset) - self.test_labels) ** 2)
        variance_score = self.regr.score(self.test_dataset, self.test_labels)

        self.assertEqual(28 * 28, len(self.regr.coef_))
        self.assertAlmostEqual(3.1, mean_squared_error, delta=0.1)
        self.assertAlmostEqual(0.59, variance_score, delta=0.1)

        image = self.regr.coef_.reshape((28, 28))
        Helper.plot_image(image)
