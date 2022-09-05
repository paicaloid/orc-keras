import numpy as np
import string

class BaseDataLoader(object):
    def __init__(self, config):
        self.config = config
        self.characters = list(string.ascii_lowercase) + [str(x) for x in range(0, 10)]

    def get_train_data(self):
        raise NotImplementedError

    def get_test_data(self):
        raise NotImplementedError

    def split_data(self, images, labels):
        # 1. Get the total size of the dataset
        size = len(images)
        # 2. Make an indices array and shuffle it, if required
        indices = np.arange(size)
        if self.config.dataset.shuffle:
            np.random.shuffle(indices)
        # 3. Get the size of training samples
        train_samples = int(size * self.config.dataset.train_size)
        # 4. Split data into training and validation sets
        x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
        x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
        return x_train, x_valid, y_train, y_valid