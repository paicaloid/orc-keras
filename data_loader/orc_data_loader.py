import os
import numpy as np
import tensorflow as tf

from base.base_data_loader import BaseDataLoader
from keras.datasets import mnist
from keras.layers import StringLookup
from pathlib import Path

class ORCDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(ORCDataLoader, self).__init__(config)
        self.char_to_num = StringLookup(
            vocabulary=list(self.characters), mask_token=None
        )
        self.num_to_char = StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
        )

    def encode_single_sample(self, img_path, label):
        # 1. Read image
        img = tf.io.read_file(img_path)
        # 2. Decode and convert to grayscale
        img = tf.io.decode_png(img, channels=1)
        # 3. Convert to float32 in [0, 1] range
        img = tf.image.convert_image_dtype(img, tf.float32)
        # 4. Resize to the desired size
        img = tf.image.resize(img, [self.config.trainer.img_height, self.config.trainer.img_width])
        # 5. Transpose the image because we want the time
        # dimension to correspond to the width of the image.
        img = tf.transpose(img, perm=[1, 0, 2])
        # 6. Map the characters in label to numbers
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        # 7. Return a dict as our model is expecting two inputs
        return {"image": img, "label": label} 
    
    def get_train_data(self):
        self.images = sorted(list(map(str, list(Path(self.config.dataset.train_path).glob("*.png")))))
        self.labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in self.images]
        x_train, x_valid, y_train, y_valid = self.split_data(np.array(self.images), np.array(self.labels))
        
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = (
            train_dataset.map(
                self.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
            )
            .batch(self.config.trainer.batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        
        validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        validation_dataset = (
            validation_dataset.map(
                self.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
            )
            .batch(self.config.trainer.batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        
        return train_dataset, validation_dataset
        
    def get_test_data(self):
        self.images = sorted(list(map(str, list(Path(self.config.dataset.test_path).glob("*.png")))))
        self.labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in self.images]
        
        test_dataset = tf.data.Dataset.from_tensor_slices((self.images, self.labels))
        test_dataset = (
            test_dataset.map(
                self.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
            )
            .batch(self.config.trainer.batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        
        return test_dataset
