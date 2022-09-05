import tensorflow as tf

from base.base_model import BaseModel
from keras.models import Model
from keras.layers import Layer, Input, Dense, Conv2D, MaxPooling2D, Dropout, Reshape, Bidirectional, LSTM
from keras.backend import ctc_batch_cost
from keras.optimizers import Adam

class CTCLayer(Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred

class ORCModel(BaseModel):
    def __init__(self, config, char_to_num):
        super(ORCModel, self).__init__(config)
        self.img_width = config.trainer.img_width
        self.img_height = config.trainer.img_height
        self.model_name = config.model.name
        self.char_to_num = char_to_num
        self.build_model()

    def build_model(self):
        input_img = Input(
            shape=(self.img_width, self.img_height, 1), name="image", dtype="float32"
        )
        labels = Input(name="label", shape=(None,), dtype="float32")
        
        x = Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name="Conv1",
        )(input_img)
        x = MaxPooling2D((2, 2), name="pool1")(x)

        # Second conv block
        x = Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name="Conv2",
        )(x)
        x = MaxPooling2D((2, 2), name="pool2")(x)
        
        # We have used two max pool with pool size and strides 2.
        # Hence, downsampled feature maps are 4x smaller. The number of
        # filters in the last layer is 64. Reshape accordingly before
        # passing the output to the RNN part of the model
        new_shape = ((self.img_width // 4), (self.img_height // 4) * 64)
        x = Reshape(target_shape=new_shape, name="reshape")(x)
        x = Dense(64, activation="relu", name="dense1")(x)
        x = Dropout(0.2)(x)

        # RNNs
        x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.25))(x)
        x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.25))(x)

        # Output layer
        x = Dense(
            len(self.char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2"
        )(x)
        
        output = CTCLayer(name="ctc_loss")(labels, x)
        
        # Define the model
        self.model = Model(
            inputs=[input_img, labels], outputs=output, name=self.model_name
        )
        # Optimizer
        opt = Adam()
        self.model.compile(optimizer=opt)
        
    def build_predict_model(self):
        self.model.load_weights(self.config.evaluate.weights_path)
        self.predict_model = Model(
            self.model.get_layer(name="image").input, self.model.get_layer(name="dense2").output
        )