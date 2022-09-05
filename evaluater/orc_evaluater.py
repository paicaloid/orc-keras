import numpy as np

import tensorflow as tf
from base.base_evaluater import BaseEvaluater
from keras.backend import ctc_decode


class ORCEvaluater(BaseEvaluater):
    def __init__(self, model, test_dataset, config, num_to_char):
        super(ORCEvaluater, self).__init__(model, test_dataset, config)
        self.test_dataset = test_dataset
        self.config = config
        self.num_to_char = num_to_char
        
    def evaluate(self, labels):
        pred = self.model.predict(self.test_dataset)
        pred_texts = self.decode_batch_predictions(pred)
        
        total_text = 0
        match_count = 0
        for label, res_pred in zip(labels, pred_texts):
            if label == res_pred:
                match_count += 1
            else:
                print("{} != {}".format(label, res_pred))
            total_text += 1
            
        print("Accuracy: {}".format(match_count / total_text))
        
    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        results = ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
            :, :self.config.trainer.max_length
        ]
        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(self.num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text