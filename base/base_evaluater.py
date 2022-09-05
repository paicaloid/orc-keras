class BaseEvaluater(object):
    def __init__(self, model, test_dataset, config):
        self.model = model
        self.test_dataset = test_dataset
        self.config = config
    
    def evaluate(self):
        raise NotImplementedError