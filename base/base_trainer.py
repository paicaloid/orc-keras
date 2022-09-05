class BaseTrain(object):
    def __init__(self, model, train_dataset, validation_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.config = config

    def train(self):
        raise NotImplementedError
