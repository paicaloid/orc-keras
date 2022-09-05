from base.base_trainer import BaseTrain
import os
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

class ORCModelTrainer(BaseTrain):
    def __init__(self, model, train_dataset, validation_dataset, config):
        super(ORCModelTrainer, self).__init__(model, train_dataset, validation_dataset, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        # self.train_dataset = train_dataset
        # self.validation_dataset = validation_dataset
        self.init_callbacks()
        
    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                # verbose=self.config.callbacks.checkpoint_verbose,
            )
        )
        self.callbacks.append(
            EarlyStopping(
                monitor=self.config.callbacks.checkpoint_monitor,
                patience=self.config.callbacks.early_stopping_patience,
                restore_best_weights=self.config.callbacks.restore_best_weights,
            )
        )
        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )
    
    def train(self):
        history = self.model.fit(
            self.train_dataset, 
            validation_data=self.validation_dataset,
            epochs=self.config.trainer.num_epochs,
            # verbose=self.config.trainer.verbose_training,
            batch_size=self.config.trainer.batch_size,
            # validation_split=self.config.trainer.validation_split,
            callbacks=self.callbacks,
        )
        self.loss.extend(history.history['loss'])
        # self.acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        # self.val_acc.extend(history.history['val_acc'])