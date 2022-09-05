from data_loader.orc_data_loader import ORCDataLoader
from models.orc_model import ORCModel
from trainers.orc_trainer import ORCModelTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    print('Create the data generator.')
    data_loader = ORCDataLoader(config)
    train_dataset, validation_dataset = data_loader.get_train_data()

    print('Create the model.')
    model = ORCModel(config, data_loader.char_to_num)

    print('Create the trainer')
    trainer = ORCModelTrainer(model.model, train_dataset, validation_dataset, config)

    print('Start training the model.')
    trainer.train()


if __name__ == '__main__':
    main()
