from data_loader.orc_data_loader import ORCDataLoader
from models.orc_model import ORCModel
from evaluater.orc_evaluater import ORCEvaluater
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
        
    print('Create the data generator.')
    data_loader = ORCDataLoader(config)
    test_dataset = data_loader.get_test_data()
    
    print('Create the model.')
    model = ORCModel(config, data_loader.char_to_num)
    model.build_model()
    model.build_predict_model()
    
    print('Create the evaluater')
    evaluater = ORCEvaluater(model.predict_model, test_dataset, config, data_loader.num_to_char)
    evaluater.evaluate(data_loader.labels)
    print()

if __name__ == '__main__':
    main()
