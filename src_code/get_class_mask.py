import os
import numpy as np
import copy
import datetime
import argparse
import torch
from dataclasses import dataclass
from utils.utils import logger
from utils.read_data import CHANNEL_NAMES, read_eeg_data, build_dataloader, EEGDataset
from utils.mask_training import class_mask_training
from classifier.models import load_resnet18
from classifier.training import test_model

DATA_FOLDER = '../eeg_data/'            # folder where the dataset is stored
DATASET_FOLDER = './saved_datasets/'    # folder where to save the dataset
RESULTS_FOLDER = './results_class_mask/' # folder where to save the results

@dataclass
class Config:
    """
    A class to store all the configuration parameters
    """
    curr_time: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    number_of_subjects: int = 5                 # number of subjects to consider
    dataset_size: int = 0                       # size of the dataset 
    batch_size: int = 32                        # batch size for the training of the classifier with additional mask layer 
    nclasses: int = 2                           # number of classes (this must be the same as the number of classes in the classifier)
    classification: str = 'ms'                  # classification type (cq, ms, both)
    network_type: str = 'resnet18'              # network type (this section was tested only with resnet18)
    model_path: str = None                      # path to the model previously trained
    save_figures: bool = True                   # save plot of each mask in results folder
    input_channels: int = len(CHANNEL_NAMES)    # number of input channels
    timewindow: float = 1.0
    lam: float = 0.01
    device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def save_config(self, file_path):
        """
        Write all the configuration parameters to file
        """
        with open(file_path, 'w') as f:
            for key, value in self.__dict__.items():
                f.write(f'{key}: {value}\n')


def load_classifier(dataset: EEGDataset) -> torch.nn.Module:

    logger.info("Loading model...")
    model = load_resnet18(nclasses = CONFIG.nclasses, pretrained = False, device = CONFIG.device, input_channels=CONFIG.input_channels)

    # load model parameters
    model.load_state_dict(torch.load(CONFIG.model_path, map_location=torch.device('cpu')))
    # move model and model parameters to device
    model.to(CONFIG.device)

    # check model test accuracy
    _, _, testloader = build_dataloader(dataset, batch_size=CONFIG.batch_size, train_rate=0.8, 
                                        valid_rate=0.1, shuffle=True, resample=False)
    test_acc, _, _, _ = test_model(model, testloader, folder=None)
    logger.info("Test accuracy of loaded model: {:.2f}%".format(test_acc))

    return model

def get_masks(dataset: EEGDataset):
    """
    Train and save masks for each class
    """
    # load classifier
    model = load_classifier(dataset)

    os.mkdir(CONFIG.dir_path) if not os.path.isdir(CONFIG.dir_path) else None
    
    nclasses = CONFIG.nclasses 
    # adjust number of classes for classification type in ['ms', 'cq']
    if CONFIG.classification in ['ms', 'cq']:
        nclasses = 2
   
    for i in range(nclasses):
        logger.info(f'Training mask for class: {i}')

        # make a directory for the masks of the class
        mask_path_class = CONFIG.dir_path + f'/class_{i}/'
        os.mkdir(mask_path_class) if not os.path.isdir(mask_path_class) else None
    
        # before using the data, normalize them
        dataset_tmp = copy.deepcopy(dataset)
        dataset_tmp.raw = list(dataset_tmp.raw)
    
        # dataset normalization (when training the network this operation is done in the dataloader)
        min_spectr = np.min([torch.min(torch.abs(torch.tensor(dataset_tmp[i][0]))) for i in range(len(dataset_tmp))])
        max_spectr = np.max([torch.max(torch.abs(torch.tensor(dataset_tmp[i][0]))) for i in range(len(dataset_tmp))])
        
        # select from dataset instances with label i == current class
        dataset_tmp = dataset_tmp.select_class(i)
        logger.info(f'Size of data with label {i}: {len(dataset_tmp)}')

        for i, _ in enumerate(dataset_tmp):
            spectr = dataset_tmp.spectrograms[i]
            spectr = torch.tensor(spectr.real)
            spectr = torch.abs(spectr)
            spectr = (spectr - min_spectr) / (max_spectr - min_spectr)
            dataset_tmp.spectrograms[i] = spectr

        # for each element in the dataset, compute its mask
        # loop over each element in the dataset
        input = dataset_tmp[0][0] # first spectrogram
        logger.info(f'Shape of first element: {dataset_tmp[0][0].shape}')

        image_size = tuple((input.shape[1], input.shape[2]))
        nchannels = input.shape[0]
        logger.info(f'Nchannels of first element: {nchannels}')

        # extract 1000 random indices in the interval 0 - len(dataset_tmp)
        nmax = np.min([len(dataset_tmp), 1000])
        indices = np.random.choice(range(len(dataset_tmp)), replace=False, size=nmax)  
        
        spectrograms, raw_signals, labels, ids, channels = [], [], [], [], []

        for idx in indices:
            item = dataset_tmp[idx]
            spectrogram, raw_signal, label, id_, channel = item
            spectrograms.append(spectrogram)
            raw_signals.append(raw_signal)
            labels.append(label)
            ids.append(id_)
            channels.append(channel)

        # size of input: (batch_size, nchannels, image_size[0], image_size[1])
        spectrograms = torch.stack(spectrograms).float()
        logger.info(f'Shape of input: {spectrograms.shape}')
        
        labels = torch.stack([torch.tensor(label) for label in labels]).long()
        ids = np.asarray(ids)
        channels = np.asarray(channels)

        class_mask_training(model, spectrograms, raw_signals, labels, ids, channels, CONFIG.lam, 
                             mask_path_class, image_size, nchannels, figures=CONFIG.save_figures)        

        del dataset_tmp
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-ct', '--classification', type=str, default='ms', help='classification type (cq, ms, both)')
    parser.add_argument('-ic', '--input_channels', type=int, default=len(CHANNEL_NAMES), help='number of input channels')
    parser.add_argument('-ns', '--number_of_subjects', type=int, default=5, help='number of subjects to consider')
    parser.add_argument('-mp', '--model_path', type=str, default='./results_classifier/resnet18_ms/best_model_params.pt', help='path to the model')    
    parser.add_argument('-sf', '--save_figures', type=bool, default=True, help='save figures')
    parser.add_argument('--lam', type=float, default=0.001, help='regularization parameter')

    args = parser.parse_args()
    CONFIG = Config(**args.__dict__)

    # check that the given trained model exists
    if not os.path.isfile(CONFIG.model_path):
        raise FileNotFoundError(f"Model not found: {CONFIG.model_path}")
    
    # retrieve number of classes and model type from model config file
    model_config = CONFIG.model_path.replace('best_model_params.pt', 'config.txt')
    with open(model_config, 'r') as f:
        for line in f:
            if 'nclasses' in line:
                CONFIG.nclasses = int(line.split(':')[-1].strip())
            elif 'network_type' in line:
                CONFIG.network_type = line.split(':')[-1].strip()
        
    os.mkdir(RESULTS_FOLDER) if not os.path.isdir(RESULTS_FOLDER) else None
    CONFIG.dir_path = f'{RESULTS_FOLDER}/{CONFIG.network_type}_{CONFIG.classification}_{CONFIG.curr_time}' # directory to save classifier results
    os.mkdir(CONFIG.dir_path) if not os.path.isdir(CONFIG.dir_path) else None

    # get the dataset
    dataset = read_eeg_data(DATA_FOLDER, DATASET_FOLDER, input_channels=CONFIG.input_channels, 
                            number_of_subjects=CONFIG.number_of_subjects, type = CONFIG.classification)
    
    
    logger.info(f'Size of data set: {len(dataset)}')
    CONFIG.dataset_size = len(dataset)

    # save config parameters
    CONFIG.save_config(f'{CONFIG.dir_path}/config_{CONFIG.curr_time}.txt')
    
    # train and save masks for each class
    get_masks(dataset)

    
