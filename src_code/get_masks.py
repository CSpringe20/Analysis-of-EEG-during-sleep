import os
import numpy as np
import copy
import datetime
import argparse
import torch
from dataclasses import dataclass
from utils.utils import logger
from utils.read_data import CHANNEL_NAMES, read_eeg_data, EEGDataset
from utils.mask_training import mask_training
from get_class_mask import load_classifier

DATA_FOLDER = '../eeg_data/'            # folder where the dataset is stored
DATASET_FOLDER = './saved_datasets/'    # folder where to save the dataset
RESULTS_FOLDER = './results_mask/'      # folder where to save the results


@dataclass
class Config:
    """
    A class to store all the configuration parameters
    """
    curr_time: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    number_of_subjects: int = 5
    dataset_size: int = 0
    batch_size: int = 32
    nclasses: int = 2
    classification: str = 'ms'
    network_type: str = 'resnet18'  
    model_path: str = None
    save_figures: bool = True
    input_channels: int = len(CHANNEL_NAMES)   
    timewindow: float = 1.0
    lam: float = 0.001
    start_idx: int = 60
    end_idx: int = 0
    device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def save_config(self, file_path):
        """
        Write all the configuration parameters to file
        """
        with open(file_path, 'w') as f:
            for key, value in self.__dict__.items():
                f.write(f'{key}: {value}\n')


def get_masks(dataset: EEGDataset):
    """
    Function to get masks for each channel
    Input:
        dataset: EEGDataset object
    """
    # load classfier
    model = load_classifier(dataset)

    os.mkdir(CONFIG.dir_path) if not os.path.isdir(CONFIG.dir_path) else None

    logger.info("Training masks...")

    batch_size = CONFIG.batch_size
    start_idx = CONFIG.start_idx
    end_idx = batch_size + start_idx

    # before using the data, normalize them
    dataset_tmp = copy.deepcopy(dataset)
    dataset_tmp.raw = list(dataset_tmp.raw)

    min_spectr = np.min([torch.min(torch.abs(torch.tensor(dataset_tmp[i][0]))) for i in range(len(dataset_tmp))])
    max_spectr = np.max([torch.max(torch.abs(torch.tensor(dataset_tmp[i][0]))) for i in range(len(dataset_tmp))])
 
    for i, _ in enumerate(dataset_tmp):
        spectr = dataset_tmp.spectrograms[i]
        spectr = torch.tensor(spectr.real)
        spectr = torch.abs(spectr)
        spectr = (spectr - min_spectr) / (max_spectr - min_spectr)
        dataset_tmp.spectrograms[i] = spectr
        dataset_tmp.raw[i] = torch.tensor(dataset.get_raw(i))

  
    input = dataset_tmp[0][0]
    image_size = tuple((input.shape[1], input.shape[2]))
    input_channels = input.shape[0]
    logger.info(f"nchannels: {input_channels}")
    logger.info(f"Shape of first element: {dataset_tmp[0][0].shape}")

    while end_idx < len(dataset):
        spectrograms, raw_signals, labels, ids, channels = dataset_tmp[start_idx:end_idx]
        spectrograms = torch.stack(spectrograms).float()
        raw_signals = torch.stack(raw_signals)
        labels = torch.stack([torch.tensor(label) for label in labels]).long()
        ids = np.asarray(ids)
        channels = np.asarray(channels)

        logger.info(f"Training masks for instances: {start_idx} to {end_idx}")
        mask_training(model, spectrograms, raw_signals, labels, ids, channels, CONFIG.lam, CONFIG.dir_path, image_size, input_channels, figures=CONFIG.save_figures)
        logger.info(f"Last image processed: {end_idx}")
        start_idx = end_idx
        end_idx = batch_size + start_idx

    del dataset_tmp

    


    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-cl', '--classification', type=str, default='ms', help='classification type (cq, ms, both)')
    parser.add_argument('-ic', '--input_channels', type=int, default=len(CHANNEL_NAMES), help='number of input channels')
    parser.add_argument('-ns', '--number_of_subjects', type=int, default=5, help='number of subjects to consider')
    parser.add_argument('-mp', '--model_path', type=str, default='./results_classifier2/resnet18_ms_20240924-225218/best_model_params.pt', help='path to the model which must take as input data with just one channel')
    parser.add_argument('-sf', '--save_figures', type=bool, default=True, help='save figures')
    parser.add_argument('-tw', '--timewindow', type=float, default=0.5, help='time window for the spectrogram')
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

    # build dataset if it does not exist
    dataset = read_eeg_data(DATA_FOLDER, DATASET_FOLDER, input_channels=CONFIG.input_channels, number_of_subjects=CONFIG.number_of_subjects, type = CONFIG.classification, time_window=CONFIG.timewindow)
        
    logger.info(f"Size of data set: {len(dataset)}")

    # save config parameters
    CONFIG.save_config(f'{CONFIG.dir_path}/config_{CONFIG.curr_time}.txt')

    get_masks(dataset)