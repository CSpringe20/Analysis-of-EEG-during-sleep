import os
import sys
import datetime
import numpy as np
import pickle
import argparse
from dataclasses import dataclass, field
from utils.correlation import plot_correlation_table, compute_correlation
from utils.masks import MaskDataset
from utils.read_data import read_eeg_data, CHANNEL_NAMES, EEGDataset
from utils.utils import logger


BAND_RANGES = [0.5, 4, 8, 12, 30, 60]                   # frequency bands
BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']    # corresponding bands names
DATA_FOLDER = '../eeg_data/'                            # folder where the original data are stored
DATASET_FOLDER = './saved_datasets/'                    # folder where to save the dataset
RESULTS_FOLDER = './results_correlation/'               # folder where to save the results
CLASSIFICATIONS = {'cq': ['good_quality_count', 'bad_quality_count'], 
                   'ms': ['rest', 'counting'], 
                   'both': ['rest', 'good_quality_count', 'bad_quality_count']} # classification types

@dataclass
class Config:
    """
    A dataclass to store all the configuration parameters
    """
    curr_time: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    number_of_subjects: int = 36        # number of subjects for which the correlation is computed
    nclasses: int = 2                   # number of classes for the classification
    classification: str = 'cq'          # classification type (cq, ms, both)
    nelectrodes: int = 20               # number of electrodes to be considered
    input_channels: int = 1             # number of input channels (always 1, it can be the raw data or the mask but computed for one channel at a time)
    mask: bool = False                  # whether to use masks or eeg data
    data_folder: str = DATA_FOLDER      # folder where the dataset is stored
    timewindow: float = 0.5               # time window for the spectrogram
    channels: list = field(default_factory=lambda: CHANNEL_NAMES)  # channels for which to compute the correlation
    
    def save_config(self, file_path):
        """
        Write all the configuration parameters to file
        """
        with open(file_path, 'w') as f:
            for key, value in self.__dict__.items():
                f.write(f'{key}: {value}\n')


def save_correlation(dataset: EEGDataset, dir_path: str, sel_class: int=None):
    """
    Save correlation results for the specified channels.
    Input:
        dataset: the dataset to use
        dir_path: the directory where to save the results
    """
    results_path = dir_path + 'correlation.txt'
    channels = CONFIG.channels
    nclasses = CONFIG.nclasses

    # extract the band we are working on from the directory path
    band = dir_path.split('/')[3]

    # save configuration file in the directory path
    CONFIG.save_config(results_path)

    table_zscore = np.zeros((len(channels), len(channels)))
    table_pvalue = np.zeros((len(channels), len(channels)))
    table_pearson = np.zeros((len(channels), len(channels)))

    # compute correlation for each pair of channels
    for i in range(len(channels)):
        for j in range(i+1, len(channels)):
            logger.info(f'Computing correlation between {channels[i]} and {channels[j]}\n')

            # if not computed already, compute correlation
            if table_zscore[i, j] != 0:
                continue

            with open(results_path, 'a') as f:
                f.write(f'\n\nComputing correlation for {channels[i]} and {channels[j]}\n\n')

            ch1, ch2 = channels[i], channels[j]

            # if mask is false, use raw data, otherwise use masks
            if not CONFIG.mask:
                dataset1 = dataset.select_channels(ch1)
                dataset2 = dataset.select_channels(ch2)
            else:
                mask_path = f'./results_masks_{CONFIG.classification}/'
                # get the masks for the two channels
                dataset1 = MaskDataset(ch=ch1, path=mask_path, nclasses=nclasses, sel_class=sel_class)
                dataset2 = MaskDataset(ch=ch2, path=mask_path, nclasses=nclasses, sel_class=sel_class)

            with open(results_path, 'a') as f:
                f.write(f'Dataset1 size: {len(dataset1)}\n')
                f.write(f'Dataset2 size: {len(dataset2)}\n\n')
            
            zscore, pvalue = None, None

            if len(dataset1) > 1 and len(dataset2) > 1:
                correlations = compute_correlation(dataset1, dataset2, file_path = results_path, mask = CONFIG.mask)
                if correlations is not None:
                    zscore, pvalue, pearson = correlations
                
                    table_zscore[i, j] = zscore.item()
                    table_pvalue[i, j] = pvalue.item()
                    table_pearson[i, j] = pearson


    # save tables as pkl in a file along with the channels
    table_path  = dir_path + 'correlation_table.pkl'
    # create an object with tables and channels
    info = {'zscore': table_zscore, 'pvalue': table_pvalue, 'pearson': table_pearson, 'channels': channels}

    with open(table_path, 'wb') as f:
        pickle.dump(info, f)

    # generate plot and save them in the directory
    plot_correlation_table(table_path, info, CONFIG.mask, band_range=band.replace('_', ' '), kind='zscore')
    plot_correlation_table(table_path, info, CONFIG.mask, band_range=band.replace('_', ' '), kind='pearson')



def correlation_per_class(dataset: EEGDataset, dir_path: str):
    """
    Compute correlation separately for each class of the dataset.
    """
    for i in range(CONFIG.nclasses):
        logger.info(f"Computing correlation for class {i} of classification problem {CONFIG.classification}, which corresponds to {CLASSIFICATIONS[CONFIG.classification][i]}")
        dataset.raw = np.array(dataset.raw)
        dataset_i = dataset.select_class(i)
        dir_path_i = dir_path + f'class_{i}/'

        if not os.path.exists(dir_path_i):
            os.makedirs(dir_path_i)
        save_correlation(dataset_i, dir_path=dir_path_i)

    

if __name__ == "__main__":

    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-ns', '--number_of_subjects', type=int, default=36, help='number of subjects for which the correlation is computed')
    parser.add_argument('-ct','--classification', type=str, default='cq', help='classification type (cq, ms, both)')
    parser.add_argument('-ch', '--channels', type=lambda s: [str(item).upper() for item in s.split(',')], default=CHANNEL_NAMES, help='channels for which to compute the masks')
    parser.add_argument('-ne', '--nelectrodes', type=int, default=20, help='number of electrodes to be considered')
    parser.add_argument('-m', '--mask', type=bool, default=False, help='whether to use masks or raw eeg data')
    parser.add_argument('-df', '--data_folder', type=str, default=DATA_FOLDER, help='folder where the data are stored')
    parser.add_argument('-tw', '--timewindow', type=int, default=.5, help='time window for the spectrogram')

    # verify each channel is valid
    assert all([ch in CHANNEL_NAMES for ch in parser.parse_args().channels]), "Error: at least one invalid channel name {}".format(parser.parse_args().channels)
    # verify timewindow is in range [0.5, 1]
    assert 0.5 <= parser.parse_args().timewindow <= 1, "Error: timewindow must be in range [0.5, 1]"
    # verify dataset folder exists
    assert os.path.exists(parser.parse_args().data_folder), "Error: dataset folder {} does not exist".format(parser.parse_args().data_folder)

    # if mask is false classification type must be one of the two [cq, ms] and number of classes must be 2
    # indeed using three different classes makes sense only if a different classifier has been trained
    # otherwise raw data are the same as for the ms classification 
    assert parser.parse_args().mask or str(parser.parse_args().classification) in ['cq', 'ms', 'both'], "Error: invalid classification type"

    args = parser.parse_args()
    
    # set the number of classes
    if str(parser.parse_args().classification) == 'both':
        args.nclasses = 3
    else:
        args.nclasses = 2

    CONFIG = Config(**args.__dict__)

    # create directory to store results
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    # get dataset
    dataset = read_eeg_data(CONFIG.data_folder, DATASET_FOLDER, CONFIG.input_channels, number_of_subjects=CONFIG.number_of_subjects, type = CONFIG.classification, time_window=CONFIG.timewindow, save_spec=False)

    CONFIG.dataset_size = len(dataset)
    
    if not CONFIG.mask:
        # 1. compute correlation for each band
        for idx in range(len(BAND_RANGES)-1):
            logger.info(f"Computing correlation for band {BANDS[idx].upper()}\n")
            lowcut = BAND_RANGES[idx]
            highcut = BAND_RANGES[idx+1]
         
            # filter dataset wrt the band
            filtered_dataset = dataset.filter_data(lowcut, highcut)
            CONFIG.dataset_size = len(filtered_dataset)

            band_dir_path = RESULTS_FOLDER + '/' + BANDS[idx] + '_band_'  + str(CONFIG.classification) + '/'

            # split dataset in more datasets one for each class 
            correlation_per_class(filtered_dataset, band_dir_path) 
   
        # 2. compute correlation for the full spectrum
        band_dir_path = RESULTS_FOLDER + '/' +  '/full_spectrum_' + str(CONFIG.classification) + '/'
        if not os.path.exists(band_dir_path):
            os.makedirs(band_dir_path) 

        correlation_per_class(dataset, band_dir_path)
    
    else:
        logger.info(f"Computing correlation for the whole dataset\n")
        #dataset.raw = np.array(dataset.raw)

        res_path = RESULTS_FOLDER + '/full_spectrum_masks_' + str(CONFIG.classification) + '/'
        if not os.path.exists(res_path):
            # print error and exit if folder does not exist
            sys.exit(f"Error: masks folder {res_path} does not exist")

        # save correlation for each class
        for i in range(CONFIG.nclasses):

            class_path = res_path +  f'/class_{i}/'

            if not os.path.exists(class_path):
                os.makedirs(class_path)
            
            for i in range(CONFIG.nclasses):
                save_correlation(dataset, dir_path=class_path, sel_class=i)

        # compute correlation for the whole dataset
        all_path = res_path + '/class_both/'
        save_correlation(dataset, dir_path=all_path)


    

    
    


            
