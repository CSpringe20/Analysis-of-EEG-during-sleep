import os
import datetime
import numpy as np
import pickle
import argparse
from dataclasses import dataclass, field
from utils.correlation import plot_correlation_table, compute_correlation
from utils.masks import MaskDataset
from utils.read_data import read_eeg_data, remove_bad_chunks, CHANNEL_NAMES, EEGDataset
from utils.utils import logger

RANGE_ENTR = [1.1, 1.5, 3.8, 4.2]
BAND_ENTR = ['entr_word','entr_syl']
DATA_FOLDER = '../data_pp/'                             # folder where the original data are stored
DATASET_FOLDER = './saved_datasets/'                    # folder where to save the dataset
RESULTS_FOLDER = './results_correlation/'               # folder where to save the results

@dataclass
class Config:
    """
    A dataclass to store all the configuration parameters
    """
    curr_time: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    number_of_subjects: int = 31        # number of subjects for which the correlation is computed
    nelectrodes: int = 124              # number of electrodes to be considered
    input_channels: int = 1             # number of input channels (always 1, it can be the raw data or the mask but computed for one channel at a time)
    mask: bool = False                  # whether to use masks or eeg data
    data_folder: str = DATA_FOLDER      # folder where the dataset is stored
    timewindow: float = 0.5             # time window for the spectrogram
    start_time: int = 0                 # starting time of the considered chunk of recording
    recording_length: int = 60          # length of the considered chunk of recording
    remove_bad: bool = True             # whether to remove bad chunks of time or keep them for the analysis
    entrainment: str = None             # band of frequencies to check for entrainment
    test_frequency: float = 0.5         # how much to add to entrainment frequency to test for it
    channels: list = field(default_factory=lambda: CHANNEL_NAMES)  # channels for which to compute the correlation

    def save_config(self, file_path):
        """
        Write all the configuration parameters to file
        """
        with open(file_path, 'w') as f:
            for key, value in self.__dict__.items():
                f.write(f'{key}: {value}\n')


def save_correlation(dataset: EEGDataset, dir_path: str):
    """
    Save correlation results for the specified channels.
    Input:
        dataset: the dataset to use
        dir_path: the directory where to save the results
        remove_bad_time: whether to consider or not bad chunks of time
    """
    results_path = dir_path + 'correlation.txt'
    channels = CONFIG.channels
    bad = CONFIG.remove_bad
    if bad:
        dataset = remove_bad_chunks(dataset)
    # extract the band we are working on from the directory path
    band = dir_path.split('/')[3]    

    # save configuration file in the directory path
    CONFIG.save_config(results_path)

    table_cor_idx = np.zeros((len(channels), len(channels)))
    table_pvalue = np.zeros((len(channels), len(channels)))
    table_pearson = np.zeros((len(channels), len(channels)))

    # compute correlation for each pair of channels
    for i in range(len(channels)):
        for j in range(i, len(channels)):
            logger.info(f'Computing correlation between {channels[i]} and {channels[j]}\n')

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
                dataset1 = MaskDataset(ch=ch1, path=mask_path)
                dataset2 = MaskDataset(ch=ch2, path=mask_path)

            with open(results_path, 'a') as f:
                f.write(f'Dataset1 size: {len(dataset1)}\n')
                f.write(f'Dataset2 size: {len(dataset2)}\n\n')
            
            cor_idx, pvalue, pearson = None, None, None

            if len(dataset1) > 1 and len(dataset2) > 1:
                correlations = compute_correlation(dataset1, dataset2, file_path = results_path, mask = CONFIG.mask)
                if correlations is not None:
                    cor_idx, pvalue, pearson = correlations
                
                    table_cor_idx[i, j] = cor_idx
                    table_pvalue[i, j] = pvalue
                    table_pearson[i, j] = pearson


    # save tables as pkl in a file along with the channels
    table_path  = dir_path + 'correlation_table.pkl'
    # create an object with tables and channels
    info = {'cor_idx': table_cor_idx, 'pvalue': table_pvalue, 'pearson': table_pearson, 'channels': channels}

    with open(table_path, 'wb') as f:
        pickle.dump(info, f)

    # generate plot and save them in the directory
    plot_correlation_table(table_path, info, CONFIG.mask, band_range=band.replace('_', ' '), kind='cor_idx')
    plot_correlation_table(table_path, info, CONFIG.mask, band_range=band.replace('_', ' '), kind='pearson')
    

if __name__ == "__main__":

    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-ns', '--number_of_subjects', type=int, default=31, help='number of subjects for which the correlation is computed')
    parser.add_argument('-ch', '--channels', type=lambda s: [str(item).upper() for item in s.split(',')], default=CHANNEL_NAMES, help='channels for which to compute the masks')
    parser.add_argument('-ne', '--nelectrodes', type=int, default=124, help='number of electrodes to be considered')
    parser.add_argument('-m', '--mask', type=bool, default=False, help='whether to use masks or raw eeg data')
    parser.add_argument('-df', '--data_folder', type=str, default=DATA_FOLDER, help='folder where the data are stored')
    parser.add_argument('-tw', '--timewindow', type=float, default=1, help='time window for the spectrogram')
    parser.add_argument('-st','--start_time', type=int, default=0, help='starting time of the considered chunk of recording')
    parser.add_argument('-rl','--recording_length', type=int, default=60, help='length of the considered chunk of recording')
    parser.add_argument('-rb', '--remove_bad', type=bool, default=True, help='whether to remove bad chunks of time or not for the analysis')
    parser.add_argument('-en', '--entrainment', type=str, default=None, choices=BAND_ENTR, help='frequency band to compute to check for entrainment')
    parser.add_argument('-ts', '--test_frequency', type=float, default=0.5, help='frequency to add to entrainment band to test entrainment')

    # verify each channel is valid
    args = parser.parse_args()
    assert all([ch in CHANNEL_NAMES for ch in args.channels]), "Error: at least one invalid channel name {}".format(args.channels)
    assert 0.5 <= args.timewindow <= 1, "Error: timewindow must be in range [0.5, 1]"
    assert os.path.exists(args.data_folder), f"Error: dataset folder {args.data_folder} does not exist"
    assert 0 < args.recording_length, "Error: recording length must be greater than zero!"
    assert 0 <= args.start_time, "Error: starting time can't be negative!"

    CONFIG = Config(**args.__dict__)

    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
    
    dataset = read_eeg_data(CONFIG.data_folder, DATASET_FOLDER, CONFIG.input_channels, 
                            number_of_subjects=CONFIG.number_of_subjects, time_window=CONFIG.timewindow, 
                            save_spec=False, start_time=CONFIG.start_time,
                            recording_length=CONFIG.recording_length, select=CONFIG.remove_bad)
    CONFIG.dataset_size = len(dataset)
    consider = 'clean' if CONFIG.remove_bad else 'not_clean'

    # Check if a specific entrainment is chosen
    if args.entrainment:
        if args.entrainment not in BAND_ENTR:
            raise ValueError(f"Band {args.entrainment} is not a valid option. Choose from {BAND_ENTR}")
        
        band_idx = BAND_ENTR.index(args.entrainment)
        lowcut = RANGE_ENTR[band_idx]
        highcut = RANGE_ENTR[band_idx + 1]
        
        logger.info(f"Computing correlation for specified band {args.entrainment.upper()}\n")
        
        # Filter dataset for the selected band
        filtered_dataset = dataset.filter_data(lowcut, highcut)
        CONFIG.dataset_size = len(filtered_dataset)
        band_dir_path = RESULTS_FOLDER + args.entrainment + '/' + consider + '/'
        if not os.path.exists(band_dir_path):
            os.makedirs(band_dir_path)     
        # Perform correlation for each class in the chosen band
        save_correlation(filtered_dataset, band_dir_path)

        test_lowcut = RANGE_ENTR[band_idx] + args.test_frequency
        test_highcut = RANGE_ENTR[band_idx + 1] + args.test_frequency
        
        logger.info(f"Computing correlation for test frequency\n")
        
        # Filter dataset for the selected band
        filtered_dataset = dataset.filter_data(test_lowcut, test_highcut)
        CONFIG.dataset_size = len(filtered_dataset)
        band_dir_path = RESULTS_FOLDER + args.entrainment + '/' + consider + f'/plus_{args.test_frequency}/'
        if not os.path.exists(band_dir_path):
            os.makedirs(band_dir_path)     
        # Perform correlation for each class in the chosen band
        save_correlation(filtered_dataset, band_dir_path)