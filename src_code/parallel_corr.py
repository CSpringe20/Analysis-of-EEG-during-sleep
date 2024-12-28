import os
import datetime
import numpy as np
import pickle
import argparse
from joblib import Parallel, delayed
from dataclasses import dataclass, field
from utils.correlation import plot_correlation_table, compute_correlation
from utils.masks import MaskDataset
from utils.read_data import read_eeg_data, remove_bad_chunks, CHANNEL_NAMES, EEGDataset
from utils.utils import logger


BAND_RANGES = [0.5, 4, 8, 12, 30, 60]                   # frequency bands
BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']    # corresponding bands names
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
    band: str = 'whole_spectrum'        # selected band for the analysis
    channels: list = field(default_factory=lambda: CHANNEL_NAMES)  # channels for which to compute the correlation

    def save_config(self, file_path):
        """
        Write all the configuration parameters to file
        """
        with open(file_path, 'w') as f:
            for key, value in self.__dict__.items():
                f.write(f'{key}: {value}\n')


def compute_correlation_worker(args):
    """
    Worker function to compute correlation between two channels.
    """
    ch1, ch2, dataset, results_path, mask = args
    logger.info(f"Computing correlation between {ch1} and {ch2}")

    try:
        # Select the datasets for the channels
        if not mask:
            dataset1 = dataset.select_channels(ch1)
            dataset2 = dataset.select_channels(ch2)
        else:
            mask_path = f'./results_masks_{CONFIG.classification}/'
            dataset1 = MaskDataset(ch=ch1, path=mask_path)
            dataset2 = MaskDataset(ch=ch2, path=mask_path)

        # Skip if datasets are insufficient
        if len(dataset1) <= 1 or len(dataset2) <= 1:
            logger.warning(f"Skipping correlation for {ch1} and {ch2}: insufficient data")
            return None, None, None, ch1, ch2

        # Compute correlation
        correlations = compute_correlation(dataset1, dataset2, results_path, mask=mask)
        if correlations is not None:
            cor_idx, pvalue, pearson = correlations
            return cor_idx, pvalue, pearson, ch1, ch2
    except Exception as e:
        logger.error(f"Error computing correlation for {ch1} and {ch2}: {e}")

    return None, None, None, ch1, ch2



def save_correlation_parallel(dataset: EEGDataset, dir_path: str):
    """
    Save correlation results for the specified channels using joblib for parallelism.
    """
    # Ensure the directory exists
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    results_path = os.path.join(dir_path, 'correlation.txt')
    channels = CONFIG.channels
    if CONFIG.remove_bad:
        dataset = remove_bad_chunks(dataset)

    # Save configuration
    CONFIG.save_config(results_path)

    num_channels = len(channels)
    table_cor_idx = np.zeros((num_channels, num_channels))
    table_pvalue = np.zeros((num_channels, num_channels))
    table_pearson = np.zeros((num_channels, num_channels))

    # Create tasks for all channel pairs where j >= i
    tasks = [(channels[i], channels[j], dataset, results_path, CONFIG.mask)
             for i in range(num_channels) for j in range(i, num_channels)]

    # Parallel computation using joblib
    results = Parallel(n_jobs=-1)(
        delayed(compute_correlation_worker)(task) for task in tasks
    )

    # Populate the results into the matrices
    for cor_idx, pvalue, pearson, ch1, ch2 in results:
        if cor_idx is not None:
            idx1 = channels.index(ch1)
            idx2 = channels.index(ch2)
            table_cor_idx[idx1, idx2] = cor_idx
            table_pvalue[idx1, idx2] = pvalue
            table_pearson[idx1, idx2] = pearson

    # Save tables as pkl
    table_path = os.path.join(dir_path, 'correlation_table.pkl')
    info = {'cor_idx': table_cor_idx, 'pvalue': table_pvalue, 'pearson': table_pearson, 'channels': channels}
    with open(table_path, 'wb') as f:
        pickle.dump(info, f)

    # Generate and save plots
    plot_correlation_table(table_path, info, CONFIG.mask, band_range=dir_path.split('/')[-2], kind='cor_idx')
    plot_correlation_table(table_path, info, CONFIG.mask, band_range=dir_path.split('/')[-2], kind='pearson')



if __name__ == "__main__":

    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-ns', '--number_of_subjects', type=int, default=31, help='number of subjects for which the correlation is computed')
    parser.add_argument('-ch', '--channels', type=lambda s: [str(item).upper() for item in s.split(',')], default=CHANNEL_NAMES, help='channels for which to compute the masks')
    parser.add_argument('-ne', '--nelectrodes', type=int, default=124, help='number of electrodes to be considered')
    parser.add_argument('-m', '--mask', type=bool, default=False, help='whether to use masks or raw eeg data')
    parser.add_argument('-df', '--data_folder', type=str, default=DATA_FOLDER, help='folder where the data are stored')
    parser.add_argument('-tw', '--timewindow', type=float, default=1, help='time window for the spectrogram')
    parser.add_argument('-b', '--band', type=str, default=None, choices=BANDS, help='frequency band to compute (e.g., delta, theta, alpha, beta, gamma)')
    parser.add_argument('-st','--start_time', type=int, default=0, help='starting time of the considered chunk of recording')
    parser.add_argument('-rl','--recording_length', type=int, default=60, help='length of the considered chunk of recording')
    parser.add_argument('-rb', '--remove_bad', type=bool, default=True, help='whether to remove bad chunks of time or not for the analysis')

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
    consider = 'clean' if  CONFIG.remove_bad else 'not_clean'

    # Check if a specific band is chosen
    if args.band:
        # Specific band
        band_idx = BANDS.index(args.band)
        lowcut = BAND_RANGES[band_idx]
        highcut = BAND_RANGES[band_idx + 1]
        
        filtered_dataset = dataset.filter_data(lowcut, highcut)
        band_dir_path = RESULTS_FOLDER + args.band + '_band/' + consider + '/'
        save_correlation_parallel(filtered_dataset, band_dir_path)
    else:
        # Iterate over all bands
        for idx in range(len(BAND_RANGES) - 1):
            lowcut = BAND_RANGES[idx]
            highcut = BAND_RANGES[idx + 1]
            
            filtered_dataset = dataset.filter_data(lowcut, highcut)
            band_dir_path = RESULTS_FOLDER + BANDS[idx] + '_band/' + consider + '/'
            save_correlation_parallel(filtered_dataset, band_dir_path)
        
        # Full spectrum
        band_dir_path = RESULTS_FOLDER + 'full_spectrum/' + consider + '/'
        save_correlation_parallel(dataset, band_dir_path)

    
    


            
