"""
This script is meant to obtain a graphical representation of Id estimation with respect to the percentage 
of available data used, both for shuffle and unshuffled data points.
"""
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from utils.correlation import compute_id
from utils.read_data import read_eeg_data, CHANNEL_NAMES, EEGDataset
from utils.utils import logger

DATA_FOLDER = '../eeg_data/'                # folder where the dataset is stored
DATASET_FOLDER = './saved_datasets/'        # folder where to save the dataset
RESULTS_FOLDER = './results_correlation/'   # folder where to save the results


def estimate_id(dataset: EEGDataset, ch1: str, ch2: str, results_path: str):
    """
    Function to estimate the Id for two given channels of the dataset.
    The Id is estimated for different percentages of data used and the results 
    are saved and plotted to evaluate the performance of the Id estimation.
    """

    # select two channels from the dataset
    d1 = dataset.select_channels(ch1)
    d2 = dataset.select_channels(ch2)
        
    std_dev_noshuffle = []
    std_dev_shuffle = []
    noshuffle = []
    shuffle = []

    full_data = np.concatenate([d1.raw, d2.raw], axis=1)
    # convert to float32
    full_data = np.array(full_data, dtype=np.float32)
    max_size = full_data.shape[0]

    # find a list of integers divisor for the number of data points
    divisors = []
    for i in range(4, max_size+1):
        if max_size % i == 0:
            divisors.append(i)

    nelements = [i for i in divisors if i <= len(full_data)]
    logger.info('nelements to consider: ', nelements)
    
    for j in nelements:
        # split full data into j subsets randomly
        nsubsets = int(len(full_data)//j)
        subsets = np.array_split(full_data, nsubsets, axis=0)

        logger.info(f'n subsets: {nsubsets}')
        logger.info(f'subset shape: {subsets[0].shape}')

        noshuffle_partial_id = []
        noshuffle_partial_err = []
        shuffle_partial_id = []
        shuffle_partial_err = []

        for i in range(min(nsubsets, 300)):
            subset = subsets[i]
            # estimate id for original data
            id, err = compute_id(subset, nrep=5)
            noshuffle_partial_id.append(id)
            noshuffle_partial_err.append(err)

            shuffle_id = []
            shuffle_err = []
            zscore = 2.5
            repeat = 0
            while zscore > 1. and repeat < 150:
                for _ in range(50):
                    left, right = subset[:, :int(subset.shape[1]/2)], subset[:, int(subset.shape[1]/2):]
                    permutation = np.random.permutation(left)
                    data=np.concatenate([permutation, right], axis=1)
                    # estimate id for shuffled data
                    shuffle_id, shuffle_err = compute_id(data, nrep=3)
                    shuffle_partial_id.append(shuffle_id)
                    shuffle_partial_err.append(shuffle_err)
                    
                    repeat += 1
                
                zscore = (id - np.mean(shuffle_partial_id))/np.std(shuffle_partial_id)            

        shuffle.append(np.mean(shuffle_partial_id))
        std_dev_shuffle.append(np.mean(shuffle_partial_err))

        noshuffle.append(np.mean(noshuffle_partial_id))
        std_dev_noshuffle.append(np.mean(noshuffle_partial_err))

    file_path = f'{results_path}.txt'

    # create folder if it does not exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # write results to file
    with open(file_path, 'w') as f:
        # write header
        f.write(f'Id estimation for channels {ch1} - {ch2}\n')
        f.write('nelements, noshuffle id, noshuffle err, shuffle id, shuffle err\n')
        for i in range(len(noshuffle)):
            f.write(f'{nelements[i]}, {noshuffle[i]}, {std_dev_noshuffle[i]}, {shuffle[i]}, {std_dev_shuffle[i]}\n')

    # put the results into a unique list of lists
    results = [nelements, noshuffle, std_dev_noshuffle, shuffle, std_dev_shuffle]

    # plot results
    plot_id_estimation(results_path, results, ch1, ch2)


def plot_id_estimation(path: str, results: list, ch1: str, ch2: str):
    """
    Function to plot the results of the Id estimation for different percentages of data used.
    Input:
        - path: path where to save the plot
        - results: list of lists containing the results of the Id estimation
        - ch1, ch2: channels used to compute the Id
    """
    
    nelements, noshuffle, std_dev_noshuffle, shuffle, std_dev_shuffle = results
    fig, ax = plt.subplots(figsize=(15, 10))
    clrs = sns.color_palette("cool", 2)

    std_m = [noshuffle[i]-std_dev_noshuffle[i] for i in range(len(noshuffle))]
    std_p = [noshuffle[i]+std_dev_noshuffle[i] for i in range(len(noshuffle))]
    ax.plot(nelements, noshuffle, c=clrs[0], label='No shuffle')
    ax.scatter(nelements, noshuffle, c=clrs[0], s=10)
    ax.fill_between(nelements, std_m, std_p,alpha=0.3, facecolor=clrs[0])

    std_mm = [shuffle[i]-std_dev_shuffle[i] for i in range(len(shuffle))]
    std_pp = [shuffle[i]+std_dev_shuffle[i] for i in range(len(shuffle))]
    ax.plot(nelements, shuffle, c=clrs[1], label='Shuffle')
    ax.scatter(nelements, shuffle, c=clrs[1], s=10)
    ax.fill_between(nelements, std_mm, std_pp, alpha=0.3, facecolor=clrs[1])
    ax.legend(fontsize=20)
    ax.title.set_text('Id estimation for channels {ch1} - {ch2}'.format(ch1=ch1, ch2=ch2))
    # set title fontsize
    ax.title.set_fontsize(25)
    # save plot    
    plt.savefig(f'{path}.png')
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-ns', '--number_of_subjects', type=int, default=36, help='number of subjects for which the correlation is computed')
    parser.add_argument('-ct','--classification', type=str, default='cq', help='classification type (cq, ms, both)')
    parser.add_argument('-ch', '--channels', type=lambda s: [str(item).upper() for item in s.split(',')], default=['F3', 'T4'], help='channels for which to compute the masks')
    parser.add_argument('-ne', '--nelectrodes', type=int, default=20, help='number of electrodes to be considered')
    parser.add_argument('-m', '--mask', type=bool, default=False, help='whether to use masks or raw eeg data')
    parser.add_argument('-df', '--dataset_folder', type=str, default=DATA_FOLDER, help='folder where the dataset is stored')

    # set the number of classes
    if parser.parse_args().classification == 'both':
        nclasses = 3
    else:
        nclasses = 2

    # verify channels are just a pair
    channels = parser.parse_args().channels
    if len(channels) != 2 or channels[0] not in CHANNEL_NAMES or channels[1] not in CHANNEL_NAMES:
        logger.error('Error: invalid number of channels')
        exit()
    
    input_channels = 1
    number_of_subjects = parser.parse_args().number_of_subjects
    classification = parser.parse_args().classification
    time_window = parser.parse_args().timewindow

    # where to save the dataset
    dataset = read_eeg_data(DATA_FOLDER, DATASET_FOLDER, input_channels, number_of_subjects, type=classification, save_spec=False)
    dataset.raw = np.array(dataset.raw)

    id_estimation_folder = f'{RESULTS_FOLDER}/id_estimation/'
    os.makedirs(id_estimation_folder, exist_ok=True)

    for i in range(nclasses):
        dataset_i = dataset.select_class(i)

        # choose two channels to construct the dataset for which
        # we want to estimate the Id
        ch1, ch2 = channels

        if ch1 not in dataset_i.channel or ch2 not in dataset_i.channel:
            logger.error('Error: invalid channel name, channel not in dataset')
            exit()
        
        file_path = f'{id_estimation_folder}/id_estimation_{ch1}_{ch2}_class_{i}'
        estimate_id(dataset_i, ch1, ch2, file_path)

    # plot results for both classes
    file_path = f'{id_estimation_folder}/id_estimation_{ch1}_{ch2}_class_both'
    estimate_id(dataset, ch1, ch2, file_path)
    

    
    


            
