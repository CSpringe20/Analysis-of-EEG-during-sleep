"""
This script contains utility functions to plot the results of the frequency masks
obtained for each class, by training a classifier with an additional layer to extract 
essential frequencies used by the network to make the classification.
"""
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mne
from utils.read_data import CHANNEL_NAMES
from utils.plot_functions import get_topographic_map

FREQUENCY_BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']

def read_class_masks(path_0: str, path_1: str, channels: list=CHANNEL_NAMES) -> tuple[np.array, np.array]:
    """
    Function to collect npy files from the two classes and store them in a list of numpy arrays
    - path_0: path to the folder containing the masks of class 0
    - path_1: path to the folder containing the masks of class 1
    - channels: list of all the channels
    """
    files = os.listdir(path_0)
    masks_channels_0 = [[]]*len(channels)
    masks_channels_1 = [[]]*len(channels)

    for file in files:
        if file.endswith('.npy') and not file.startswith('A2'): # A2 was removed being the reference electrode
            mask = np.load(os.path.join(path_0, file))
            # extract the channel name
            channel = file.split('.')[0]
            # get the index of the channel
            index = channels.index(channel)
            masks_channels_0[index]=(mask)

    shape = masks_channels_0[0].shape
    for ch in range(len(channels)):
        # if mask_channels_0[ch] is empty, then fill it with zeros
        if type(masks_channels_0[ch]) is not np.ndarray:
            masks_channels_0[ch] = np.zeros(shape)

    # repeat the same for class 1
    files = os.listdir(path_1)
    for file in files:
        if file.endswith('.npy') and not file.startswith('A2'):
            mask = np.load(os.path.join(path_1, file))
            # extract the channel name
            channel = file.split('.')[0]
            # get the index of the channel
            index = channels.index(channel)
            masks_channels_1[index]=(mask)

    for ch in range(len(channels)):
        # if mask_channels_1[ch] is empty, then fill it with zeros
        if type(masks_channels_1[ch]) is not np.ndarray:
            masks_channels_1[ch] = np.zeros(shape)

    # convert the lists to numpy arrays
    masks_channels_0 = np.array(masks_channels_0)
    masks_channels_1 = np.array(masks_channels_1)
    return masks_channels_0, masks_channels_1


def plot_mask_sum(path_0: str, path_1: str, channels: list=CHANNEL_NAMES, normalization: bool=False, classification_type: str='ms'):
    """
    Function to plot the sum of the masks for each class. 
    The sum is computed along the channels axis.
    - path_0: path to the folder containing the masks of class 0
    - path_1: path to the folder containing the masks of class 1
    - channels: list of all the channels
    - normalization: if True, the sum of the masks is normalized
    - classification_type: 'ms' for mental state classification, 'solver' for solver classification
    """
    masks_channels_0, masks_channels_1 = read_class_masks(path_0, path_1, channels)
    sns.set_style()

    # sum of masks for class 0 and 1
    sum_0 = masks_channels_0.sum(axis=0)
    sum_1 = masks_channels_1.sum(axis=0)

    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)

    if normalization:
        sum_0 = sum_0/np.max(sum_0)

    plt.imshow(sum_0, aspect='auto', origin='lower', extent=[0, 250, 0, 60], cmap='Blues') # modify extent to match the frequency bands
    plt.title('Rest state',fontsize=20) if classification_type == 'ms' else plt.title('Fast solvers', fontsize=20)

    # draw horizontal lines to separate the frequency bands
    plt.plot([0,250],[4,4], color='black', linestyle='--', linewidth=1, label='delta')
    plt.plot([0,250],[8,8], color='black', linestyle='--', linewidth=1, label='theta')
    plt.plot([0,250],[12,12], color='black', linestyle='--', linewidth=1, label='alpha')
    plt.plot([0,250],[30,30], color='black', linestyle='--', linewidth=1, label='beta')
    # set text above a line
    plt.text(0, 1, 'delta', color='black', fontsize=13)
    plt.text(0, 5, 'theta', color='black', fontsize=13)
    plt.text(0, 9, 'alpha', color='black', fontsize=13)
    plt.text(0, 13, 'beta', color='black', fontsize=13)
    plt.text(0, 31, 'gamma', color='black', fontsize=13)
    plt.xticks([])
    plt.yticks(fontsize=12)

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)

    plt.subplot(1,2,2)
    if normalization:
        sum_1 = sum_1/np.max(sum_1)

    plt.imshow(sum_1, aspect='auto', origin='lower', extent=[0, 250, 0, 60], cmap='Blues')
    plt.title('Mental workload', fontsize=20) if classification_type == 'ms' else plt.title('Slow solvers', fontsize=20)

    plt.plot([0,250],[4,4], color='black', linestyle='--', linewidth=1, label='delta')
    plt.plot([0,250],[8,8], color='black', linestyle='--', linewidth=1, label='theta')
    plt.plot([0,250],[12,12], color='black', linestyle='--', linewidth=1, label='alpha')
    plt.plot([0,250],[30,30], color='black', linestyle='--', linewidth=1, label='beta')
    # set text above a line
    plt.text(0, 1, 'delta', color='black', fontsize=13)
    plt.text(0, 5, 'theta', color='black', fontsize=13)
    plt.text(0, 9, 'alpha', color='black', fontsize=13)
    plt.text(0, 13, 'beta', color='black', fontsize=13)
    plt.text(0, 31, 'gamma', color='black', fontsize=13)

    plt.grid(False)
    plt.xticks([])
    plt.yticks(fontsize=12)

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)


  
def plot_topographic_mask(path_0: str, path_1: str, edf_file: str, channels: list=CHANNEL_NAMES, classification_type: str='ms', normalization: bool=False):
    """
    Function to plot the topographic map of the masks for each class.
    - path_0: path to the folder containing the masks of class 0
    - path_1: path to the folder containing the masks of class 1
    - channels: list of all the channels
    - edf_file: path to the edf file to get the electrode positions
    - classification_type: 'ms' for mental state classification, 'solver' for solver classification
    """
    # get electrode positions
    raw, positions = get_topographic_map(edf_file);

    masks_channels_0, masks_channels_1 = read_class_masks(path_0, path_1, channels)

    channel_mask_0 = np.zeros((masks_channels_0.shape[0] - 1))
    channel_mask_1 = np.zeros((masks_channels_1.shape[0] - 1))

    for i in range(len(channels)):
        if CHANNEL_NAMES[i] == 'A2':
            continue
        channel_mask_0[i] = np.sum(masks_channels_0[i])
        channel_mask_1[i] = np.sum(masks_channels_1[i])

    if normalization:
        channel_mask_0 = channel_mask_0/np.sum(channel_mask_0)
        channel_mask_1 = channel_mask_1/np.sum(channel_mask_1)
    

    fig = plt.figure(figsize=(15,15))
    fig, [ax1, ax2]= plt.subplots(ncols=2)

    im, _ = mne.viz.plot_topomap(channel_mask_0, positions, ch_type='eeg', axes=ax1, cmap='Blues', size=5, show=False, names=raw.ch_names);
    ax1.set_title('Rest state') if classification_type == 'ms' else ax1.set_title('Fast solvers')
    # colorbar
    fig.colorbar(im, ax=ax1, shrink=0.4)

    im, _ = mne.viz.plot_topomap(channel_mask_1, positions, ch_type='eeg', axes=ax2, cmap='Blues',  size=5, show=False, names=raw.ch_names);
    ax2.set_title('Mental workload') if classification_type == 'ms' else ax2.set_title('Slow solvers')
    # colorbar
    fig.colorbar(im, ax=ax2, shrink=0.4)

    # add space between the two plots
    plt.subplots_adjust(wspace=0.5)


def plot_topographic_mask_bands(path_0: str, path_1: str, edf_file: str, channels: list=CHANNEL_NAMES, classification_type: str='ms', normalization: bool=False):
    """
    Function to plot the topographic map of the masks for each class, for each frequency band.
    - path_0: path to the folder containing the masks of class 0
    - path_1: path to the folder containing the masks of class 1
    - channels: list of all the channels
    - classification_type: 'ms' for mental state classification, 'solver' for solver classification
    - normalization: if True, the masks are normalized
    """
    
    raw, positions = get_topographic_map(edf_file);

    masks_channels_0, masks_channels_1 = read_class_masks(path_0, path_1, channels)
    
    channel_mask_0 = np.zeros((len(FREQUENCY_BANDS), masks_channels_0.shape[0] - 1))
    channel_mask_1 = np.zeros((len(FREQUENCY_BANDS), masks_channels_1.shape[0] - 1))

    
    # sum of masks for class 0 and 1
    sum_0 = masks_channels_0.sum(axis=0)
    sum_1 = masks_channels_1.sum(axis=0)

    for j, band in enumerate(FREQUENCY_BANDS):
        if band == 'delta':
            start_idx = 0
            end_idx = 2
        elif band == 'theta':
            start_idx = 2
            end_idx = 4
        elif band == 'alpha':
            start_idx = 4
            end_idx = 6
        elif band == 'beta':
            start_idx = 6
            end_idx = 15
        elif band == 'gamma':
            start_idx = 15
            end_idx = 30
        else:
            start_idx = 0

        for i in range(len(channels)):
            if CHANNEL_NAMES[i] == 'A2':
                continue
            mask = masks_channels_0[i]
            mask0 = np.sum(mask[start_idx:end_idx, :])
            channel_mask_0[j,i] = mask0

            mask = masks_channels_1[i]
            mask1 = np.sum(mask[start_idx:end_idx, :])
            channel_mask_1[j,i] = mask1

        if normalization:
            channel_mask_0[j] = channel_mask_0[j]/np.sum(sum_0[start_idx:end_idx,:])
            channel_mask_1[j] = channel_mask_1[j]/np.sum(sum_1[start_idx:end_idx,:])
            
        

    fig, axs = plt.subplots(5, 2, figsize=(5,10))

    for j, band in enumerate(FREQUENCY_BANDS):
        im, _ = mne.viz.plot_topomap(channel_mask_0[j], positions, ch_type='eeg', axes=axs[j,0], cmap='Blues', 
                                     size=5, contours=1, show=False, names=raw.ch_names);
        axs[j,0].set_title(f'{band} band')
        fig.colorbar(im, ax=axs[j,0], shrink=0.5)

        im, _ = mne.viz.plot_topomap(channel_mask_1[j], positions, ch_type='eeg', axes=axs[j,1], cmap='Blues', 
                                     size=5, contours=1, show=False, names=raw.ch_names);
        axs[j,1].set_title(f'{band} band')
        # make subplots closer to each other
        fig.subplots_adjust(hspace=0.5)
        # colorbar
        fig.colorbar(im, ax=axs[j,1], shrink=0.5)
        
        # suptitle
        title = 'Rest state         Mental workload' if classification_type == 'ms' else 'GCQ' + ' '*20 + 'BCQ'
        fig.suptitle(title, fontsize=16, y=1.0)


    plt.subplots_adjust(wspace=.8)