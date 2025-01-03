import seaborn as sns
import numpy as np
import os
import copy
import pickle
import torch
import mne
from neurodsp.timefrequency.wavelets import compute_wavelet_transform
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix
from utils.utils import get_topographic_map

#############################################
## UTILITY FUNCTIONS FOR PLOTTING RESULTS  ##
#############################################


def plot_training_results(file_path: str):
    """
    Function to plot train and validation loss and accuracy
    for each epoch
    - file_path: path to the file containing the results
    """

    text = open(file_path, 'r').read()

    # read results and plot them
    train_loss, train_acc, val_loss, val_acc = [], [], [], []

    for line in text.split('\n'):
        # skip non-numerical lines
        # if line starts with alphabet, skip
        if not line or not line[0].isdigit() or len(line) == 0:
            continue
        # split line into 4 values
        line = line.split(',')
        # convert to float
        train_loss.append(float(line[1]))
        train_acc.append(float(line[2]))
        val_loss.append(float(line[3]))
        val_acc.append(float(line[4]))

    sns.set_style('darkgrid')
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train loss', color='orange', linewidth=2)
    plt.plot(val_loss, label='val loss', color='darkcyan', linewidth=2)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='train acc', color='orange', linewidth=2)
    plt.plot(val_acc, label='val acc', color='darkcyan', linewidth=2)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.savefig(file_path[:-4] + '.png')
    plt.close()




def get_confusion_matrix(ytrue: list, ypred: list, path: str = None) -> np.array:
    """
    Function to plot confusion matrix
    Input:
        ytrue: true labels
        ypred: predicted labels
        path: path to save the confusion matrix
    """
    cm = confusion_matrix(ytrue, ypred)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    # remove colorbar
    ax.collections[0].colorbar.remove()

    if path is not None:
        plt.savefig(path + '/confusion_matrix.png')
        plt.close()
    else:
        plt.show()

    return cm



def plot_loss(model: torch.nn.Module, losses: list, path: str, channels: list, figures: bool = True):
    """
    Function to plot loss of the model learning class masks
    and save the masks
    Input:
        model: model
        losses: list of losses
        path: path to save the loss plot
        channels: list of channels
        figures: whether to save an image of the mask or not
    """
    plt.figure(figsize=(30,20))
    plt.plot(losses, linewidth=5)
    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    # make ticks larger
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(path + "loss.png")
    plt.close()

    mask=model.mask.M.detach().cpu()
    mask=mask.squeeze().numpy()
   
    for j in range(len(channels[0])):
        # get mask for each channel
        mask_ch = mask[j] 
        filename = str(channels[0][j])
        plot_mask(path +"/"+ filename +".png", mask_ch) if figures else None
        # save mask
        np.save(path +"/"+ filename +".npy", mask_ch)

    del mask

def plot_mask(file_path: str, mask: np.array):
    """
    Plot the mask and save it to file_path
    """
    # plot mask
    plt.figure(figsize=(30,20))
    plt.imshow(mask,  aspect='auto', origin='lower', cmap='Blues', vmin=0)
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Frequency", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(file_path)
    plt.close()


#############################################
## PLOT FUNCTIONS FOR CORRELATION ANALYSIS ##
#############################################

def plot_correlation_differences(correlation_folder: str, classification: str, pears_th: float=0.4, cor_idx_th: float=0.005):
    """
    Plot differences in the Pearson and Id-based correlation values 
    found for the EEG raw data.
    Input:
        correlation_folder: folder containing the correlation tables
        classification: type of classification (e.g. 'cq', 'ms')
        pears_th: threshold for the Pearson correlation
        cor_idx_th: threshold for the cor_idx
    """
    band_ranges = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'full_spectrum']
    for curr_band in band_ranges:
        for c in [0,1]:
            print(f'Band: {curr_band} - Class: {c}')
            band = f'{curr_band}_band' if curr_band != 'full_spectrum' else curr_band

            table_path  =  f'{correlation_folder}/{band}_{classification}/class_{str(c)}/correlation_table.pkl'

            # open file and read tables
            with open(table_path, 'rb') as f:
                info = pickle.load(f)

            # generate plot
            plt.figure(figsize=(10, 5))
            table_cor_idx, pvalues, table_pearson = info['cor_idx'], info['pvalue'], info['pearson']

            # set inferior threshold for pearson
            cp_pearson = copy.deepcopy(table_pearson)
            cp_pearson[cp_pearson > pears_th] = 0

            # set superior threshold for cor_idx
            cp_cor_idx = copy.deepcopy(table_cor_idx)
            cp_cor_idx[pvalues > cor_idx_th] = 0

            intersection_table = copy.deepcopy(cp_cor_idx)
            intersection_table[cp_pearson == 0] = 0

            # plot pearson with threshold
            fix, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(20, 5))
            ax1.imshow(cp_pearson, cmap='coolwarm_r', vmin=-1, vmax=1)
            ax1.set_title(f'Pearson < {pears_th}', fontsize=15)

            ax2.imshow(cp_cor_idx, cmap='Blues_r')
            ax2.set_title(f'Correlation index with pvalue < {cor_idx_th}', fontsize=15)

            # intersection
            ax3.imshow(intersection_table, cmap='Greens_r')
            ax3.set_title('Differences', fontsize=15)


            # set ticks for all plots
            for ax in [ax1, ax2, ax3]:
                ax.set_xticks(np.arange(len(info['channels'])))
                ax.set_yticks(np.arange(len(info['channels'])))
                ax.set_xticklabels(info['channels'])
                ax.set_yticklabels(info['channels'])
                ax.set_xlabel('Channel 1', fontsize=15)
                ax.set_ylabel('Channel 2', fontsize=15)

            plt.savefig(f'{correlation_folder}/{band}_{classification}/class_{str(c)}/corr_differences.png', dpi=100)
            plt.close('all')
            
            # print elements of intersection > 0
            for i in range(len(info['channels'])):
                for j in range(i, len(info['channels'])):
                    if intersection_table[i, j] != 0:
                        print(f'Channels {info["channels"][i]} and {info["channels"][j]} have different correlation values  - Pearson: {cp_pearson[i, j]}, Correlation index: {cp_cor_idx[i, j]}')
            print('-----------------------------------')


def correlation_connectivity_graphs(correlation_folder: str, set_file: str, type: str='nonlinear', band: str=None, rmv_bad: str=None, pvalue_th: float=0.50, cor_idx_th: float=0.70, pears_th: float=0.70):
    """
    Plot connectivity graphs for the correlation analysis
    Input:
        correlation_folder: folder containing the correlation tables
        set_file: path to one of the original edf files (used to retrieve electrode positions)
        type: type of correlation ('cor_idx', 'pearson', 'nonlinear')
        band: band for which to plot correlation connectivity graph
        rmv_bad: if the dataset was cleaned or not
        pvalue_th: threshold for pvalue considered
        cor_idx_th: threshold for cor_idx condidered
        pears_th: threshold for pearson condidered
    """
    assert type in ['cor_idx', 'pearson', 'nonlinear'], "Error: please select from ['cor_idx', 'pearson', 'nonlinear']"
    assert rmv_bad in ['clean', 'not_clean'], "Error: please select from ['clean', 'not_clean']"
    assert band, "Error: please select a band directory."
    results_folder = f'{correlation_folder}connectivity_graphs'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    table_path  =  f'{correlation_folder}/{band}/{rmv_bad}/correlation_table.pkl' 
    assert os.path.exists(table_path), f"Error: The path '{table_path}' does not exist."

    print(f'Band: {band}')
    cp_cor_idx = []
    cp_pearson = []

    # open file and read tables
    with open(table_path, 'rb') as f:
        info = pickle.load(f)

    table_cor_idx, pvalues, table_pearson = info['cor_idx'], info['pvalue'], info['pearson']

    if type=='cor_idx':
        # plot only strong and plausible cor_idx correlations
        mask = pvalues < pvalue_th
        cp_cor_idx = copy.deepcopy(table_cor_idx)
        cp_cor_idx[cp_cor_idx < cor_idx_th] = 0
        cp_cor_idx = np.where(mask, cp_cor_idx, 0)

    elif type=='pearson':
        # plot only strong and plausible linear correlations
        cp_pearson = copy.deepcopy(table_pearson)
        cp_pearson[cp_pearson < pears_th] = 0
    elif type=='nonlinear':
        # plot only strong and plausible nonlinear correlations
        mask = pvalues < pvalue_th
        cp_pearson = copy.deepcopy(table_pearson)
        cp_cor_idx = copy.deepcopy(table_cor_idx)
        cp_pearson[cp_pearson < pears_th] = 0
        cp_cor_idx[cp_cor_idx < cor_idx_th] = 0
        cp_cor_idx = np.where(mask, cp_cor_idx, 0)
        #differences = cp_cor_idx - np.abs(cp_pearson)
        #mask = differences < 0
        #cp_cor_idx = np.where(mask, cp_cor_idx, 0)  


    # save graph
    file_path = f'{results_folder}/{type}_{band}.png'
    if type=='pearson':
        save_graph(cp_pearson, file_path, set_file, index=type, band=band, rmv_bad=rmv_bad)
    else:
        save_graph(cp_cor_idx, file_path, set_file, index=type, band=band, rmv_bad=rmv_bad)



def save_graph(graph: np.array=None, file_path: str='', set_file: str='', intensity: bool=False, index: str=None, band: str='', rmv_bad: str=''):
    """
    Function to draw and save connectivity graph for a given correlation matrix
    Input:
        graph: correlation matrix
        file_path: path to save the graph
        set_file: path to one of the original edf files (used to retrieve electrode positions)
        intensity: whether to plot the intensity of the edges
        index: index of the correlation matrix ('pearson', 'cor_idx', 'nonlinear')
    """

    # retrieve topographic map for the given eeg data
    raw, positions = get_topographic_map(set_file)

    # Original colormap
    original_cmap = plt.cm.Greys
    # Number of colors in the modified colormap
    n_colors = 256
    # Create the modified colormap
    color_array = original_cmap(np.linspace(0, 1, n_colors))
    start_index = int(n_colors * 0.3)  
    color_array[:start_index, :] = color_array[start_index, :]  # Set start colors to a specific value
    modified_cmap = mcolors.ListedColormap(color_array)

    G1 = nx.Graph()

    # create nodes using positions
    for i in range(len(positions)):
        G1.add_node(i, pos=(positions[i, 0], positions[i, 1]))

    # create edges
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            if graph[i, j] != 0:
                G1.add_edge(i, j, weight=graph[i, j])

    extent = [-0.13, 0.13, -0.13, 0.10]
    # set label names as channel names
    labels = {}
    for i in range(len(positions)):
        labels[i] = str(raw.ch_names[i])
    
    background = 'images/montage.png'
    # plot graph
    plt.figure(figsize=(10, 10))
    pos = nx.get_node_attributes(G1, 'pos')
    nx.draw(G1, pos, with_labels=True, node_size=100, font_size=10, labels=labels, font_color='black', node_color='silver', edge_color='rosybrown', width=2)
    # make the edge darker if the weight is higher
    if intensity:
        if index == 'cor_idx':
            # invert the sign of the weights
            for u,v in G1.edges:
                G1[u][v]['weight'] = np.abs(G1[u][v]['weight'])

        max_weight = max([G1[u][v]['weight'] for u,v in G1.edges]) 
        g1_edge_colors = [G1[u][v]['weight']/max_weight for u,v in G1.edges]
        nx.draw_networkx_edges(G1, pos, edge_color=g1_edge_colors, edge_cmap=modified_cmap, width=1)
    labels = nx.get_edge_attributes(G1, 'weight')

    # add a png image in the background
    im = plt.imread(background)
    plt.imshow(im, extent=extent)
    plt.title(f'{index.upper()} CORRELATIONS for ')
    plt.savefig(file_path, dpi=300)
    plt.close('all')




def plot_spectrogram(raw_signal: np.array, channel: str, id: str):
    """
    Function to show single channel spectrogram
    Input:
        raw_signal: raw signal
        channel: channel name
        id: id of the recording
    """

    spectrogram = compute_wavelet_transform(raw_signal, fs=500, n_cycles=5, freqs=np.arange(10, 70, 2))

    plt.imshow(np.abs(spectrogram), origin='lower', vmin=0,  aspect='auto', extent=[0, 250, 10, 70])
    plt.colorbar(label='Amplitude(V)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
    subject = int(id[7:9])
    epoch = id.split('_')[2]
    plt.title('Subject ' + str(subject) + ' recording - channel ' + str(channel) + ' epoch ' + str(epoch))
    plt.show()

##############################################
## PLOT FUNCTIONS FOR TOPOMAP VISUALIZATION ##
##############################################

def plot_topographic_map_freq(data: np.array, positions: np.array, raw: mne.io.Raw):
    """
    Function to plot topographic map for a given frequency
    Input:
        data: data to plot
        positions: positions of the electrodes
        raw: raw data
    """
    fig = plt.figure(figsize=(20,10))
    fig, ax1= plt.subplots(ncols=1)
    im, _ = mne.viz.plot_topomap(data, positions, ch_type='eeg', axes=ax1,  cmap="viridis", size=5, show=False, names=raw.ch_names);
    ax_x_start = 0.95
    ax_x_width = 0.03
    ax_y_start = 0.0
    ax_y_height = 0.9
    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb = fig.colorbar(im, cax=cbar_ax)
    clb.ax.set_title('dB',fontsize=15) # title on top of colorbar
    plt.show()
    
def plot_topographic_map_time(data: np.array, positions: np.array, size: int=100):
    """
    Function to plot topographic map for a given time
    """
    names = None #raw.ch_names
    # plot topographic map
    fig = plt.figure(figsize=(50,10))
    fig,(ax1,ax2, ax3, ax4) = plt.subplots(ncols=4)
    idx1, idx2, idx3, idx4 = size//4, size//2, 3*size//4, size-1
    im,cm   = mne.viz.plot_topomap(data[:,idx1], positions, ch_type='eeg', cmap='jet', axes=ax1, show=False, names=names)
    ax1.set_title(f'{2*idx1} ms')
    im,cm   = mne.viz.plot_topomap(data[:,idx2], positions, ch_type='eeg', cmap='jet', axes=ax2, show=False, names=names)   
    ax2.set_title(f'{2*idx2} ms')
    im,cm   = mne.viz.plot_topomap(data[:,idx3], positions, ch_type='eeg', cmap='jet', axes=ax3, show=False, names=names)
    ax3.set_title(f'{2*idx3} ms')
    im,cm   = mne.viz.plot_topomap(data[:,idx4], positions, ch_type='eeg', cmap='jet', axes=ax4, show=False, names=names)   
    ax4.set_title(f'{2*idx4} ms')
    # manually fiddle the position of colorbar
    ax_x_start = 0.95
    ax_x_width = 0.02
    ax_y_start = 0.3
    ax_y_height = 0.4
    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb = fig.colorbar(im, cax=cbar_ax)
    clb.ax.set_title('uV',fontsize=15) # title on top of colorbar
    plt.show()
