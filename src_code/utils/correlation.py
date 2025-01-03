"""
The functions provided in this file are used to compute the correlation between the essential frequencies represented by 
the masks for each channel. The correlation is computed using a novel Id based method.
"""

import numpy as np
import scipy
import scipy.stats
from tqdm import tqdm
from dadapy.data import Data
import matplotlib.pyplot as plt
from utils.utils import logger

# A function to get mean and std of cosine similarity of each pair with same index 
# of rows between the two matrices. So sim(row1_dat1, row1_dat2),... then take mean and std of these.
def cosine_sim(dataset1: np.array, dataset2: np.array) -> tuple[float, float]:
    '''
    Function to compute cosine similarity between two data sets

    Param dataset1 (np.array): first data set to correlate
    Param dataset2 (np.array): second data set to correlate

    Return: mean and standard deviation of cosine similarities
    '''
    product=np.sum(dataset1*dataset2, axis=1)
    norm1=np.linalg.norm(dataset1, 2, axis=1)
    norm2=np.linalg.norm(dataset2, 2, axis=1)
    corrs=np.divide(np.divide(product, norm1), norm2)
    return np.mean(corrs), np.std(corrs)

def compute_id(dataset, nrep=10) -> tuple:
    '''
    Function to compute Intrinsic Dimension of a data set
    Input: dataset
    Return: intrinsic dimension of data set, estimated with TwoNN (Facco et al., 2017)
    '''
    # data object, container for all dadapy functions
    # maxk is the maximum number of neighbors to consider
    data = Data(dataset, maxk=2) 
    del dataset
    
    return data.compute_id_2NN(data_fraction=0.9, n_iter=nrep)[:2]

def compute_correlation(dataset1, dataset2, file_path: str, mask: bool=False) -> tuple:
    """
    Function to compute correlation between the essential frequencies represented by
    the masks for each channel or between raw signals themselves. 
    The correlation is computed using a novel Id based method.

    The ID for the dataset made of the concatenation of the two masks (or the two raw signals) is computed,
    along with the ID of the two separated masks. After that the correlation index is computed.
    Then, the p-value is computed by estimating the ID for shuffles of first set of data
    in the concatenated dataset and comparing it with cor_idx.
    The p-value is computed to assess the significance of the correlation index.

    Input:
        dataset1 (MaskDataset or EEGDataset): first dataset
        dataset2 (MaskDataset or EEGDataset): second dataset
        file_path: path to the file where to store the results
        mask: whether to use raw data or masks learned on spectrograms

    Return: 
        cor_idx: index of the correlation
        pvalue: pvalue of the correlation index
        pearson_coeff: Pearson coefficient of the correlation
    """

    # get indices of the masks of the two lobes
    _, ind1, ind2 = np.intersect1d(dataset1.id, dataset2.id, assume_unique=True, return_indices=True)
    
    if not mask:
        dataset1.raw = np.array(dataset1.raw)
        dataset2.raw = np.array(dataset2.raw)

        data1 = dataset1.raw[np.array(ind1)]
        data2 = dataset2.raw[np.array(ind2)]       
        
        data1 = np.array(data1, dtype=np.float32)
        data2 = np.array(data2, dtype=np.float32)
        # remove dimension 1
        if len(data1.shape) > 2:
            data1 = np.squeeze(data1, axis=1)
            data2 = np.squeeze(data2, axis=1)

    else:
        data1=np.array(dataset1.masks)[ind1.astype(int)]
        data2=np.array(dataset2.masks)[ind2.astype(int)]        
        # sum along temporal dimension
        data1 = np.sum(data1, axis=1)
        data2 = np.sum(data2, axis=1)

        data1 = data1.reshape(-1, data1.shape[1]*data1.shape[2])
        data2 = data2.reshape(-1, data2.shape[1]*data2.shape[2])
    
    # if the number of data points is too low, return
    if len(data1) < 5:
        try:
            with open(file_path, 'a+') as f:
                f.write(f'Error: Too few matching data provided for correlation computation. Found only {len(data1)} in union\n')
            return 
        except:
            logger.error(f'Error: Too few matching data provided for correlation computation. Found only {len(data1)} in union\n')
            return
        
    
    mu, sigma = cosine_sim(data1, data2)
    full_data = np.concatenate([data1, data2], axis=1)
    idC, err = compute_id(full_data, nrep=5)
    id1, _ = compute_id(data1, nrep=5)
    id2, _ = compute_id(data2, nrep=5)
    
    cor_idx = (id1 + id2 - idC) / max(id1, id2)
    
    # compute Pearson coefficient
    pearson_coeff = np.corrcoef(data1.flatten(), data2.flatten())
    assert pearson_coeff.shape == (2, 2), f'Error: Pearson coefficient has shape {pearson_coeff.shape}'
    pearson_coeff = pearson_coeff[0,1] 
    
    S = 0
    C = 0
    while S < 100:
        for _ in tqdm(range(50)):
            permutation = np.random.permutation(data1)
            del full_data
            full_data=np.concatenate([permutation, data2], axis=1)
            id, _ = compute_id(full_data, nrep=1)
            if id <= idC:
                C += 1
            S += 1
    
    pvalue = (C + 1) / (S + 1)
    
    try:
        with open(file_path, 'a') as f:
            f.write(f'Number of data: {len(data1)}\n')
            f.write(f'Data shape: {data1.shape}\n')
            f.write(f'Pearson coefficient: {pearson_coeff}\n')

            f.write(f'Cosine similarity: mean {mu}, std {sigma}\n')
            f.write(f'Correlation index: {cor_idx}\n')
            f.write(f'P-value: {pvalue}\n')
            f.write(f'---------------------------------------\n\n')
    except:
        logger.error('Error: cannot write on file')

    return cor_idx, pvalue, pearson_coeff




def plot_correlation_table(path: str, results: dict, mask: bool, band_range: str, kind:str = 'cor_idx'):
    """
    Function to save the correlation tables as images
    """
    if kind == 'cor_idx':
        table = results['cor_idx']
        pvalues = results['pvalue']
        title = 'Correlation index between raw signals' if not mask else 'Correlation index between masks'
        title += ' - ' + band_range 
        file_path = path[:-9]+'cor_idx.png'

    elif kind == 'pearson':
        table = results['pearson']
        # make table symmetric
        # table = table + table.T - np.diag(np.diag(table))
        title = 'Pearson Coefficient between raw signals' if not mask else 'Pearson Coefficient between masks'
        title += ' - ' + band_range 
        file_path = path[:-9]+'pearson.png'
    
    else:
        logger.error('Error: invalid kind of correlation')
        return
    
    channels = results['channels']

    # plot correlation table for cor_idx and pvalue
    fig, ax = plt.subplots(figsize=(25, 25))
    im = ax.imshow(table, cmap='Blues') if kind == 'cor_idx' else ax.imshow(table, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(channels)))
    ax.set_yticks(np.arange(len(channels)))
    ax.set_xticklabels(channels)
    ax.set_yticklabels(channels)
    
    ax.set_title(title, fontsize=20)
    ax.set_xlabel('Channel 1', fontsize=20)
    ax.set_ylabel('Channel 2', fontsize=20)
    # put cor_idx in each cell
    for i in range(len(channels)):
        for j in range(i, len(channels)):
            _ = ax.text(j, i, round(table[i, j], 2), ha="center", va="center", color="k", fontsize=1)
            # and also pvalue in scientific notation
            if kind == 'cor_idx':
                _ = ax.text(j, i+0.3, '{:.1E}'.format(pvalues[i, j]), ha="center", va="center", color="k", fontsize=1)

    plt.colorbar(im, ax=ax, shrink=0.8)
    # increase image resolution
    plt.savefig(file_path, dpi=800)
    plt.close()

    
