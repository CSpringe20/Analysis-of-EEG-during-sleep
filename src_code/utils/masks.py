import torch
import os
import numpy as np


class Mask(torch.nn.Module):
    '''
    Class that inherits from torch.nn.Module, implementing the Fourier modulatory mask as a pre-processing layer
    Attribute M (torch.tensor): tensor that stores the entries of the mask
    '''

    def __init__(self, mask_size: tuple = (1, 32, 32)):
        super().__init__()
        assert len(mask_size) == 3
        kernel = torch.ones((1, *mask_size))
        self.M = torch.nn.Parameter(kernel)
        torch.nn.init.ones_(self.M)

    def forward(self, x):
        x = self.M * x       # multiply by mask
        return x
   
   
class MaskedClf(torch.nn.Module):
    '''
    Class that inherits from torch.nn.Module, implementing a end-to-end 'masked' classifier

    Attribute mask (Mask): pre-processing layer doing mask modulation
    Attribute clf (torch.nn.Module): pre-trained classifier
    '''

    def __init__(self, mask, clf):
        super().__init__()
        self.mask=mask
        self.clf=clf

    def forward(self, x):
        x=self.mask(x)  # apply mask
        x=self.clf(x)   # apply classifier
        return x



class MaskDataset(torch.utils.data.Dataset):
    '''
    Class that inherits from torch.utils.data.Dataset to store masks

    Input:
    - path (str): path to the masks
    - nclasses (int): number of classes
    - sel_class (int): class to select
    - ch (str): channel to select

    Attributes:
    - masks (list): list of masks
    - labels (list): list of labels
    - id (list): list of ids
    - channel (list): list of channels
    '''

    def __init__(self, ch=None, path = None, nclasses=None, sel_class=None):

        self.masks=[]
        self.labels=[]
        self.id=[]
        self.channel=[]

        if path is not None:
            classes = np.arange(nclasses) if sel_class is None else np.asarray([sel_class])
            for c in classes:
                class_list=sorted(os.listdir(path+str(c)))
                
                for mask in class_list:
                    channel = str(mask).split('_')[3][:-4]
                    if (ch is not None and channel != ch): # if channel is specified, skip masks that do not belong to that channel
                        continue
                    self.masks.append(np.load(path+str(c)+"/"+mask, allow_pickle=True))
                    self.labels.append(c)
                    id = f'{str(mask)[7:].split("_")[0]}_{str(mask).split("_")[1]}_{str(mask).split("_")[2]}' # subject + label + epoch
                    self.id.append(id) 
                    self.channel.append(channel)

        
    def add_instance(self, elem):
        self.masks.append(elem[0])
        self.labels.append(elem[1])
        self.id.append(elem[2])
        self.channel.append(elem[3])
    
    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        return self.masks[idx], self.labels[idx], self.id[idx], self.channel[idx]
