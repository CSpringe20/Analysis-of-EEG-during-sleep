"""
In this file, the code for training the masks which will provide the essential
frequencies for each image in the dataset is provided.
Only correctly classified images (spectrograms) will be used for training.

The following code an adaption of code available @ https://github.com/lorenzobasile/ImplicitBiasAdversarial/tree/main
Which provides the implementation for the paper: Relating Implicit Bias and Adversarial Attacks
through Intrinsic Dimension, by L. Basile, N. Karantzas, A. D'Onofrio, L. Bortolussi, A. Rodriguez, F. Anselmi

"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from utils.masks import Mask, MaskedClf
from utils.utils import logger  
from utils.plot_functions import plot_loss

def mask_training(base_model: torchvision.models, spectrograms: torch.tensor, raw_sigs: torch.tensor, labels: torch.tensor, 
              ids: str, channels: str, input_channels: int, lam: float, results_folder: str, img_size: tuple, figures: bool = False):

    '''
    Function to train essential frequency masks for each image in the dataset

    - base_model: pre-trained classifier
    - spectrograms: tensor containing a batch of images representing spectrograms    
    - raw_sigs: tensor containing a batch of 1D eeg signals
    - labels: tensor containing a batch of image labels
    - ids: ids of the images to be processed
    - channels: channel names from which the images (spectrograms) were obtained
    - input_channels: number of channels of the images
    - lam: parameter governing l_1 regularization
    - results_folder: path to the folder used to store masks
    - img_size: (height, width) of the images
    - figures: True if masks are to be stored after training, False otherwise
    '''

    loss=torch.nn.CrossEntropyLoss()
    device=torch.device("cuda" if next(base_model.parameters()).is_cuda else "cpu")

    n_instances = len(labels)
    labels = labels.to(device)
    raw_sigs = raw_sigs.to(device)
    spectrograms = spectrograms.to(device)
    if len(spectrograms.shape) > 4:
        spectrograms = spectrograms.squeeze(1)
    
    logger.info(f'Spectrograms shape: {spectrograms.shape}')
    logger.info(f'Labels shape: {labels.shape}')
    logger.info(f'Raw signals shape: {raw_sigs.shape}')

    # output of the base model
    base_out = base_model(spectrograms)

    losses=[[] for i in range(n_instances)]

    # we consider only correctly classified images
    correct_cls=(np.where((torch.argmax(base_out, axis=1)==labels).cpu())[0])

    logger.info("Start training masks...")    

    for i in range(n_instances):
        folder = results_folder + str(labels[i].item()) 
        filename = str(ids[i]) + '_' + str(channels[i])
        
        # if the file does not exist already and the image was correctly classified
        if i in correct_cls and not os.path.isfile(folder +"/"+ filename +".npy"):
            since = time.time()
            model = MaskedClf(Mask((input_channels, img_size[0], img_size[1])).to(device), base_model)
            for p in model.clf.parameters(): # freeze classifier parameters
                p.requires_grad=False

            model.mask.train()
            optimizer = torch.optim.Adam(model.mask.parameters(), lr=0.01)
            epoch = 0
            while True:
                out = model(spectrograms[i])
                l = loss(out, labels[i].reshape(1))
                penalty = model.mask.M.abs().sum()
                l += penalty*lam 
                losses[i].append(l.item())
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                # mask entries are in [0,1]
                model.mask.M.data.clamp_(0.,1.)
                epoch += 1

                # train until convergence
                if (epoch > 100 and abs(l.item()-np.mean(losses[i][-20:]))<1e-3) or epoch > 1000:
                    correct = torch.argmax(out, axis=1) == labels[i]

                    mask=model.mask.M.detach().cpu()
                    mask=mask.squeeze().numpy()
                    if np.sum(mask) <= 0:
                        logger.info("Mask is all zeros, skipping...")

                    logger.info(f'Training time for {i}th image: ', time.time()-since, ' epoch ', epoch, '- ', correct)

                    if correct:
                        #  make directory to save results
                        figures_folder = results_folder + "figures/" + str(labels[i].item()) + "/"
                        folder = results_folder + str(labels[i].item()) 
                        if not os.path.isdir(folder):
                            os.mkdir(folder)
                        if not os.path.isdir(figures_folder):
                            os.mkdir(figures_folder)

                        # plot loss
                        if figures:
                            plt.figure(figsize=(30,20))
                            plt.plot(losses[i], linewidth=5)
                            plt.xlabel("Epoch", fontsize=20)
                            plt.ylabel("Loss", fontsize=20)
                            plt.xticks(fontsize=20)
                            plt.yticks(fontsize=20)
                            plt.savefig(figures_folder + str(ids[i][0]) +"_loss.png")
                            plt.close()

                        mask=model.mask.M.detach().cpu()
                        mask=mask.squeeze().numpy()
                        
                        if input_channels == 1:
                            filename = str(ids[i]) + '_' + str(channels[i])
                            save_figure(figures_folder, filename, spectrograms[i][0], raw_sigs[i][0], mask) if figures else None
                            # save mask
                            mask_ch = np.sum(mask[0], axis=1)
                            np.save(folder +"/"+ filename +".npy", mask_ch)
                        else:
                            for j in range(input_channels):
                                # sum mask across time dimension
                                mask_ch = mask[j]
                            
                                filename = str(ids[i][j]) + '_' + str(channels[i][j])
                                save_figure(figures_folder, filename, spectrograms[i][j], raw_sigs[i][j], mask[j]) if figures else None
                                # save mask
                                np.save(folder +"/"+ filename +".npy", mask_ch)
                               
                        del mask
                    break



def save_figure(figures_folder: str, filename: str, spectrogram: torch.tensor, raw_sig: torch.tensor, mask: np.array):
    """
    Function to save the figure of the original signal, spectrogram, mask and filtered spectrogram
    """
    spectrogram = spectrogram.detach().cpu().numpy()
    img_recon = mask*spectrogram

    # plot orginal signal, corresponding spectrogram, learned mask abd filtered spectrogram 
    _, axes = plt.subplots(1, 4, gridspec_kw={'width_ratios': [2.5, 2.5, 2.5, 2.5]}, figsize=(120, 23))
    
    font_size = 80
    titles_size = 100
    # original signal
    axes[0].set_title('Original signal', fontsize=titles_size, pad=35)
    axes[0].plot(raw_sig.detach().cpu().numpy(), color='steelblue', linewidth=5)
    axes[0].tick_params(labelsize=60)
    axes[0].yaxis.offsetText.set_fontsize(font_size)
    axes[0].set_xlabel('Time', fontsize=titles_size, labelpad=25)
    axes[0].set_ylabel('Amplitude', fontsize=titles_size, labelpad=25)
    axes[0].set_xticks([])

    # corresponding spectrogram
    axes[1].set_title('Original scalogram', fontsize=titles_size, pad=35)                
    ims = axes[1].imshow(np.abs(spectrogram), aspect='auto', origin='lower', extent=[0, 250, 0, 62])
    axes[1].tick_params(labelsize=60)
    axes[1].set_xlabel('Time', fontsize=titles_size, labelpad=25)
    axes[1].set_ylabel('Frequency', fontsize=titles_size, labelpad=25)
    axes[1].set_xticks([])

    cbar = plt.colorbar(ims, ax=axes[1])
    cbar.ax.tick_params(labelsize=60)
    cbar.ax.yaxis.offsetText.set_fontsize(font_size)

    # learned mask
    axes[2].set_title('Computed mask', fontsize=titles_size, pad=35)
    ims = axes[2].imshow(mask,  aspect='auto', origin='lower', cmap='Blues', vmin=0, extent=[0, 250, 0, 62])
    axes[2].tick_params(labelsize=60)
    axes[2].set_xticks([])

    cbar = plt.colorbar(ims, ax=axes[2])
    cbar.ax.tick_params(labelsize=60)
    cbar.ax.yaxis.offsetText.set_fontsize(font_size)

    # filtered spectrogram
    axes[3].set_title('Filtered scalogram', fontsize=titles_size, pad = 35)
    ims = axes[3].imshow(np.abs(img_recon),  aspect='auto', origin='lower', extent=[0, 250, 0, 62])
    axes[3].tick_params(labelsize=60)
    axes[3].set_xlabel('Time', fontsize=titles_size, labelpad=25)
    axes[3].set_ylabel('Frequency', fontsize=titles_size, labelpad=25)
    axes[3].set_xticks([])
    
    cbar = plt.colorbar(ims, ax=axes[3])
    cbar.ax.tick_params(labelsize=60)
    cbar.ax.yaxis.offsetText.set_fontsize(font_size)


    plt.tight_layout()
    # set white space so that the same white space is present 
    # between all the subplots
    plt.subplots_adjust(wspace=0.5)

    plt.savefig(figures_folder + filename +".png")
    plt.close()



def get_correctly_classified(base_model: torchvision.models, spectrograms: torch.tensor, labels: torch.tensor, 
                             ids: np.array, channels: np.array):
    """
    Return only those data instances which are correctly classified by the pretrained model
    """
    base_out = base_model(spectrograms)
    correct_cls=(np.where((torch.argmax(base_out, axis=1)==labels).cpu())[0])
    spectrograms = spectrograms[correct_cls]
    labels = labels[correct_cls]
    ids = ids[correct_cls]
    channels = channels[correct_cls]

    return spectrograms, labels, ids, channels


def class_mask_training(base_model: torchvision.models, spectrograms: torch.tensor, raw_sigs: torch.tensor, 
                         labels: torch.tensor, ids: np.array, channels: np.array, lam: float, path: str, 
                         img_size: tuple, input_channels: int = 20, figures: bool = False):
    '''
    Function to train one single mask for all the instances of one class

    Input:
    - base_model: pre-trained classifier
    - spectrograms: all the spectrograms of one class  
    - raw_sigs: tensor containing corresponding 1D eeg signals
    - labels: tensor containing the image labels
    - ids: ids of the images to be processed
    - channels: channels from which the images (spectrograms) were obtained
    - input_channels: number of channels of the images
    - lam: parameter governing l_1 regularization
    - path: path to the folder used to store masks
    - img_size: (height, width) of the images
    - figures: True if masks are also saved as images, False otherwise
    '''

    loss=torch.nn.CrossEntropyLoss()
    device=torch.device("cuda" if next(base_model.parameters()).is_cuda else "cpu")

    labels = labels.to(device)

    # check that all labels are the same
    if not torch.all(torch.eq(labels, labels[0])):
        logger.error("Error: labels are not all the same")
        return

    spectrograms = spectrograms.to(device)
    if len(spectrograms.shape) > 4:
        spectrograms = spectrograms.squeeze(1)

    spectrograms, labels, ids, channels = get_correctly_classified(base_model, spectrograms, labels, ids, channels)
    
    logger.info(f'Spectrograms shape: {spectrograms.shape}')
    logger.info(f'Labels shape: {labels.shape}')
    logger.info('Start class specific mask training...')    

    # one single mask model for all the images of the same class
    model = MaskedClf(Mask((input_channels, img_size[0], img_size[1])).to(device), base_model)
    for p in model.clf.parameters(): # freeze classifier parameters
        p.requires_grad=False

    model.mask.train()
    optimizer = torch.optim.Adam(model.mask.parameters(), lr=0.01)
    losses = []
    max_nepochs = 5000
    
    for epoch in range(max_nepochs):
        since = time.time()
        # give to the model the full batch of spectrograms
        out = model(spectrograms)
        l = loss(out, labels)
        penalty = model.mask.M.abs().sum()
        l += penalty*lam 
        losses.append(l.item())
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        # mask entries are in [0,1]
        model.mask.M.data.clamp_(0.,1.)
        epoch += 1
        if epoch % 10 == 0:
            logger.info(f'Epoch training time: {time.time()-since} - epoch  {epoch}')
            logger.info(f'Loss: {l.item()}')

        # train until convergence, for no less than a certain number of epochs and no more than 3000 epochs
        if (epoch > 100 and abs(l.item() - np.mean(losses[-20:]))<1e-3) or epoch > 3000:
            mask=model.mask.M.detach().cpu()
            mask=mask.squeeze().numpy()

            logger.info(f'Training time: {time.time()-since} - epoch {epoch}')

            plot_loss(model, losses, path, channels, figures)
            break



