import os
import numpy as np
import math
import csv
import mne
import copy
import pickle
import scipy.signal
import torch
from torch.utils.data import DataLoader
from neurodsp.timefrequency.wavelets import compute_wavelet_transform
from utils.utils import logger
from utils.constants import ch_names, subject_id

CHANNEL_NAMES = ch_names
SUBJECT_ID = subject_id

class EEGDataset(torch.utils.data.Dataset):
    """
    A custom dataset class for the EEG data
    Input:
        spectrograms: a list of spectrograms
        raw: a list of raw eeg data
        bad_time: a list of chunks of data to discard
        id: a list of subject ID and recording ID
        channel: a list of channel names
        info: a dictionary containing all the information about the dataset
    """

    def __init__(self, spectrograms, raw, bad_time, id, channel, info):
        self.spectrograms = spectrograms # spectrograms
        self.raw = np.array(raw) # raw eeg data
        self.bad_time = bad_time # list of chunks to discard
        self.id = id  # subject ID and recording ID
        self.channel = channel # channel name
        self.info = info
        

    def __len__(self):
        return len(self.spectrograms)
    
    def __getinfo__(self):
        # get info attached to eeg dataset, which are necessary 
        # to recover a bunch of information about the dataset itself
        return self.info
    
    def __getshape__(self):
        # get shape of the dataset
        return self.spectrograms[0].shape[2]

    def __getitem__(self, idx):
        return self.spectrograms[idx], self.raw[idx], self.labels[idx], self.id[idx], self.channel[idx]
    
    def get_spectrogram(self, idx):
        return self.spectrograms[idx]
    
    def get_raw(self, idx):
        return self.raw[idx]
    
    def get_bad_time(self, idx):
        return self.bad_time[idx]
    
    def get_id(self, idx):
        return self.id[idx]
    
    def get_channel(self, idx):
        return self.channel[idx]
    
    
    def set_spectrogram(self, idx, spectrogram):  
        self.spectrograms[idx] = spectrogram
    
    def set_raw(self, idx, raw):
        self.raw[idx] = raw


    def select_channels(self, ch):
        """
        A function which returns an EEGDataset object with only the selected channels
        """
        # get indices of the channels to be selected
        indices = [i for i, channel in enumerate(self.channel) if channel == ch]
        # select only the desired channels
        spectrograms = [self.spectrograms[i] for i in indices] if len(self.spectrograms) > 0 else []
        raw = [self.raw[i] for i in indices]
        bad_time = [self.bad_time[i] for i in indices]
        id = [self.id[i] for i in indices]
        channel = [self.channel[i] for i in indices]

        return EEGDataset(spectrograms, raw, bad_time, id, channel, self.info)
    
    def filter_data(self, lowcut, highcut, order=2):
        """
        Filter the EEG data in a specific frequency range
        """
        raw = []
        # filter raw data
        for idx, _ in enumerate(self.raw):
            raw.append(filter_eeg_data(self.raw[idx], fs=500, lowcut=lowcut, highcut=highcut, order=order))

        return EEGDataset(self.spectrograms, raw, self.bad_time, self.id, self.channel, self.info)
        
    def remove_item(self, idx):
        del self.spectrograms[idx]
        del self.raw[idx]
        del self.bad_time[idx]
        del self.id[idx]
        del self.channel[idx]



def filter_eeg_data(data: np.array, fs: float, lowcut: float, highcut: float, order: int) -> np.array:
    """
    Filter the EEG data in a specific frequency range
    Input:
        data: raw EEG data
        fs: sampling frequency
        lowcut: lower frequency limit
        highcut: higher frequency limit
        order: order of the filter
    Output:
        filtered_eeg_data: filtered raw EEG data
    """
  
    # filter data in a specific frequency range 
    # calculate the Nyquist frequency
    nyquist = 0.5 * fs

    # design the band-pass filter using the Butterworth filter design
    b, a = scipy.signal.butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')

    # apply the filter to the EEG data
    filtered_eeg_data = scipy.signal.lfilter(b, a, data)

    return filtered_eeg_data

def get_subject_bad_time(file_path: str, start_time: float, time_range: float, time_window: float) -> list:
    """
    Reads a CSV file and returns a list of booleans indicating the presence or absence
    of an effect for each time window in the given range.

    Parameters:
        file_path (str): Path to the CSV file.
        start_time (float): The starting time of the window.
        time_range (float): The length of the range (in seconds) to evaluate the effect.
        time_window (float): The size of each time window (between 0.5 and 1.0 seconds).

    Returns:
        list: A list of booleans, where True indicates the effect is active during
              that time window, and False otherwise.
    """
    if not (0.5 <= time_window <= 1.0):
        raise ValueError("time_window must be between 0.5 and 1.0 seconds.")

    num_intervals = math.ceil(time_range / time_window)
    bad_time_list = [False] * num_intervals

    # read the onset and duration data from the CSV file
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            onset = float(row['onset'])
            duration = float(row[' duration'])  # header for duration has a space!

            # skip processing if the effect is completely outside the time window
            if onset >= (start_time + time_range) or (onset + duration) <= start_time:
                continue

            # calculate start and end indices of the effect in the windowed list
            effect_start = max(onset, start_time)
            effect_end = min(onset + duration, start_time + time_range)

            start_index = int((effect_start - start_time) / time_window)
            end_index = int(math.ceil((effect_end - start_time) / time_window))

            # update the boolean list for each affected time window
            for i in range(start_index, end_index):
                if 0 <= i < num_intervals:
                    bad_time_list[i] = True

    return bad_time_list


# TO DO: FIX THIS FUNCTION
def remove_bad_chunks(dataset: EEGDataset) -> EEGDataset:
    """
    Removes the bad chunks of time from the dataset.

    Parameters:
        dataset (EEGDataset): The dataset from which bad chunks need to be removed.

    Returns:
        EEGDataset: A new dataset without bad chunks of time.
    """
    spectrograms = []
    raw = []
    bad_time = []
    ids = []
    channels = []

    for i in range(len(dataset)):
        current_spectrogram, current_raw, current_bad_time, current_id, current_channels = (
            dataset.get_spectrogram(i),
            dataset.get_raw(i),
            dataset.get_bad_time(i),
            dataset.get_id(i),
            dataset.get_channel(i)
        )

        if not current_bad_time:
            spectrograms.append(current_spectrogram)
            raw.append(current_raw)
            bad_time.append(current_bad_time)
            ids.append(current_id)
            channels.append(current_channels)

    return EEGDataset(
        spectrograms=spectrograms,
        raw=raw,
        bad_time=bad_time,
        id=ids,
        channel=channels,
        info=dataset.__getinfo__(),
    ) 


def read_eeg_data(folder: str, data_path: str, input_channels: int, number_of_subjects: int = 31, 
                  time_window: float = 1, save_spec: bool = False, channel_list: list = None, 
                  start_time: int = 0, recording_length: int = 60, select: bool = True) -> EEGDataset:
    """
    Create an EEGDataset object using the data in the folder
    Input:
        folder: path to the folder containing the data
        data_path: path to the file where to save the dataset
        input_channels: number of input channels
        number_of_subjects: number of subjects to be loaded 
        time_window: length of the time window in seconds
        save_spec: if True, save spectrograms, so that if they are not needed, the size of the dataset is smaller
        channel: if not None, select only the specified channel
    Output: 
        dataset: an EEGDataset object
    """
    if number_of_subjects < 1 or number_of_subjects > len(SUBJECT_ID):
        raise ValueError(f"Invalid input: please select a proper number of subjects.")
    # final destination of the dataset
    elec = len(channel_list) if channel_list is not None else len(CHANNEL_NAMES)
    sel = 'clean' if select else 'not_clean'
    file_path = f'{data_path}/eeg_dataset_ns_{number_of_subjects}_nc_{str(elec)}_ch_{input_channels}_tw{str(time_window).replace(".", "")}sec_st{str(start_time)}_rl{str(recording_length)}_{sel}.pkl'

    # mkdir if the folder does not exist
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # if the file containing the dataset already exists, load dataset from there
    if os.path.exists(file_path):
        logger.info("Loading dataset...")
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset
    else:
        logger.info("Creating dataset... this may take a while")
    
    # for each edf file in folder, we save the following information
    ids = [] # subject ID
    channels = [] # channel names
    raw = [] # raw eeg data
    spectrograms = [] # corresponding spectrograms
    bad_time = [] # chunks to discard

    nchannels = len(CHANNEL_NAMES) 

    for subj in SUBJECT_ID[:number_of_subjects]:
        # format the pattern
        pattern = f"_BB{str(subj).zfill(2)}"
        matching_files = [f for f in os.listdir(folder) if pattern in f and f.endswith('.set')]    

        # check if there is a corresponding .set file
        if len(matching_files) != 1:
            print(f"Subject {subj}: No matching file or multiple files found. Skipping.")
            continue
        
        file = matching_files[0]
        csv = file.split('eeg.set')[0] + 'badTime.csv'
        logger.info(f'Processing file {file}')

        # id should be of the form BBxy
        id = file.split('_')[9]

        # read set file and change reference to average
        data = mne.io.read_raw_eeglab(folder + file, preload=True, verbose=False)
        data.set_eeg_reference(ref_channels="average", verbose=False)

        # frequency of the data
        fs = data.info['sfreq']

        factor = 1/time_window
        segment_length = int(fs/factor) 
        n_segments = int(factor*recording_length) # number of segments in the recording

        # deal with bad chunks of time
        bad_chunks = get_subject_bad_time(folder + csv, start_time=start_time, 
                                          time_range=recording_length, time_window=time_window)
        bad_chunks_subject = [element for element in bad_chunks for _ in range(nchannels)] if channel_list is None else [element for element in bad_chunks for _ in range(len(channel_list))]
        if input_channels == 1:
            bad_time.extend(bad_chunks_subject)
        else:
            bad_time.append(bad_chunks_subject)

        for j in range(n_segments):
            img_eeg = []
            raw_eeg = []
            identifiers = []

            for i in range(len(CHANNEL_NAMES)):
                if channel_list is not None and CHANNEL_NAMES[i] not in channel_list:
                    continue
                sample = data.get_data(i)[0]
                eeg_data = sample[j*segment_length:(j+1)*segment_length] 
                
                raw_eeg.append(eeg_data)

                if save_spec:
                    freqs = np.arange(0.5, 60, 2)
                    ncycles = np.linspace(.5,5,len(freqs))
                    mwt = compute_wavelet_transform(eeg_data, fs=fs, n_cycles=ncycles, freqs=freqs)
                    mwt = scipy.signal.resample(mwt, num=200, axis=1) # resample spectrogram to 200 points so that they occupy less space
                    img_eeg.append(np.abs(mwt))

                else:
                    img_eeg.append([])

                identifiers.append(f'{id}_{j}')
            
            # input channels are needed only to choose how data is saved
            # if input_channels = 1, each spectrogram is a data item
            # else a single data item is made of the spectrograms of all channels which will be concatenated 
            if input_channels == 1:
                spectrograms.extend(np.array(img_eeg))
                raw.extend(raw_eeg)
                ids.extend(identifiers)
                channels.extend(CHANNEL_NAMES) if channel_list is None else channels.extend(channel_list)

            else:
                spectrograms.append(np.array(img_eeg))
                raw.append(raw_eeg)
                ids.append(identifiers)
                channels.append(CHANNEL_NAMES) if channel_list is None else channels.append(channel_list)

    dataset = EEGDataset(spectrograms, raw, bad_time, ids, channels, {})

    with open(file_path, 'wb') as f:
        pickle.dump(dataset, f)
        
    return dataset
    



def build_dataloader(dataset: EEGDataset, batch_size: int, train_rate: float = 0.8, valid_rate: float = 0.1, shuffle: bool = True, resample: bool = False) -> tuple:
    """
    A function which provides all the dataloaders needed for training, validation and testing
    Input:
        dataset: a custom dataset
        batch_size: the batch size
        train_rate: the percentage of the dataset used for training
        valid_rate: the percentage of the dataset used for validation
        test_rate: the percentage of the dataset used for testing
        shuffle: whether to shuffle the dataset before splitting it
        resample: whether to resample the spectrograms to make them smaller
    Output:
        trainloader: a dataloader for training
        validloader: a dataloader for validation
        testloader: a dataloader for testing
    """

    # build trainloader
    train_size = int(train_rate * len(dataset))
    valid_size = int(valid_rate * len(dataset))
    test_size = len(dataset) - train_size - valid_size

    # before loading data into dataloader, normalize the data
    dataset_tmp = copy.deepcopy(dataset)

    if resample:
        # resample spectrograms
        for idx, _ in enumerate(dataset):
            # resample spectrogram
            spectrogram = dataset_tmp.get_spectrogram(idx)
            dataset_tmp.spectrograms[idx] = scipy.signal.resample(spectrogram, 100, axis=2)
            if idx == 0:
                logger.info("Shape of spectrogram after resampling: ", dataset_tmp.spectrograms[idx].shape)

    # transform data to tensors if not already
    for idx in range(len(dataset_tmp.raw)):
        dataset_tmp.spectrograms[idx] = torch.tensor(dataset_tmp.spectrograms[idx].real).float() if len(dataset_tmp.spectrograms) > 0  else dataset_tmp.spectrograms[idx]
        dataset_tmp.raw[idx] = torch.tensor(np.array(dataset_tmp.raw[idx])).float() 
        dataset_tmp.labels[idx] = torch.tensor(dataset_tmp.labels[idx]).long() 

    min_spectr = np.min([torch.min(torch.abs(dataset_tmp[i][0])) for i in range(len(dataset_tmp))])
    max_spectr = np.max([torch.max(torch.abs(dataset_tmp[i][0])) for i in range(len(dataset_tmp))])


    # normalize spectrograms
    for idx, _ in enumerate(dataset):
        spectrogram = torch.abs(dataset_tmp.spectrograms[idx])
        dataset_tmp.spectrograms[idx] = (spectrogram - min_spectr) / (max_spectr - min_spectr)
    

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset_tmp, [train_size, valid_size, test_size])
    del dataset_tmp

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    validloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle) if valid_size > 0 else None
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)  if test_size > 0 else None

    return trainloader, validloader, testloader
