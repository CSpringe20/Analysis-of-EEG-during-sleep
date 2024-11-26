import mne
'''
from utils.constants import subject_id

file = '124cha_a_i_is_a_fh1e-01_fl40_Session_20170308_BB01_eeg.set'
data = mne.io.read_raw_eeglab('../data_pp/' + file, preload=True, verbose=False)
data.set_eeg_reference(ref_channels="average", verbose=False)
print(data.info['sfreq'])

sample = data.get_data(15)[0]
print(sample.shape)
id = file.split('eeg.set')[0] + 'badTime.csv'
print(id)

from utils.read_data import get_subject_bad_time

bad = get_subject_bad_time('../data_pp/124cha_a_i_is_a_fh1e-01_fl40_Session_20170308_BB01_badTime.csv')

print(bad[0][1])

for subj in subject_id:
    print(subj, end=", ")
'''

from utils.read_data import read_eeg_data

dataset = read_eeg_data(folder='../data_pp/', data_path='./', input_channels=1, number_of_subjects=1,
                        save_spec=False,)