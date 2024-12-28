from utils.read_data import read_eeg_data
from utils.plot_functions import correlation_connectivity_graphs
import numpy as np
import pickle
'''
folder1 = '../data_pp/'
folder2 = './saved_datasets/'
dataset = read_eeg_data(folder1, folder2, 1, 
                            number_of_subjects=31, time_window=1, 
                            save_spec=False, start_time=0, recording_length=60)
list_of_falses = [x == False for x in dataset.bad_time[:7440]]
print(np.sum(list_of_falses))

'''

folder3 = './results_correlation/'
str_band = 'alpha_band'
rmv_bad = 'clean'
table_path  =  f'{folder3}{str_band}/{rmv_bad}/correlation_table.pkl' 

DATA_FOLDER = '../data_pp/' 
file_path = '124cha_a_i_is_a_fh1e-01_fl40_Session_20170308_BB01_eeg.set'
set_file = DATA_FOLDER+file_path
'''
# open file and read tables
with open(table_path, 'rb') as f:
    info = pickle.load(f)

table_cor_idx, pvalues, table_pearson = info['cor_idx'], info['pvalue'], info['pearson']

mask = pvalues < 0.050
new_table = np.where(mask, table_cor_idx, 0)

'''

#electrode_dicts = {ch_name: montage_pos[ch_name] for ch_name in raw.info['ch_names']}

correlation_connectivity_graphs(correlation_folder=folder3, set_file=set_file, 
                                type='pearson', band=str_band, rmv_bad=rmv_bad, 
                                pvalue_th=0.05, cor_idx_th=0.8, pears_th=0.8)

