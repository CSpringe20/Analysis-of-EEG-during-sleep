from utils.correlation import plot_correlation_table
import pickle


dir_path = './results_correlation/alpha_band/clean/correlation_table.pkl'
with open(dir_path, 'rb') as f:
    my_dict = pickle.load(f)
plot_correlation_table(dir_path,my_dict,False,'alpha band','cor_idx')