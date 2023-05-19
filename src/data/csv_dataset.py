
from torch.utils.data import Dataset
import pandas as pd
import os
import warnings

class CSVDataset(Dataset):
    """Dataset class for a .csv containing paths to files and their labels
    
    Arguments:
    csv_file -- path of the .csv 
    fn_col -- name of column with the filenames
    lbl_col -- name of column with the class labels
    transform -- transform to apply
    """

    def __init__(self, csv_file, fn_col = None, lbl_col = None, transform = None): 
        self.csv_file = csv_file
        df = pd.read_csv(csv_file)
        self.fn_col = fn_col if fn_col != None else df.columns[0]
        self.lbl_col = lbl_col if lbl_col != None else df.columns[1]
        # check if all samples exists
        file_exists = df[self.fn_col].apply(lambda x: os.path.isfile(x))
        self.df = df[file_exists]
        missing_files = df[-file_exists]
        for _, row in missing_files.iterrows():
            sample = row[self.fn_col]
            warnings.warn('Waning: missing sample {}'.format(sample))
        self.samples = self.df[self.fn_col].values
        self.targets = self.df[self.lbl_col].values
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.transform != None:
            sample = self.transform(sample)
        lbl = self.targets[idx]
        out_tuple = (sample, lbl)
        return out_tuple
    