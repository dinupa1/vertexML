"""
Base class for processing .csv files containing raw data

This model is originally developed in particleflow repository.
Original source: https://github.com/jpata/particleflow/blob/b1cb537b4e89b82048c73a42c750b4c6f4ae1990/mlpf/pyg/PFGraphDataset.py

Following changes were made to original code:
    (a). Use pandas data frame to process raw data

dinupa3@gmail.com
08-26-2022
"""

import os.path as osp
from glob import glob

import pandas as pd
import numpy as np

import torch
from torch import Tensor

import torch_geometric
import torch_geometric.utils
from torch_geometric.data import Data, Dataset, Batch, download_url

import multiprocessing

LABELS = {
    'X1': ['q1', 'q1'],
    'X2': ['x1', 'x2'],
    'X3': ['y1', 'y2'],
    'X4': ['z1', 'z2'],
    'X5': ['px1', 'px2'],
    'X6': ['py1', 'py2'],
    'X7': ['pz1', 'pz2'],
    'Y': ['q2', 'vtx', 'vty', 'vtz', 'vpx', 'vpy', 'vpz']
}


def process_func(args):
    self, fns, idx_file = args
    return self.process_multiple_files(fns, idx_file)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class ProcessedData(Dataset):
    """
    Original source: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html

    Load events from .csv file and generate a pytorch geomatric Data() object and store them in .pt format

    Args:
        raw_file_name: a .csv file

    Returns:
        batched_data: a list of Data() objects of the from;
        Data(x=[#detectors, 7], y=[1, 7])
    """
    def __init__(self, root, transform=None, pre_transform=None):
        super(ProcessedData, self).__init__(root, transform, pre_transform)

        self._processed_dir = Dataset.processed_dir.fget(self)

    @property
    def raw_file_names(self):
        raw_list = glob(osp.join(self.raw_dir, '*.csv'))
        print("num. of raw files = {}".format(len(raw_list)))
        # print(raw_list)
        return sorted([l.replace(self.processed_dir, '') for l in raw_list])

    def _download(self):
        pass

    def _process(self):
        pass

    @property
    def processed_dir(self):
        return self._processed_dir

    @property
    def processed_file_names(self):
        proc_list = glob(osp.join(self.processed_dir, '*.pt'))
        return sorted([l.replace(self.processed_dir, '.') for l in proc_list])

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        pass

    def process_single_file(self, raw_file_name):
        df = pd.read_csv(raw_file_name)

        batched_data = []
        for i in range(df.shape[0]):
            # print(i)
            x = np.array([df[LABELS['X1']].values[i], df[LABELS['X2']].values[i], df[LABELS['X3']].values[i],
                          df[LABELS['X4']].values[i], df[LABELS['X5']].values[i], df[LABELS['X6']].values[i],
                          df[LABELS['X7']].values[i]])
            y = np.array(df[LABELS['Y']].values[i])
            x = torch.tensor(x, dtype=torch.float)
            # print(x.size())
            y = torch.tensor(y, dtype=torch.float)

            d = Data(x=x, y=y)
            # print(d)

            batched_data.append(d)

        return batched_data

    def process_multiple_files(self, file_names, idx_file):
        datas = [self.process_single_file(fn) for fn in file_names]
        datas = sum(datas, [])
        # print(datas)
        p = osp.join(self.processed_dir, 'data_{}.pt'.format(idx_file))
        print(p)
        torch.save(datas, p)

    def process(self, num_files_to_batch):
        idx_file = 0
        for fns in chunks(self.raw_file_names, num_files_to_batch):
            self.process_multiple_files(fns, idx_file)
            idx_file += 1

    def process_parallel(self, num_files_to_batch, num_proc):
        pars = []
        idx_file = 0
        for fns in chunks(self.raw_file_names, num_files_to_batch):
            pars += [(self, fns, idx_file)]
            idx_file += 1
        pool = multiprocessing.Pool(num_proc)
        pool.map(process_func, pars)

    def get(self, idx):
        p = osp.join(self.processed_dir, 'data_{}.pt'.format(idx))
        data = torch.load(p, map_location='cpu')
        return data

    def __getitem__(self, idx):
        return self.get(idx)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Input data path")
    parser.add_argument("--processed_dir", type=str, help="processed", required=False, default=None)
    # parser.add_argument("--raw_file_name", typr=str, required=True, help="Raw file name")
    parser.add_argument("--num-files-merge", type=int, default=10, help="number of files to merge")
    parser.add_argument("--num-proc", type=int, default=24, help="number of processes")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    dataset = ProcessedData(root=args.dataset)

    if args.processed_dir:
        dataset._processed_dir = args.processed_dir

    dataset.process_parallel(args.num_files_merge, args.num_proc)
    # dataset.process_single_file(args.raw_file_name)