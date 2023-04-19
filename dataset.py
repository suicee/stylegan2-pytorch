from io import BytesIO

# import lmdb
from PIL import Image
from torch.utils.data import Dataset
from natsort import natsorted
import logging
import torch
import re
import os
import numpy as np
from glob import glob

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img

import h5py

class NaiveDataset(Dataset):
    def __init__(self, data_dir,norm=False):
      
        with h5py.File(data_dir, 'r') as F:
            images = np.array(F['images'])
            labels = np.array(F['ans'])

        total_data=np.transpose(images[:10000],axes=(0,3,1,2))
            
        logging.info(f'Creating dataset with {total_data.shape[0]} examples')

        if norm:
            self.total_data=self.preprocess(total_data,tp='data',mean=self.data_mean,std=self.data_std)
        else:
            self.total_data=total_data

    def __len__(self):
        return (self.total_data.shape[0])


    @classmethod
    def preprocess(cls,data):

        return data

    def __getitem__(self, i):
        data=self.total_data[i]

        return {
            'data': torch.from_numpy(data).type(torch.FloatTensor),
        }