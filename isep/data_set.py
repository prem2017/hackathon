# -*- coding: utf-8 -*-

# Copyright (c) 2019 Prem, Ishwar, Arunava, Navdeep, Rohil
# Permision is hereby granted, free of charge, to any person obtaining the copy subject to
# following condition:
# The above copyright permision shall be included in all copied of its use.

import h5py
import torch
import numpy as np


from torch.utils.data import Dataset



class OberserverTSDataset(Dataset):
	"""
	
	"""

	def __init__(self, hdf5_filepath):
		super(OberserverTSDataset, self).__init__()
		
		self.hdf5_filepath = hdf5_filepath
		
		self.h5py_ref =  h5py.File(self.hdf5_filepath, 'r')
		self.data_len = self.h5py_ref.attrs['len']
		
	
	# TODO: perform hdf5 open and close each time if that is efficient
	def __getitem__(self, index):
	
		return torch.tensor(self.h5py_ref['X'][index, :]), torch.tensor(self.h5py_ref['Y'][index, :])
	
	def __len__(self):
		return self.data_len
	
	
	def close(self):
		self.h5py_ref.close()


class TestTSDataset(Dataset):
	"""

	"""
	
	def __init__(self, hdf5_filepath):
		super(TestTSDataset, self).__init__()
		
		self.hdf5_filepath = hdf5_filepath
		
		self.h5py_ref = h5py.File(self.hdf5_filepath, 'r')
		self.data_len = self.h5py_ref.attrs['len']
	
	# TODO: perform hdf5 open and close each time if that is efficient
	def __getitem__(self, index):
		return torch.tensor(self.h5py_ref['X'][index, :]), torch.tensor(-1)
	
	def __len__(self):
		return self.data_len
	
	def close(self):
		self.h5py_ref.close()
