# -*- coding:utf-8 -*-

# Copyright (c) 2019 Prem, Ishwar, Arunava, Navdeep, Rohil
# Permision is hereby granted, free of charge, to any person obtaining the copy subject to
# following condition:
# The above copyright permision shall be included in all copied of its use.

import os
import pandas as pd
import numpy as np
import h5py



import torch


import util

obv_fnames = [f for f in os.listdir(util.get_obs_datapath()) if '.csv' in f.lower()]


def tranform_observed_data(seq_len, forcast_len, advance_len=None, val=0.1, test=0.1):
	"""
	:param seq_len:
	:param forcast_len:
	:param advance_len:
	:return: saves the transformed data
	"""
	if advance_len is None:
		advance_len = seq_len
	
	X = None
	Y = None
	
	train_fname = util.get_hdf5_file('train', seq_len, forcast_len, advance_len)
	val_fname = util.get_hdf5_file('val', seq_len, forcast_len, advance_len)
	test_fname = util.get_hdf5_file('test', seq_len, forcast_len, advance_len)
	
	for fname in obv_fnames:
		df = pd.read_csv(os.path.join(util.get_obs_datapath(), fname))
		df.sort_values(by=['Timestamp'])
		df = df.values[:, -3:]
		counter = 0
		while (counter + seq_len + forcast_len) < df.shape[0]:
			xi = df[counter: counter+seq_len, :]
			xi = xi.reshape(1, -1, 3)
			yi = df[counter+seq_len: counter+seq_len+forcast_len, :]
			yi = yi.reshape(1, -1, 3)
			if X is None:
				X = xi
				Y = yi
			else:
				X = np.vstack((X, xi))
				Y = np.vstack((Y, yi))
			counter += advance_len
	permutations = np.random.permutation(X.shape[0])
	X = X[permutations]
	Y = Y[permutations]
		
	train_size = int(X.shape[0] * 0.8)
	val_size = (int(X.shape[0] * 0.1) + 1)
	test_size = X.shape[0] - train_size - val_size
	
	def save_in_hdf5(fname_hdf5, Xs, Ys):
		with h5py.File(os.path.join(util.get_obs_datapath(), fname_hdf5), 'w') as f:
			f.attrs['seq_len'] = seq_len
			f.attrs['forcast_len'] = forcast_len
			f.attrs['advance_len'] = advance_len
			f.attrs['len'] = Xs.shape[0]
			f.create_dataset('X', data=Xs)
			f.create_dataset('Y', data=Ys)
		
	for fname_hdf5, start_pos, size_len in zip([train_fname, val_fname, test_fname], [0, train_size, train_size + val_size], [train_size, val_size, test_size]):
		print('\nsaved for = ', fname_hdf5)
		save_in_hdf5(fname_hdf5, np.array(X[start_pos:start_pos+size_len, :, :]).astype(np.float64), np.array(Y[start_pos: start_pos+size_len, :, :]).astype(np.float64))
		


def transform_test_data():
	pass








if __name__ == '__main__':
	print('[DataLoader] Module')
	
	# tranform_observed_data(256, 64, 128)
	tranform_observed_data(192, 48, 16)
	# tranform_observed_data(96, 48, 8)
	
	# TODO: normalize data and then store train, val, test
	
	
	
	
	
	
	
	
	
	
	
	