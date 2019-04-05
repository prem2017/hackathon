# -*- coding: utf-8 -*-

# Copyright (c) 2019 Prem, Ishwar, Arunava, Navdeep, Rohil
# Permision is hereby granted, free of charge, to any person obtaining the copy subject to
# following condition:
# The above copyright permision shall be included in all copied of its use.

import os
import argparse

import torch
from torch.utils.data import DataLoader
from models import *
from data_set import *


import util
logger = util.logger


def load_trained_model(model_fname):
	model_path = util.get_trained_model_path() + model_fname
	
	model = {}
	
	model['encoder_gru'] = EncoderGRU()
	model['decoder_gru'] = DecoderGRU()
	
	saved_state_dict = torch.load(model_path, map_location= lambda storage, loc: storage)
	encoder_gru = saved_state_dict['encoder_gru']
	decoder_gru = saved_state_dict['decoder_gru']
	
	model['encoder_gru'].load_state_dict(encoder_gru)
	model['decoder_gru'].load_state_dict(decoder_gru)
	
	model = {k: v.eval() for k, v in model.items()}
	
	return model
	



def forcast_weather(model, test_fname, output_seq_len=192, only_test=False):
	
	if only_test:
		dataset = TestTSDataset(test_fname)
	else:
		dataset = OberserverTSDataset(test_fname)
	
	# TODO: handle location (specific to lat-lonf) wise prediction for respectiv geographical location
	dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
	
	model = {k: v.to(util.device) for k, v in model.items()}
	with torch.no_grad:
		for i, (x, y) in enumerate(dataloader):
			x, y = x.to(device=util.device, dtype=torch.float), y.to(device=util.device, dtype=torch.float)
			x_encoded = model['encoder_gru'](x)
			for j in range(output_seq_len):
				pass

	




if __name__ == '__main__':
	print('[Run Test]')
	util.reset_logger('test_output.log')
	# TODO: support for external raw-file and transformation in test data
	
	seq_info = util.seq_info_parser()
	
	
	
	util.set_trained_model_name(**seq_info) # That is load the model trained on 1 day (seq=96) of history with 2 hour difference (advance=8) b/w data point and forcast for next 12 hours (forcast=48)
	model_fname = util.get_trained_model_name()
	model = load_trained_model(model_fname)
	
	test_fname = util.get_hdf5_file(type='train', **seq_info)