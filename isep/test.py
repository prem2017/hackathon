# -*- coding: utf-8 -*-

# Copyright (c) 2019 Prem, Ishwar, Arunava, Navdeep, Rohil
# Permision is hereby granted, free of charge, to any person obtaining the copy subject to
# following condition:
# The above copyright permision shall be included in all copied of its use.

import os
import pandas as pd

import torch
from torch.utils.data import DataLoader
from models import *
from data_set import *


import util
util.reset_logger('test_output.log')
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
	



def forcast_weather(model, test_fname, forcast_len=192, only_test=False):
	
	test_path = util.get_test_datapath()+test_fname
	if only_test:
		dataset = TestTSDataset(test_path)
	else:
		dataset = OberserverTSDataset(test_path)
	
	# TODO: handle location (specific to lat-lonf) wise prediction for respectiv geographical location
	dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
	
	model = {k: v.to(util.device).eval() for k, v in model.items()}
	ypred_all = []
	ytrue_all = []
	with torch.no_grad():
		for i, (x, y) in enumerate(dataloader):
			x, y = x.to(device=util.device, dtype=torch.float), y.to(device=util.device, dtype=torch.float)
			x_encoded_ouput = hidden = model['encoder_gru'](x)
			y_pred_seq = None
			# y_pred_ts = torch.zeros(x.shape[0], 1, y.shape[-1] if y != -1 else util.OUTPUT_FEATURES_NUM)
			# [Note]: Input sequence for start of prediction can be zero of last known observation i.e. last timestamp from the sequence encoder by encoder
			y_pred_ts = x[:, -1, :].view(-1, 1, x.shape[-1]) if x.shape[-1] == y.shape[-1] else torch.zeros(x.shape[0], 1, util.OUTPUT_FEATURES_NUM)
			# y_pred_ts = torch.zeros(x.shape[0], 1, util.OUTPUT_FEATURES_NUM)
			y_pred_ts = y_pred_ts.to(util.device)
			for j in range(forcast_len):
				y_pred_ts, hidden = model['decoder_gru'](y_pred_ts, hidden) # y_pred_ts.size = #batch_size * 1 * output
				y_pred_seq = torch.cat((y_pred_seq, y_pred_ts), dim=-2) if y_pred_seq is not None else y_pred_ts # contanate along seq i.e. 2nd from last
			
			ypred_all.append(y_pred_seq.cpu().numpy())
			if not only_test:
				ytrue_all.append(y.cpu().numpy())
			
		
	
	test_loc_names = dataset.get_loc_names()
	errors = {}
	output_dir = util.get_test_result_dir()
	y_columns = ['ypred_' + name for name in util.OUPUT_FEATURES_NAME_LIST]
	if not only_test:
		y_columns = ['ytrue_' + name for name in util.OUPUT_FEATURES_NAME_LIST] + y_columns

		
	for bi in range(len(ypred_all)):
		current_batch_ouput = ypred_all[bi]
		for i, output in enumerate(current_batch_ouput):
			if only_test:
				df = pd.DataFrame(np.around(output, 3), columns=y_columns)
			else:
				ytrue = ytrue_all[bi][i, :, :]
				concat_data = np.hstack((ytrue, np.around(output, 3)))
				df = pd.DataFrame(concat_data, columns=y_columns)
				
			key = str(bi) + '_' + str(i)
			path = output_dir + key + '.csv'
			df.to_csv(path, sep=',', index=None, header=y_columns)
			
			if not only_test:
				errors[key] = {}
				errors[key]['mse'] = util.Metric.mean_square_error(ytrue_all[bi][i, :, :], output)
				errors[key]['mape'] = util.Metric.mean_average_percentage_error(ytrue_all[bi][i, :, :], output)
	msg = '\n\n[Test] Errors = \n' + util.construct_dictstring(errors)
	logger.info(msg); print(msg)
			
		
		
	




if __name__ == '__main__':
	print('[Run Test]')
	
	# TODO: support for external raw-file and transformation in test data & for only test file whern there is no Y to compare
	seq_info = util.seq_info_parser()
	
	util.set_trained_model_name(**seq_info) # That is load the model trained on 1 day (seq=96) of history with 2 hour difference (advance=8) b/w data point and forcast for next 12 hours (forcast=48)
	model_fname = util.get_trained_model_name()
	model = load_trained_model(model_fname)
	
	test_fname = util.get_hdf5_file(type='test', **seq_info)
	
	forcast_weather(model=model,
	                test_fname=test_fname,
	                forcast_len=seq_info['forcast_len'],
	                only_test=False)
	
	
	
	
	
	
	
	
	
	
	
	