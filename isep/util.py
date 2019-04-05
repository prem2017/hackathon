# -*- coding: utf-8 -*-

# Copyright (c) 2019 Prem, Ishwar, Arunava, Navdeep, Rohil
# Permision is hereby granted, free of charge, to any person obtaining the copy subject to
# following condition:
# The above copyright permision shall be included in all copied of its use.

import os
import numpy as np
import pickle
import argparse

import pandas as pd
import torch


OUTPUT_FEATURES_NUM = 3

TRAIN_BATCH_SIZE = 8
VALIDATION_BATCH_SIZE = 1
import logging
logging.basicConfig(level=logging.INFO, format='%(messages)s')
logger = logging.getLogger()



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Datapath
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_PATH = os.path.join(PROJECT_PATH, 'data/')

def get_data_path(file_name):
	return os.path.join(BASE_DATA_PATH, file_name)
	
RESULT_DIR = os.path.join(PROJECT_PATH, 'results_n_plots/')
get_result_dir = lambda : RESULT_DIR

# Store trained model
TRAINED_MODELPATH = os.path.join(PROJECT_PATH, 'models/')
get_trained_model_path = lambda : TRAINED_MODELPATH


TRAINED_MODELNAME = 'rnn_weather_net.model'
def set_trained_model_name(seq_len=192, forcast_len=48, advance_len=16):
	global TRAINED_MODELNAME
	TRAINED_MODELNAME = "rnn_weather_net_seq_{}_forcast_{}_advance_{}.model".format(seq_len, forcast_len, advance_len)
	return TRAINED_MODELNAME

get_trained_model_name = lambda : TRAINED_MODELNAME




def get_obs_datapath(dirname='observations/'):
	return os.path.join(BASE_DATA_PATH, dirname)



def get_hdf5_file(type, seq_len, forcast_len, advance_len):
	return "{}_seq_{}_forcast_{}_advance_{}.hdf5".format(type, seq_len, forcast_len, advance_len)



def reset_logger(filename='train_output.log'):
	logger.handlers = []
	filepath = os.path.join(RESULT_DIR, filename)
	logger.addHandler(logging.FileHandler(filepath, 'w'))


def add_logger(filename):
	filepath = os.path.join(RESULT_DIR, filename)
	logger.addHandler(logging.FileHandler(filepath, 'w'))


def setup_logger(filename='output.log'):
	filepath = os.path.join(RESULT_DIR, filename)
	logger.addHandler(logging.FileHandler(filepath, 'a'))


class Metric(object):
	"""docstring for Metric
		Different metric for measuring performance
	"""
	def __init__(self):
		super(Metric, self).__init__()

	@staticmethod # MAPE
	def mean_average_percentage_error(y_true: np.ndarray, y_forcast: np.ndarray) -> float:
		y_true = y_true.reshape(-1)
		y_true += 1e-9 # TODO: hack for avoiding division by zero
		y_forcast = y_forcast.reshape(-1)
		
		return float(np.mean(np.abs((y_true - y_forcast) / y_true)))

	@staticmethod # TODO: https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
	def mean_absolute_scaled_err(y_true: np.ndarray, y_forcast: np.ndarray) -> float:
		
		
		
		return None


# parser for file
def seq_info_parser(seq_len=192, forcast_len=48, advance_len=16):
	parser = argparse.ArgumentParser(
		description='Test parameters such seqence-length, forcast-length, advance-length (should be optinal eventually) etc.')
	parser.add_argument('-s', '--seq_len', type=int, default=seq_len,
	                    help='Lenght of day which is used from past timestamps for prediction. For example 1 day = 24*4 (=96), 2 days = 48*4 (=192) ',
	                    required=False)
	parser.add_argument('-f', '--forcast_len', type=int, default=forcast_len,
	                    help='Length of day for which prediction is performed 1 day = 24*4 (=94)', required=False)
	parser.add_argument('-a', '--advance_len', type=int, default=advance_len,
	                    help='Advance timestamp by a length for next sequence formation for example two-hours = 2*4 (= 8), eight-hours = 5*4 (=32)',
	                    required=False)
	args = parser.parse_args()
	
	seq_info = {}
	seq_info['seq_len'] = args.seq_len
	seq_info['forcast_len'] = args.forcast_len
	seq_info['advance_len'] = args.advance_len
	
	return seq_info


if __name__ == '__main__':
	print('[Util] module')