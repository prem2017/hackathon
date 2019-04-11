# -*- coding:utf-8 -*-

# Copyright (c) 2019 Prem, Ishwar, Arunava, Navdeep, Rohil
# Permision is hereby granted, free of charge, to any person obtaining the copy subject to
# following condition:
# The above copyright permision shall be included in all copied of its use.

import numpy as np
import torch
import torch.nn as nn
from torch.tensor import Tensor


import util


class EncoderGRU(nn.Module):
	"""docstring for EncoderGRU
	"""
	def __init__(self, input_dimensions=3,
	             hidden_dimensions=256,
	             output_dimensions=3,
	             rnn_hidden_layers=1,
	             input_seq_len=96,
	             output_seq_len=48,
	             dropout=0.0,
	             batch_first=True,
	             bidirectional=False,
	             use_batchnorm=False):
		super(EncoderGRU, self).__init__()
		self.input_dimensions = input_dimensions
		self.hidden_dimensions = hidden_dimensions
		self.output_dimensions=output_dimensions
		
		self.gru_hidden_layers = rnn_hidden_layers
		
		self.input_seq_len = input_seq_len
		self.output_seq_len = output_seq_len
		
		self.dropout_val = dropout
		self.batch_first = batch_first
		
		self.bidirectional = bidirectional
		self.use_batchnorm = use_batchnorm

		self.batch_index = 0 if self.batch_first else 1
		
		self.num_directions = 2 if self.bidirectional else 1
		
		
		self.gru_encoder = nn.GRU(input_size=self.input_dimensions,
		                  hidden_size=self.hidden_dimensions,
		                  num_layers=self.gru_hidden_layers,
		                  batch_first=self.batch_first,
		                  dropout=self.dropout_val,
		                  bidirectional=self.bidirectional
		                  )
		
		self.init_states_encoder = None
		# TODO: https://discuss.pytorch.org/t/how-does-the-batch-normalization-work-for-sequence-data/30839
		# self.gru.permute(0, 2, 1)
		if self.use_batchnorm:
			print('Using batch normalisation')
			util.logger.info('Using batch normalisation')
			self.batch_norm1D = nn.BatchNorm1d(num_features=self.hidden_dimensions,
			                                   eps=1e-5,
			                                   momentum=0.1)
		
		
		if self.dropout_val > 0.0 and self.use_extra_dropout:
			self.dropout = nn.Dropout(p=self.dropout)
		
		
		self.fc = nn.Linear(in_features=self.hidden_dimensions, out_features=self.output_dimensions)
		self.init_states = None
	
	def get_output_seq_len(self):
		return self.output_seq_len
	
		
	def forward(self, X:Tensor, initial_states=None):
		
		#if self.init_states is None:
		self.init_states = torch.zeros(self.gru_hidden_layers * self.num_directions,
			                             X.size(self.batch_index),
			                             self.hidden_dimensions)
		
		# self.init_states = self.init_states.to(util.device)
		
		# TODO
		if X.shape[self.batch_index] != self.init_states.shape[1]:
			pass
		
		#
		output_gru, initial_states = self.gru_encoder(X, self.init_states)
		
		# TODO: if batchnorm handle differently
		# if self.use_batchnorm:
		# 	pass
		
		# TODO: if birdirectional handle differently [Note?]: This task should not need bidirectional RNN
		# Remember that initial states will be [(self.gru_hidden_layers * self.num_directions) x (X.shape[self.batch_index]) x (self.hidden_dimensions)]
		#
		# initial_states[-self.num_directions:, :, :] # output_gru[:,-1, :self.hidden_dimensions].view(1, -1, self.hidden_dimensions)
		rnn_output_of_last_seq = output_gru[:,-1:, :self.hidden_dimensions] # .view(1, -1, self.hidden_dimensions) # initial_states[-1, :, :]
		output = self.fc(rnn_output_of_last_seq)
		hidden = initial_states # assume that number of hidden layers and directions are same in decoder as well
		return output, hidden
		
		
		
		
	


class DecoderGRU(nn.Module):
	"""docstring for DecoderGRU

	"""
	def __init__(self, output_dimensions=util.OUTPUT_FEATURES_NUM, hidden_dimension=256, bidirectional=False, batch_first=True, dropout=0.0):
		super(DecoderGRU, self).__init__()
		self.output_dimensions = output_dimensions
		self.hidden_dimensions = hidden_dimension
		self.bidirectional= bidirectional
		self.batch_first = batch_first
		
		self.num_directions = 2 if self.bidirectional else 1
		# Decoder
		self.decoder_in_features = self.num_directions * self.hidden_dimensions
		
		self.gru_decoder = nn.GRU(input_size=self.output_dimensions,
		                          hidden_size=self.decoder_in_features,
		                          batch_first=self.batch_first)
		self.dropout = nn.Dropout(p=dropout)
		self.fc1 = nn.Linear(in_features=self.decoder_in_features,
		                    out_features=self.decoder_in_features // 2)
		self.leaky_rely = nn.LeakyReLU()
		self.fc2 = nn.Linear(in_features=self.decoder_in_features // 2, out_features=self.output_dimensions)
		self.init_states = None
		
	
	def forward(self, Xp, hidden_states=None):
		
		if hidden_states is None:
			hidden_states = self.zeros(1, 1, self.decoder_in_features)
		
		# No need to detach as BPTT should propagate throuh all the sequence
		# hidden_states = hidden_states.detach()
		output, hidden_states = self.gru_decoder(Xp, hidden_states)

		output = self.fc1(output)
		output = self.leaky_rely(output)
		output = self.fc2(output)
		return output, hidden_states
		
	
