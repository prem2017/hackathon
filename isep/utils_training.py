# -*- coding: utf-8 -*-


# Copyright (c) 2019 Prem, Ishwar, Arunava, Navdeep, Rohil
# Permision is hereby granted, free of charge, to any person obtaining the copy subject to
# following condition:
# The above copyright permision shall be included in all copied of its use.

import torch
import  torch.optim as optim






class OptimizerUtils(object):
	"""docstring for Optimizer"""
	def __init__(self):
		super(OptimizerUtils, self).__init__()
		
		
	@staticmethod
	def rmsprop_optimizer(params, lr=1e-3, weight_decay=1e-6):
		return optim.RMSprop(params=params, lr=lr, alpha=0.99, eps=1e-6, centered=True, weight_decay=weight_decay)


	@staticmethod
	def adam_optimizer(params, lr=1e-3, weight_decay=1e-6):
		return optim.Adam(params=params, lr=lr, weight_decay=weight_decay)

	@staticmethod
	def sgd_optimizer(params, lr=1e-6, weight_decay=1e-6, momentum=0.9):
		return optim.SGD(params=params, lr=lr, weight_decay=weight_decay, momentum=momentum)


