# -*- coding:utf-8 -*-

# Copyright (c) 2019 Prem, Ishwar, Arunava, Navdeep, Rohil
# Permision is hereby granted, free of charge, to any person obtaining the copy subject to
# following condition:
# The above copyright permision shall be included in all copied of its use.

import torch.nn as nn
import math


class MSELoss(nn.Module):
	"""docstring for MSELoss"""
	def __init__(self, average=True):
		super(MSELoss, self).__init__()
		self.average = average
		self.loss_func = nn.MSELoss(size_average=average)
		
	
	def forward(self, output, y_target):
		loss = self.loss_func(output, y_target)
		
		if math.isnan(loss.item()):
			print('[Loss] = ', loss.item())
			print('[output] =', output)
			print('[Target] = ', y_target)
		return loss
