# -*- coding:utf-8 -*-

# Copyright (c) 2019 Prem, Ishwar, Arunava, Navdeep, Rohil
# Permision is hereby granted, free of charge, to any person obtaining the copy subject to
# following condition:
# The above copyright permision shall be included in all copied of its use.

import os
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from loss_functions import *
from models import *
from data_set import *

import util


logger = util.logger




# Train the network
def train_network(dataloader, model, loss_function, optimizer, start_lr, end_lr, num_epochs=90):
	model = {k: v.train() for k, v in model.items()}
	logger_msg = '\nDataLoader = %s' \
	             '\nModel = %s' \
	             '\nLossFucntion = %s' \
	             '\nOptimizer = %s' \
	             '\nStartLR = %s, EndLR = %s' \
	             '\nNumEpochs = %s' % (dataloader, model, loss_function, optimizer, start_lr, end_lr, num_epochs)

	logger.info(logger_msg), print(logger_msg)

	# [https://arxiv.org/abs/1803.09820]
	# This is used to find optimal learning-rate which can be used in one-cycle training policy
	# [LR]TODO: for finding optimal learning rate
	# lr_scheduler = [MultiStepLR(optimizer=opt,
	#                            milestones=list(np.arange(2, 24, 2)),
	#                            gamma=10, last_epoch=-1) for k, opt in optimizer.items()]

	def get_lr():
		lr = []
		for _, opt in optimizer.items():
			for param_group in opt.param_groups:
				lr.append(np.round(param_group['lr'], 11))
		return lr

	def set_lr(lr):
		for _, opt in optimizer.items():
			for param_group in opt.param_groups:
				param_group['lr'] = lr

	total_len = 0
	current_epoch_batchwise_loss = []
	avg_epoch_loss_container = []  # Stores loss for each epoch averged over
	avg_val_loss_container = []
	mape_val_container = []

	all_epoch_batchwise_loss = []

	extra_epochs = 20
	total_epochs = num_epochs + extra_epochs

	# One cycle setting of Learning Rate
	num_steps_upndown = 20
	further_lowering_factor = 10
	further_lowering_factor_step = 4

	def one_cycle_lr_setter(current_epoch):
		if current_epoch <= num_epochs:
			lr_inc_rate = (end_lr - start_lr) / (2 * num_steps_upndown)
			lr_inc_epoch_step_len = int(num_epochs / (2 * num_steps_upndown))

			steps_completed = int(current_epoch / lr_inc_epoch_step_len)
			if steps_completed < num_steps_upndown:
				current_lr = start_lr - (steps_completed * lr_inc_rate)
			else:
				current_lr = end_lr + ((steps_completed - num_steps_upndown) * lr_inc_rate)
			set_lr(current_lr)
		else:
			current_lr = end_lr / (
						further_lowering_factor ** (((current_epoch - num_epochs) / further_lowering_factor_step) + 1))
			set_lr(current_lr)

	for epoch in range(total_epochs):
		msg = '\n\n\n[Epoch] = %s' % (epoch + 1)
		print(msg)
		for i, (x, y) in enumerate(dataloader['train']):
			loss = 0
			x, y = x.to(device=util.device, dtype=torch.float), y.to(device=util.device, dtype=torch.float)
			
			# TODO: early breaker
			# if i == 2:
			# 	print('[Break] by force for validation check')
			# 	break

			for _, opt in optimizer.items():
				opt.zero_grad()
			
			y_encoded_hidden = model['encoder_gru'](x) # initial_states[-1, :, :] i.e shape => [-1 (#layers) x batch-size x hidden-size]
			output_seq_len = model['encoder_gru'].get_output_seq_len()
			
			y_decoded_tstamp = torch.zeros(x.shape[0], 1, y.shape[-1]) # shape => [batch-size x seq-size (here is 11) x feature-size]
			for j in range(output_seq_len):
				y_decoded_tstamp, y_encoded_hidden  = model['decoder_gru'](y_decoded_tstamp, y_encoded_hidden)
				loss += loss_function(y[:, j:j+1, :].contiguous().view(-1), y_decoded_tstamp.view(-1))
			
		
			loss.backward()
			for _, opt in optimizer.items():
				opt.step()

			current_epoch_batchwise_loss.append(loss.item() / output_seq_len)
			all_epoch_batchwise_loss.append(loss.item()/output_seq_len)

			batch_run_msg = '\nEpoch: [%s/%s], Step: [%s/%s], InitialLR: %s, CurrentLR: %s, Loss: %s' \
			                % (epoch + 1, total_epochs, i + 1, len(dataloader['train']), start_lr, get_lr(), loss.item()/output_seq_len)
			print(batch_run_msg)
		# store average loss
		avg_epoch_loss = np.round(sum(current_epoch_batchwise_loss) / (i + 1.0), 6)
		current_epoch_batchwise_loss = []
		avg_epoch_loss_container.append(avg_epoch_loss)
		
		# [LR]TODO: validation loss
		val_loss, val_mape = calc_validation_loss(model, dataloader['validation'], loss_function)
		# TODO: save model for minimum val_loss and val_mape
		
		# [LR]TODO:
		avg_val_loss_container.append(val_loss)
		mape_val_container.append(val_mape)
		
		
		# Logger msg
		msg = '\n\n\n\n\nEpoch: [%s/%s], InitialLR: %s, CurrentLR= %s \n' \
		      '\n[Train] Average Epoch Loss = %s\n' \
		      '\n[Validation] Epoch wise loss = %s\n' \
		      '\n[Validation] Mean Average Percentage Error (MAPE) = %s\n'\
		      %(epoch+1, total_epochs, start_lr, get_lr(), avg_epoch_loss_container, avg_val_loss_container, mape_val_container)
		logger.info(msg); print(msg)
		
		
		
		if avg_epoch_loss < 1e-6 or get_lr()[0] < 1e-9 or get_lr()[0] >= 10:
			msg = '\n\nAvg. Loss = {} or Current LR = {} thus stopping training'.format(avg_epoch_loss, get_lr())
			logger.info(msg)
			print(msg)
			break
			
		
		# [LR]TODO:
		# for lr_s in lr_scheduler:
		# 	lr_s.step(epoch+1) # TODO: Only for estimating good learning rate
		one_cycle_lr_setter(epoch + 1)

	# Print the loss
	msg = '\n\n[Epoch Loss] = {}'.format(avg_epoch_loss_container)
	logger.info(msg)
	print(msg)

	
	# [LR]TODO: change for lr finder
	plot_loss({'train': avg_epoch_loss_container, 'val': avg_val_loss_container},
	          plot_file_name='training_vs_val_avg_epoch_loss.png',
	          title='Training vs Validation Epoch Loss')
	plot_loss(all_epoch_batchwise_loss, plot_file_name='training_batchwise.png', title='Training Batchwise Loss',
	          xlabel='#Batchwise')
	
	# [LR]Mape loss
	plot_loss(mape_val_container,
	          plot_file_name='validation_mape_epoch.png',
	          title='Validation MAPE Epoch wise')

	# Save the model
	save_model(model)
	

# TODO: validation loss calculation
def calc_validation_loss(model, val_dataloader, loss_func):
	""" Computed validation loss on developement data also called validation data
	:param model:
	:param val_dataloader:
	:param loss_func:
	:return: (Validation_Loss, MAPE(Mean_Average_Percentage_Error)
	"""
	
	model = {k: v.eval() for k, v in model.items()}
	
	loss_val = 0
	y_true = None
	y_forcast = None
	for i, (x, y) in enumerate(val_dataloader):
		loss_val = 0
		
		x = x.to(device=util.device, dtype=torch.float)
		y = y.to(device=util.device, dtype=torch.float)
		# print('[Val comp] i = ', i)
		
		y_encoded_hidden = model['encoder_gru'](x)
		output_length = model['encoder_gru'].get_output_seq_len()
		
		y_decoded_tstamp = torch.zeros(x.shape[0], 1, y.shape[-1])
		loss = 0
		with torch.no_grad():
			for j in range(output_length):
				y_decoded_tstamp, y_encoded_hidden = model['decoder_gru'](y_decoded_tstamp, y_encoded_hidden)
				loss += loss_func(y[:, j, :].contiguous().view(-1), y_decoded_tstamp.view(-1))
				y_true = np.vstack((y_true, y[:, j:j+1, :].view(-1, 1).numpy())) if y_true is not None else y[:, j, :].view(-1, 1).numpy()
				y_forcast =  np.vstack((y_forcast, y_decoded_tstamp.view(-1, 1).numpy())) if y_forcast is not None else y_decoded_tstamp.view(-1, 1).numpy()
		
		loss_val += (loss.item() / output_length)
		
	mape_score = util.Metric.mean_average_percentage_error(y_true, y_forcast)
	model = {v.train() for k, v in model.items()}
	return np.round(loss_val / i+1.0, 6), np.round(mape_score, 6)


# Plot training loss
def plot_loss(losses, plot_file_name='training_loss.png', title='Training Loss', xlabel='Epochs'):
	fig = plt.figure()

	if isinstance(losses, dict):
		for k, v in losses.items():
			plt.plot(range(1, len(v)+1), v, '-*', markersize=6, lw=2, alpha=0.6)
	else:
		plt.plot(range(1, len(losses)+1), losses, '-*', markersize=6, lw=2, alpha=0.6)
	
	
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel('Cross Entropy Loss')
	full_path = os.path.join(util.RESULT_DIR, plot_file_name)
	fig.tight_layout()  # https://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots-in-matplotlib
	fig.savefig(full_path)
	plt.close(fig)  # clo


def save_model(models_dict):
	model_path = os.path.join(util.TRAINED_MODELPATH, util.get_trained_model_name())
	logger.info(model_path)
	save_dict = {}
	for k, model in models_dict.items():
		if next(model.parameters()).is_cuda:
			model = model.cpu().float()
		save_dict[k] = model.state_dict()
		
	torch.save(save_dict, model_path)
	
	models_dict = {k: v.to(util.device) for k, v in models_dict.items()}
	return models_dict


# Pre-requisite setup for training process
def train_weather_prediction_ts(train_datapth, val_datapath):
	"""
	Setup the environment for training such as model, loss function, optimizer e.t.c.
	:return: None
	"""
	msg = '\n\n[Train] datapath = {}\n[Validation] datapath= {}\n\n'.format(train_datapth, val_datapath)
	logger.info(msg), print(msg)
	
	train_params = {}
	# [LR]
	train_params['start_lr'] = start_lr =  7e-3  # 7e-4 # 1e-7
	train_params['end_lr'] = 11e-3  # 11e-4 # 10
	train_params['num_epochs'] = 60 # 90

	weight_decay = 1e-6
	dropout = 0.5

	model = {'encoder_gru': EncoderGRU(), 'decoder_gru': DecoderGRU()}
	
	train_params['model'] = model = {k: v.to(util.device) for k, v in model.items()}

	loss_function = MSELoss()
	train_params['loss_function'] = loss_function.to(util.device)
	
	optimizer = {'encoder_gru': optim.RMSprop(params=model['encoder_gru'].parameters(),
	                                          lr=start_lr,
	                                          alpha=0.99,
	                                          eps=1e-6,
	                                          centered=True,
	                                          weight_decay=weight_decay),
	             'decoder_gru': optim.RMSprop(params=model['decoder_gru'].parameters(),
	                                          lr=start_lr,
	                                          alpha=0.99,
	                                          eps=1e-6,
	                                          centered=True,
	                                          weight_decay=weight_decay)}
	

	train_params['optimizer'] = optimizer
	# optim.SGD(params=model.parameters(), lr=start_lr, weight_decay=weight_decay, momentum=0.9)
	
	dataset = {}
	dataset['train'] = OberserverTSDataset(hdf5_filepath=train_datapth)
	dataset['validation'] = OberserverTSDataset(hdf5_filepath=val_datapath)
	
	dataloader = {}
	dataloader['train'] = DataLoader(dataset=dataset['train'], batch_size=util.TRAIN_BATCH_SIZE, shuffle=True)
	dataloader['validation'] = DataLoader(dataset=dataset['validation'], batch_size=util.VALIDATION_BATCH_SIZE)
	train_params['dataloader'] = dataloader

	# Train the network
	train_network(**train_params)




if __name__ == '__main__':
	print('Hackathon')
	torch.manual_seed(999)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(999)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	util.reset_logger()
	
	print(os.getpid())
	
	seq_info = util.seq_info_parser()
	
	util.set_trained_model_name(**seq_info)
	train_fname = util.get_hdf5_file(type='train', **seq_info)
	val_fname = util.get_hdf5_file(type='val', **seq_info)
	
	train_weather_prediction_ts(train_datapth=util.get_obs_datapath() + train_fname, # 'train_seq_96_forcast_48_advance_8.hdf5',
	                           val_datapath=util.get_obs_datapath() + val_fname) # 'val_seq_96_forcast_48_advance_8.hdf5')
	msg = '\n\n********************** Training Complete **********************\n\n'
	logger.info(msg)
	print(msg)