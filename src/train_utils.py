import math
import os
import pickle
import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
import datalib.constants as constants

def str2bool(v):
	return v.lower() in ('true')
def str2intlist(v):
	return list(map(int, v.split(',')))
def str2floatlist(v):
	return list(map(float, v.split(',')))
def str2floattuple(v):
	return tuple(map(float, v.split(',')))

# def norm(x):
# 	return (x - x.mean()) / (x.std() + 1e-10)
def makedirs(path):
	if not os.path.exists(path):
		os.makedirs(path)

def check_values(m, name, error_trigger=True):
	num_inf = torch.isinf(m).sum().item()
	num_nan = torch.isnan(m).sum().item()
	if num_inf > 0 or num_nan > 0 or (not error_trigger):
		print('{} has {} inf values, {} nan values'.format(name, num_inf, num_nan))
	
def debug_time_msg(debug, last_time_stamp=None, msg=None):
	if debug:
		now = time.time()
		if last_time_stamp is not None:
			print(f'{msg} time {now - last_time_stamp :.4f} s')
		return now
	else:
		return None

def rampup(cur_step, start, end, interval, rampup_alpha = 5.0, avoid_zero = False, eps = 1e-10):
	if start is None:
		return 1.0

	if cur_step <= start:
		return eps if avoid_zero else 0.0 
	elif cur_step < end:
		p = (cur_step - start) // interval * interval / (end - start)
		if rampup_alpha is None:
			return eps if p == 0 and avoid_zero else p
		else:
			p = 1 - p
			return math.exp(- p * p * rampup_alpha)
	else:
		return 1.0
def rampdown(cur_step, start, end, interval, rampdown_alpha = 12.5, avoid_zero = False, eps = 1e-10):
	if start is None:
		return 1.0

	if cur_step <= start:
		return 1.0
	elif cur_step < end:
		p = (cur_step - start) // interval * interval / (end - start)
		if rampdown_alpha is None:
			return eps if p == 1 and avoid_zero else (1 - p) #(1 - p) if p != 1 else eps
		else:
			return math.exp(- p * p * rampdown_alpha)
	else:
		return eps if avoid_zero else 0.0 

def update_lr(optimizer, init_lr, step, interval, up_start, up_end, down_start, down_end, up_alpha, down_alpha, eps):
	
	if up_start is not None and step >= up_start and step <= up_end:

		optimizer.param_groups[0]['lr'] = init_lr * rampup(step, up_start, up_end, interval, up_alpha, True, eps)
		print('ramping up lr to {:.6f}'.format(optimizer.param_groups[0]['lr']))

	if down_start is not None and step >= down_start and step <= down_end:

		optimizer.param_groups[0]['lr'] = init_lr * rampdown(step, down_start, down_end, interval, down_alpha, True, eps)
		print('ramping down lr to {:.6f}'.format(optimizer.param_groups[0]['lr']))

# class LinearLRSchedule(object):
# 	'''
# 	Implements a linear learning rate schedule. Since learning rate schedulers in Pytorch want a
# 	mutiplicative factor we have to use a non-intuitive computation for linear annealing.
# 	This needs to be a top-level class in order to pickle it, even though a nested function would
# 	otherwise work.
# 	'''
# 	def __init__(self, initial_lr, final_lr, total_steps):
# 		''' Initialize the learning rate schedule '''
# 		self.initial_lr = initial_lr
# 		self.lr_rate = (initial_lr - final_lr) / total_steps

# 	def __call__(self, step):
# 		''' The actual learning rate schedule '''
# 		# Compute the what the previous learning rate should be
# 		prev_lr = self.initial_lr - step * self.lr_rate

# 		# Calculate the current multiplicative factor
# 		return prev_lr / (prev_lr + (step + 1) * self.lr_rate)

class WDLRSchedule(object):
	'''
	Implements a linear learning rate schedule. Since learning rate schedulers in Pytorch want a
	mutiplicative factor we have to use a non-intuitive computation for linear annealing.
	This needs to be a top-level class in order to pickle it, even though a nested function would
	otherwise work.
	'''
	def __init__(self, warmup_steps, decay_steps, decay_mode, min_factor, decay_rate):
		''' Initialize the learning rate schedule '''
		self.warmup_steps = warmup_steps
		self.decay_mode = decay_mode
		self.decay_steps = decay_steps
		# self.total_steps = total_steps
		self.min_factor = min_factor
		self.decay_rate = decay_rate

	def __call__(self, step):
		''' The actual learning rate schedule '''
		# Compute the what the previous learning rate should be
		if step < self.warmup_steps:
			return (step + 1) / self.warmup_steps
		if self.decay_steps == 0:
			return 1

		if self.decay_mode == 'linear':
			return 1 - min(step + 1 - self.warmup_steps, self.decay_steps) * (1 - self.min_factor) / self.decay_steps
		elif self.decay_mode == 'exp':
			return self.decay_rate ** min(step + 1 - self.max_steps, self.decay_steps)
		else:
			raise ValueError('unsurported learning rate decay mode {}!'.format(self.decay_mode))



class NoamLRSchedule(object):
	'''
	Implement the learning rate schedule from Attention is All You Need
	This needs to be a top-level class in order to pickle it, even though a nested function would
	otherwise work.
	'''
	def __init__(self, warmup_steps=4000):
		''' Initialize the learning rate schedule '''
		self.warmup_steps = warmup_steps

	def __call__(self, step):
		''' The actual learning rate schedule '''
		# the schedule doesn't allow for step to be zero (it's raised to the negative power),
		# but the input step is zero-based so just do a max with 1
		# step = max(1, step)
		return min((step+1) ** -0.5, (step+1) * self.warmup_steps ** -1.5)

def build_lr_scheduler(use_noam, optimizer, warmup_steps=0, decay_steps=0, decay_mode=None, min_factor=None, decay_rate=None):
	if use_noam:
		return LambdaLR(optimizer, NoamLRSchedule(warmup_steps))
	else:
		return LambdaLR(optimizer, WDLRSchedule(warmup_steps, decay_steps, decay_mode, min_factor, decay_rate))
	

def update_arg(init_value, up, step, interval, start, end, up_alpha, down_alpha, avoid_zero = False, eps = 1e-10):
	if up is None:
		return init_value
	elif up == True:
		return init_value * rampup(step, start, end, interval, up_alpha, avoid_zero, eps)
	else:
		return init_value * rampdown(step, start, end, interval, down_alpha, avoid_zero, eps)

def build_optimizer(optim_method, model, lr, momentum, weight_decay, use_transformer=False):
	if optim_method == 'adam':
		betas = (0.9, 0.98) if use_transformer else (0.9, 0.999)
		return optim.Adam(model.parameters(), lr, betas=betas, eps=1e-9, weight_decay = weight_decay)
	elif optim_method == 'rmsprop':
		return optim.RMSprop(model.parameters(), lr, weight_decay = weight_decay)
	elif optim_method == 'sgd':
		return optim.SGD(model.parameters(), lr, momentum = momentum, weight_decay = weight_decay)
	else:
		raise ValueError('unsurported optimization method!')

def frozen_model(model):
	model.requires_grad_(False)

def print_loss(loss_values):
	msg = ''
	for i, k in enumerate(loss_values.keys()):
		if k.endswith('acc'):
			msg += '{}: {:.2%} '.format(k, loss_values[k])
		else:
			msg += '{}: {:.4f} '.format(k, loss_values[k]) if loss_values[k] is not None else '{}: None '.format(k)
		if i != 0 and i % 4 == 0:
			msg += '\n'
	print(msg)

def update_accl(a, result):
	if len(a) == 0:
		for k in result:
			a[k] = result[k]
	else:
		for k in result:
			if k in a:
				a[k] += result[k]
			else:
				a[k] = result[k]
def get_final_accl(a, c):
	for k in a:
		a[k] /= c

def zero_accl(a):
	for k in a:
		a[k] = 0

def reorder(order, x):
	rx = list(range(len(x)))
	for i, a in zip(order, x):
		rx[i] = a
	return rx

def to_sentences(outputs, lens, itos, corpus, batch_first=False):
	outputs = outputs.data.cpu().numpy() if batch_first else outputs.data.t().cpu().numpy()
	lens = lens.data.cpu().numpy()
	for line, length in zip(outputs, lens):
		sen = [itos[line[i]] for i in range(length)]
		corpus.append(sen)

def to_sentence_list(outputs, lens, itos, batch_first=False):
	result = []
	outputs = outputs.data.cpu().numpy() if batch_first else outputs.data.t().cpu().numpy()
	lens = lens.data.cpu().numpy()
	for line, length in zip(outputs, lens):
		sen = [itos[line[i]] for i in range(length)]
		result.append(sen)
	return result

def select_sentence(sens1, sens2, inds):
	result=[]
	inds = inds.data.cpu().numpy()
	for s1, s2, i in zip(sens1, sens2, inds):
		result.append(s1 if i==0 else s2)
	return result

def to_details(ent_mat, probs_mat, top_inds, lens, itos, corpus):
	ent_mat = ent_mat.data.t().cpu().numpy()
	probs_mat = probs_mat.data.transpose(0, 1).cpu().numpy()
	top_inds = top_inds.data.transpose(0, 1).cpu().numpy()
	lens = lens.data.cpu().numpy()
	for ent, probs, inds, length in zip(ent_mat, probs_mat, top_inds, lens):
		s = []
		for i in range(length):
			detail = [f'{itos[ind]}({p:.4f})' for p, ind in zip(probs[i], inds[i])]
			s.append(' '.join(detail)+f' | Ent:{ent[i]:.4f}')
		corpus.append('\n'.join(s))

def append_scores(file, acc, fluency, self_bleu, human_bleu):
	with open(file, 'a') as f:
		f.write('transfer accuracy: {:.2%}\n'.format(acc))
		f.write('transfer perplexity: {:.2f}\n'.format(fluency))
		f.write('self bleu: {:.2f}\n'.format(self_bleu))
		f.write('human bleu: ' + ('{:.2f}'.format(human_bleu) if human_bleu is not None else 'NA') +'\n')

def save_outputs(outputs, start, end, file):
	with open(file, 'w') as f:
		for i in range(start, end):
			f.write(' '.join(outputs[i])+'\n')

def save_results(inputs, outputs, file):
	with open(file, 'w') as f:
		for src, tgt in zip(inputs, outputs):
			f.write(' '.join(src) + '\t' + ' '.join(tgt) + '\n')
def save_parallel_results(inputs, labels, outputs, file):
	result_list = []
	for src, l, tgt in zip(inputs, labels, outputs):
		result_list.append([' '.join(src), l, ' '.join(tgt)])
	with open(file, 'wb') as f:
		pickle.dump(result_list, f, -1)
def save_details(inputs, outputs, details, file):
	with open(file, 'w') as f:
		for src, tgt, detail in zip(inputs, outputs, details):
			f.write('Input: '+' '.join(src) + '\n')
			f.write('Output: '+' '.join(tgt) + '\n')
			f.write(detail+'\n\n')

# def save_detailed_results(inputs, outputs, pos_outputs, dep_outputs, file):
# 	with open(file, 'w') as f:
# 		for i in range(len(inputs)):
# 			f.write('Input: '+' '.join(inputs[i]) + '\n')
# 			f.write('Output: '+'\t'.join(outputs[i]) + '\n')
# 			if pos_outputs is not None:
# 				f.write('POS: '+'\t'.join(pos_outputs[i]) + '\n')
# 			if dep_outputs is not None:
# 				f.write('DEP: '+'\t'.join(dep_outputs[i]) + '\n')



def add_to_writer(loss_values, step, prefix, writer):
	for key, value in loss_values.items():
		
		writer.add_scalar('{}/{}'.format(prefix, key), value if value is not None else 0, step)


