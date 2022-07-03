import time
import sys
import os
from collections import namedtuple
import math
import random
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from datalib.iterator import BucketIterator
from data_loader import Corpus
from train_utils import *
from loss import *
from layers import get_padding_mask
from models import language_model
from torch.utils.tensorboard import SummaryWriter



Record = namedtuple('Record', 'step ppl the_other', defaults = (0, float('inf'), float('inf')))

class Solver(object):
	"""docstring for Solver"""
	def __init__(self, config):
		super(Solver, self).__init__()
		self.select_label = config.select_label
		self.max_grad_norm = config.max_grad_norm
		# self.eps = config.eps
		# self.lr = config.lr
		# self.update_interval = config.update_interval
		# self.up_alpha = config.up_alpha
		# self.down_alpha = config.down_alpha
		# self.lr_up_start = config.lr_up_start
		# self.lr_up_end = config.lr_up_end
		# self.lr_down_start = config.lr_down_start
		# self.lr_down_end = config.lr_down_end
		self.n_iters = config.n_iters
		self.log_interval = config.log_interval
		self.eval_interval = config.eval_interval
		self.eval_only = config.eval_only
		device = torch.device('cuda', config.gpu)

		datasets = Corpus.iters_dataset(config.work_dir, config.dataset, 'lm', config.max_sen_len,
							config.cut_length, config.min_freq, config.max_vocab_size)
		if config.ignore_splits:
			# for classifier training, we don't use the test split for style transfer as it is too small
			# so we split the training set to new dev and test
			datasets = datasets[0].split([0.8, 0.1, 0.1], True)
		if self.select_label is not None:
			datasets = zip(*[d.stratify_split('label') if d is not None else None for d in datasets])
			datasets = list(datasets)[self.select_label]
		self.train_loader, self.valid_loader, self.test_loader = BucketIterator.splits(datasets, 
							batch_sizes = [config.batch_size, config.eval_batch_size, config.eval_batch_size], 
							device = device, retain_order = False)

		self.model_path = os.path.join(config.work_dir, 'model', config.dataset, 'lm_'+('all' if self.select_label is None else str(self.select_label)), 
										config.version)
		self.summary_path = os.path.join(config.work_dir, 'summary', config.dataset, 'lm_'+('all' if self.select_label is None else str(self.select_label)), 
										config.version)
		makedirs(self.model_path)
		makedirs(self.summary_path)
		if not (self.eval_only or config.save_only):
			self.summary_writer = SummaryWriter(self.summary_path)

		vocab_size = len(self.train_loader.dataset.fields['text'].vocab)

		print('train size:', len(self.train_loader.dataset))
		print('valid size:', len(self.valid_loader.dataset))
		print('test size:', len(self.test_loader.dataset))
		print('vocab size:', vocab_size)

		num_styles = len(self.train_loader.dataset.fields['label'].vocab)
		print('number of styles:', num_styles)
		self.diff_bias = config.diff_bias

		self.model = language_model(vocab_size, config.emb_size, config.emb_max_norm, config.rnn_size, config.rnn_type, config.dropout_rate, num_styles if self.diff_bias==True else 1)
		if config.pretrained_emb is not None:
			text_vocab = self.train_loader.dataset.fields['text'].vocab
			text_vocab.load_vectors(config.pretrained_emb, cache=os.path.join(config.work_dir, 'word_vectors'), max_vectors=config.pretrained_emb_max)
			self.model.emb.weight.data.copy_(text_vocab.vectors)
			text_vocab.vectors = None
		self.model.to(device)
		self.optimizer = build_optimizer(config.optim_method, self.model, config.lr, config.momentum, config.weight_decay)
		self.lr_scheduler = build_lr_scheduler(False, self.optimizer, config.lr_warmup_steps, 
												config.lr_decay_steps, config.lr_decay_mode, config.lr_min_factor, config.lr_decay_rate)
		self.step = 1
		self.best_results = {'valid': Record(), 'test': Record()}
		if config.train_from is not None:
			check_point=torch.load(config.train_from, map_location=lambda storage, loc: storage)
			self.model.load_state_dict(check_point['model_state'])
			self.optimizer.load_state_dict(check_point['optimizer_state'])
			self.lr_scheduler.load_state_dict(check_point['lr_scheduler_state'])
			self.step = check_point['step']
			self.best_results = check_point['best_results']
			del check_point

		

	def save_states(self, prefix = ''):
		check_point = {
			'step': self.step,
			'model_state': self.model.state_dict(),
			'optimizer_state': self.optimizer.state_dict(),
			'lr_scheduler_state': self.lr_scheduler.state_dict(),
			'best_results': self.best_results
		}
		filename = os.path.join(self.model_path, '{}model-{}'.format(prefix, self.step))
		torch.save(check_point, filename)

	def save_model(self):
		check_point = {
			'model': self.model
		}
		filename = os.path.join(self.model_path, 'full-model-{}'.format(self.step))
		torch.save(check_point, filename)
		
	def prepare_batch(self, batch):
		x, lens, t = batch.text
		style = batch.label if self.diff_bias==True else None
		padding_mask = get_padding_mask(x, lens)
		return x, lens, padding_mask, t, style

	def train_batch(self, batch):
		x, lens, padding_mask, t, style = self.prepare_batch(batch)
		self.optimizer.zero_grad()
		y = self.model(x, style)
		loss = seq_ce_logits_loss(y, t, lens, padding_mask, True)
		acc = seq_acc(y, t, lens, padding_mask, True)
		loss_value = loss.item()
		ppl = math.exp(loss_value)
		loss.backward()
		clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
		self.optimizer.step()
		self.lr_scheduler.step()

		self.summary_writer.add_scalar('train/loss', loss_value, self.step)
		self.summary_writer.add_scalar('train/ppl', ppl, self.step)
		self.summary_writer.add_scalar('train/acc', acc, self.step)

		return loss_value, acc, ppl

	def train(self):
		self.model.train()
		data_iter = iter(self.train_loader)
		start = time.time()
		while self.step <= self.n_iters:
			# update_lr(self.optimizer, self.lr, self.step, self.update_interval, 
							# self.lr_up_start, self.lr_up_end, self.lr_down_start, self.lr_down_end, self.up_alpha, self.down_alpha, self.eps)
			batch = next(data_iter)
			loss, acc, ppl = self.train_batch(batch)
			
			if self.step % self.log_interval == 0:
				print('step [{}/{}] loss: {:.4f}; acc: {:.2%}; ppl: {:.4f} | {:.2f} s elapsed'. format(
							self.step, self.n_iters, loss, acc, ppl, time.time() - start))
			if self.step % self.eval_interval == 0:
				valid_ppl = self.eval('valid', self.valid_loader)
				test_ppl = self.eval('test', self.test_loader)

				save_flag = False
				if valid_ppl < self.best_results['valid'].ppl:
					save_flag = True
					self.best_results['valid'] = Record(step = self.step, ppl = valid_ppl, the_other = test_ppl)
				if test_ppl < self.best_results['test'].ppl:
					save_flag = True
					self.best_results['test'] = Record(step = self.step, ppl = test_ppl, the_other = valid_ppl)
				print('current best valid: step {0.step} ppl {0.ppl:.4f} [{0.the_other:.4f}]'.format(self.best_results['valid']))
				print('current best test: step {0.step} ppl {0.ppl:.4f} [{0.the_other:.4f}]'.format(self.best_results['test']))
				if save_flag:
					self.save_states()
				self.save_states('latest-')
				# to save space
				if self.step != self.eval_interval:
					os.remove(os.path.join(self.model_path, 'latest-model-{}'.format(self.step - self.eval_interval)))
			self.step += 1
		self.summary_writer.close()

	def eval(self, name, dataset_loader):
		self.model.eval()
		n_total = len(dataset_loader.dataset)
		total_loss, total_acc = 0, 0
		start = time.time()
		with torch.no_grad():
			for batch in dataset_loader:
				x, lens, padding_mask, t, style = self.prepare_batch(batch)
				y = self.model(x, style)
				total_loss += seq_ce_logits_loss(y, t, lens, padding_mask, False).item()
				total_acc += seq_acc(y, t, lens, padding_mask, False)
		total_loss /= n_total
		total_acc /= n_total
		total_ppl = math.exp(total_loss)

		if not self.eval_only:
			self.summary_writer.add_scalar('{}/loss'.format(name), total_loss, self.step)
			self.summary_writer.add_scalar('{}/ppl'.format(name), total_ppl, self.step)
			self.summary_writer.add_scalar('{}/acc'.format(name), total_acc, self.step)

		print('{} performance of model at step {}'.format(name, self.step))
		print('loss: {:.4f}; acc: {:.2%}; ppl: {:.4f} | {:.2f} s for evaluation'. format(
							total_loss, total_acc, total_ppl, time.time() - start))
		self.model.train()
		return total_ppl

if __name__=='__main__':
	parser=argparse.ArgumentParser()
	parser.add_argument('version', type=str)
	parser.add_argument('-seed', type=int, default=1)
	parser.add_argument('-gpu', type=int, default=0)
	parser.add_argument('-eval_only', type=str2bool, default=False)
	parser.add_argument('-save_only', type=str2bool, default=False)
	parser.add_argument('-ignore_splits', type=str2bool, default=False)
	parser.add_argument('-train_from', type=str, default=None, nargs='?')
	parser.add_argument('-select_label', type=int, default=None, nargs='?')
	parser.add_argument('-diff_bias', type=str2bool, default=False, nargs='?')
	parser.add_argument('-pretrained_emb', type=str, default=None, nargs='?')
	parser.add_argument('-pretrained_emb_max', type=int, default=None, nargs='?')

	parser.add_argument('-work_dir', type=str, default='/sata/')
	parser.add_argument('-dataset', type=str, default='yelp')
	parser.add_argument('-max_sen_len', type=int, default=None, nargs='?')
	parser.add_argument('-cut_length', type=int, default=None, nargs='?')
	parser.add_argument('-min_freq', type=int, default=1)
	parser.add_argument('-max_vocab_size', type=int, default=None, nargs='?')
	parser.add_argument('-batch_size', type=int, default=64)
	parser.add_argument('-eval_batch_size', type=int, default=64)
	# parser.add_argument('-drop_last', type=str2bool, default=False)

	parser.add_argument('-emb_size', type=int, default=200)
	parser.add_argument('-emb_max_norm', type=float, default=1.0)
	parser.add_argument('-rnn_size', type=int, default=100)
	parser.add_argument('-rnn_type', type=str, default='GRU')
	parser.add_argument('-dropout_rate', type=float, default=0)

	# parser.add_argument('-eps', type=float, default=1e-10, nargs='?')
	parser.add_argument('-max_grad_norm', type=float, default=2.0)
	parser.add_argument('-optim_method', type=str, default='adam')
	parser.add_argument('-momentum', type=float, default=None, nargs='?')
	parser.add_argument('-weight_decay', type=float, default=0)
	parser.add_argument('-lr', type=float, default=0.001)
	parser.add_argument('-lr_warmup_steps', type=int, default=4000)
	parser.add_argument('-lr_decay_steps', type=int, default=0)
	parser.add_argument('-lr_decay_mode', type=str, default=None, nargs='?')
	parser.add_argument('-lr_min_factor', type=float, default=None, nargs='?')
	parser.add_argument('-lr_decay_rate', type=float, default=None, nargs='?')
	parser.add_argument('-n_iters', type=int, default=100000)
	parser.add_argument('-log_interval', type=int, default=10)
	parser.add_argument('-eval_interval', type=int, default=500)

	config=parser.parse_args()
	print(' '.join(sys.argv))
	print(config)

	random.seed(config.seed)
	np.random.seed(config.seed+1)
	torch.manual_seed(config.seed+2)
	torch.cuda.manual_seed(config.seed+3)

	print('Start time: ', time.strftime('%X %x %Z'))
	trainer = Solver(config)
	if config.eval_only:
		trainer.eval('test', trainer.test_loader)
	elif config.save_only:
		trainer.save_model()
	else:
		trainer.train()
	print('Finish time: ', time.strftime('%X %x %Z'))

