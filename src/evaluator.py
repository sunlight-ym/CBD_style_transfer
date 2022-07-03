import sys
import codecs
import math
import argparse
import time
import torch
from datalib.iterator import BucketIterator
from data_loader import *
from train_utils import *
from loss import *
from layers import get_padding_mask
from datalib.batch import Batch
class Evaluator(object):
	"""docstring for Evaluator"""
	def __init__(self, config):
		super(Evaluator, self).__init__()
		self.bin_size = config.bin_size
		self.num_bins = config.num_bins
		self.batch_size = config.batch_size
		self.num_ref = config.num_ref
		device = torch.device('cuda', config.gpu)
		self.device = device
		testset = Corpus.iters_output(config.work_dir, config.dataset, config.model_name, config.text_vocab, config.label_vocab, model_is_file_path=config.model_is_file_path)
		print('performance of model {} on dataset {}'.format(config.model_name, config.dataset))
		self.test_loader = BucketIterator(testset, batch_size=config.batch_size, train=False, device=device)
		# self.true_sens = read_sens(get_human_file(config.work_dir, config.dataset)) if config.with_truth else None
		# self.src_sens = read_sens(get_model_src_file(config.work_dir, config.dataset, config.model_out_dir))

		print('test size:', len(self.test_loader.dataset))
		# print('src input size:', len(self.src_sens))
		# print('true output size:', len(self.true_sens) if self.true_sens is not None else 0)
		self.num_classes = len(testset.fields['label'].vocab)
		self.human_files = get_human_files(config.work_dir, config.dataset, self.num_classes, config.num_ref)
		self.test_files = get_test_files(config.work_dir, config.dataset, self.num_classes)
		if config.model_is_file_path:
			self.model_out_files = get_model_files_custom(config.model_name, self.num_classes)
		else:
			self.model_out_files = get_model_files(config.work_dir, config.dataset, config.model_name, self.num_classes)

		# load evaluators
		self.style_eval_tool = torch.load(config.style_eval_tool, map_location=device)['model']
		frozen_model(self.style_eval_tool)
		self.style_eval_tool.eval()

		self.fluency_eval_tool = torch.load(config.fluency_eval_tool, map_location=device)['model']
		frozen_model(self.fluency_eval_tool)
		self.fluency_eval_tool.eval()

	def prepare_batch(self, batch):
		x_class, x_lm, y_lm, lm_lens = batch.text
		tgt_style = batch.label
		class_padding_mask = get_padding_mask(x_class, lm_lens - 1)
		lm_padding_mask = get_padding_mask(x_lm, lm_lens)
		return x_class, class_padding_mask, tgt_style, x_lm, y_lm, lm_lens, lm_padding_mask

	def eval(self):
		n_total = len(self.test_loader.dataset)
		total_acc, total_nll = 0, 0
		start = time.time()
		with torch.no_grad():
			for batch in self.test_loader:
				x_class, class_padding_mask, tgt_style, x_lm, y_lm, lm_lens, lm_padding_mask = self.prepare_batch(Batch(batch, self.test_loader.dataset, self.device))
				style_eval_logits = self.style_eval_tool(x_class, class_padding_mask)
				total_acc += unit_acc(style_eval_logits, tgt_style, False)

				fluency_eval_logits = self.fluency_eval_tool(x_lm, tgt_style)
				total_nll += seq_ce_logits_loss(fluency_eval_logits, y_lm, lm_lens, lm_padding_mask, False).item()
		total_acc /= n_total
		total_nll /= n_total
		total_ppl = math.exp(total_nll)
		# trans_sens = list(self.test_loader.dataset.text)
		# self_bleu = get_bleu(trans_sens, self.src_sens)
		# human_bleu = get_bleu(trans_sens, self.true_sens) if self.true_sens is not None else None

		
		self_bleu = 0
		human_bleu = 0
		for i in range(self.num_classes):
			self_bleu += compute_bleu_score(self.test_files[i], self.model_out_files[i])
			human_bleu += compute_bleu_score(self.human_files[i], self.model_out_files[i])
		self_bleu /= self.num_classes
		human_bleu /= self.num_classes

		print('transfer accuracy: {:.2%}\n'.format(total_acc))
		print('transfer perplexity: {:.2f}\n'.format(total_ppl))
		print('self bleu: {:.2f}\n'.format(self_bleu))
		print('human bleu: {:.2f}\n'.format(human_bleu))
		print('{:.2f} s for evaluation'.format(time.time() - start))


	def bleu_cal_for_bin(self, src, gen, tgt_list, bin_id):
		with open(f'src_bin_{bin_id}', 'w') as f:
			for line in src:
				f.write(line)
		with open(f'gen_bin_{bin_id}', 'w') as f:
			for line in gen:
				f.write(line)
		for i in range(self.num_ref):
			with open(f'tgt_bin_{bin_id}_{i}', 'w') as f:
				for line in tgt_list[i]:
					f.write(line)

		self_bleu = compute_bleu_score(f'src_bin_{bin_id}', f'gen_bin_{bin_id}')
		human_bleu = compute_bleu_score([f'tgt_bin_{bin_id}_{i}' for i in range(self.num_ref)], f'gen_bin_{bin_id}')
		os.remove(f'src_bin_{bin_id}')
		os.remove(f'gen_bin_{bin_id}')
		for i in range(self.num_ref):
			os.remove(f'tgt_bin_{bin_id}_{i}')
		return self_bleu, human_bleu

		
	def eval_bins(self):
		start = time.time()
		n_total = len(self.test_loader.dataset)
		# total_acc, total_nll = 0, 0
		acc_stat = torch.zeros(self.num_bins, dtype=torch.float, device=self.device)
		nll_stat = torch.zeros(self.num_bins, dtype=torch.float, device=self.device)
		self_bleu_stat = torch.empty(self.num_bins, dtype=torch.float)
		human_bleu_stat = torch.empty(self.num_bins, dtype=torch.float)
		bin_capacity = torch.zeros(self.num_bins, dtype=torch.float, device=self.device)
		
		record_bins_test = [[] for i in range(self.num_bins)]
		record_bins_model = [[] for i in range(self.num_bins)]
		record_bins_human = [[[] for k in range(self.num_ref)] for i in range(self.num_bins)]
		ind_list = torch.empty(n_total, dtype=torch.long, device=self.device)
		j = 0
		max_len = 0
		for test_file, model_out, human_file_list in zip(self.test_files, self.model_out_files, self.human_files):
			with open(test_file, 'r') as ft:
				with codecs.open(model_out, 'r', encoding='utf-8', errors='ignore') as fm:
					fhs = [codecs.open(ref, 'r', encoding='utf-8', errors='ignore') for ref in human_file_list]
					for linet in ft:
						length = len(linet.strip().split(' '))
						if length > max_len:
							max_len = length
						ind = min(length//self.bin_size, self.num_bins-1)
						ind_list[j] = ind
						j += 1
						bin_capacity[ind] += 1
						record_bins_test[ind].append(linet)
						record_bins_model[ind].append(fm.readline())
						for k in range(self.num_ref):
							record_bins_human[ind][k].append(fhs[k].readline())
					for fh in fhs:
						fh.close()

		for i in range(self.num_bins):
			self_bleu_stat[i], human_bleu_stat[i] = self.bleu_cal_for_bin(record_bins_test[i], record_bins_model[i], record_bins_human[i], i)
		del record_bins_test, record_bins_model, record_bins_human
		
		assert bin_capacity.sum().item() == n_total



		
		with torch.no_grad():
			for batch_id, batch in enumerate(self.test_loader):
				if batch_id == 0:
					ind_list = ind_list[torch.tensor(self.test_loader.order, dtype=torch.long, device=self.device)]
				x_class, class_padding_mask, tgt_style, x_lm, y_lm, lm_lens, lm_padding_mask = self.prepare_batch(Batch(batch, self.test_loader.dataset, self.device))
				style_eval_logits = self.style_eval_tool(x_class, class_padding_mask)
				batch_acc = unit_acc(style_eval_logits, tgt_style, False, False)

				fluency_eval_logits = self.fluency_eval_tool(x_lm, tgt_style)
				batch_nll = seq_ce_logits_loss(fluency_eval_logits, y_lm, lm_lens, lm_padding_mask, False, reduction=False)
				for bin_id, acc, nll in zip(ind_list[(batch_id*self.batch_size):(batch_id*self.batch_size+batch_acc.size(0))], batch_acc, batch_nll):
					acc_stat[bin_id] += acc
					nll_stat[bin_id] += nll


		# total_acc /= n_total
		# total_nll /= n_total
		# total_ppl = math.exp(total_nll)
		acc_stat /= bin_capacity
		nll_stat /= bin_capacity
		ppl_stat = torch.exp(nll_stat)
		# trans_sens = list(self.test_loader.dataset.text)
		# self_bleu = get_bleu(trans_sens, self.src_sens)
		# human_bleu = get_bleu(trans_sens, self.true_sens) if self.true_sens is not None else None

		
		# self_bleu = 0
		# human_bleu = 0
		# for i in range(self.num_classes):
		# 	self_bleu += compute_bleu_score(self.test_files[i], self.model_out_files[i])
		# 	human_bleu += compute_bleu_score(self.human_files[i], self.model_out_files[i])
		# self_bleu /= self.num_classes
		# human_bleu /= self.num_classes
		print('max length', max_len)
		bounds = [f'[{i*self.bin_size},{(i+1)*self.bin_size})' for i in range(self.num_bins-1)]
		bounds.append(f'[{(self.num_bins-1)*self.bin_size},inf)')
		print('bins:\t', bounds)
		print('bin capacity:\t', bin_capacity)

		print('transfer accuracy:\t', acc_stat)
		print('transfer perplexity:\t', ppl_stat)
		print('self bleu:\t', self_bleu_stat)
		print('human bleu:\t', human_bleu_stat)
		print('{:.2f} s for evaluation'.format(time.time() - start))

if __name__=='__main__':

	parser=argparse.ArgumentParser()
	parser.add_argument('-gpu', type=int, default=0)
	parser.add_argument('-work_dir', type=str, default='./')
	parser.add_argument('-dataset', type=str, default='yelp')
	parser.add_argument('-model_name', type=str, default='cae')
	
	parser.add_argument('-num_ref', type=int, default=1)
	parser.add_argument('-text_vocab', type=str, default=None)
	parser.add_argument('-label_vocab', type=str, default=None)
	
	parser.add_argument('-batch_size', type=int, default=64)

	parser.add_argument('-style_eval_tool', type=str, default=None)
	parser.add_argument('-fluency_eval_tool', type=str, default=None)

	parser.add_argument('-model_is_file_path', type=str2bool, default=False)
	parser.add_argument('-cal_div_only', type=str2bool, default=False)
	parser.add_argument('-div_file1', type=str, default=None, nargs='?')
	parser.add_argument('-div_file2', type=str, default=None, nargs='?')
	parser.add_argument('-div_num_classes', type=int, default=2, nargs='?')
	parser.add_argument('-eval_by_len', type=str2bool, default=False)
	parser.add_argument('-bin_size', type=int, default=2, nargs='?')
	parser.add_argument('-num_bins', type=int, default=5, nargs='?')

	config=parser.parse_args()
	print(' '.join(sys.argv))
	print(config)

	print('Start time: ', time.strftime('%X %x %Z'))
	if config.cal_div_only:
		diversity = 0
		for i in range(config.div_num_classes):
			f1 = f'{config.div_file1}.{i}'
			f2 = f'{config.div_file2}.{i}'
			diversity += compute_bleu_score(f1, f2)
		diversity /= config.div_num_classes
		print(f'The bleu between {config.div_file1}\n and {config.div_file2}\n is {diversity:.4f}')
	else:
		evaluator = Evaluator(config)
		if config.eval_by_len:
			evaluator.eval_bins()
		else:
			evaluator.eval()
	print('Finish time: ', time.strftime('%X %x %Z'))
