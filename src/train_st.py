import time
import gc
import argparse
import sys
import os
import copy
from collections import namedtuple, OrderedDict
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from datalib.iterator import BucketIterator
from data_loader import *
from train_utils import *
from loss import *
from layers import get_padding_mask, repeat_tensors, reverse_seq, reverse_seq_value
from models import bd_style_transfer, bd_style_transfer_transformer, add_one_class, change_output_size
from search import SequenceGenerator
from torch.utils.tensorboard import SummaryWriter
import datalib.constants as constants
#torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark=True




class Solver(object):
	"""docstring for Solver"""
	def __init__(self, config):
		super(Solver, self).__init__()
		self.debug = config.debug
		self.use_transformer = config.use_transformer
		self.bt_cross = config.bt_cross
		self.csd_detach_enc = config.csd_detach_enc
		self.keep_only_full = config.keep_only_full
		self.tau = config.tau
		self.tau_up = config.tau_up
		self.tau_update_start = config.tau_update_start
		self.tau_update_end = config.tau_update_end
		self.greedy_train = config.greedy_train
		self.trans_extra_len = config.trans_extra_len
		self.trans_extra_len_eval = config.trans_extra_len_eval
		self.eval_beam_search = config.eval_beam_search
		self.eval_dropout = config.eval_dropout
		self.beam_size = config.beam_size
		self.smooth = config.smooth
		self.low_w_src = config.low_w_src
		self.high_w_src = config.high_w_src
		self.low_w_tsf = config.low_w_tsf
		self.high_w_tsf = config.high_w_tsf
		self.highlight = config.highlight
		self.enable_bt_hl = config.enable_bt_hl
		self.enable_cs_hl = config.enable_cs_hl
		self.enable_cd_hl = config.enable_cd_hl
		self.delete_cxt_win = config.delete_cxt_win
		self.insert_cxt_win = config.insert_cxt_win

		self.parallel = config.parallel

		self.rec_weight = config.rec_weight
		self.bt_weight = config.bt_weight
		self.clf_weight = config.clf_weight
		self.cs_weight = config.cs_weight
		self.cd_weight = config.cd_weight
		self.dis_weight = config.dis_weight
		self.bd = config.bd

		self.mono = self.bt_weight > 0 or self.clf_weight > 0 or ((self.cs_weight > 0 or self.cd_weight > 0 or self.dis_weight > 0) and self.bd)
		self.cs_sentence = config.cs_sentence
		self.bt_sg = config.bt_sg

		

		self.clf_adv = config.clf_adv
		self.clf_adv_mode = config.clf_adv_mode
		self.clf_adv_scale = config.clf_adv_scale
		self.clf_adv_src_scale = config.clf_adv_src_scale
		
		self.weight_up_start = config.weight_up_start
		self.weight_up_end = config.weight_up_end
		self.weight_down_start = config.weight_down_start
		self.weight_down_end = config.weight_down_end
		self.rec_down = config.rec_down
		self.dis_down = config.dis_down

		self.max_grad_norm = config.max_grad_norm
		self.update_interval = config.update_interval
		self.up_alpha = config.up_alpha
		self.down_alpha = config.down_alpha

		self.n_iters = config.n_iters
		self.log_interval = config.log_interval
		self.eval_interval = config.eval_interval
		self.eval_only = config.eval_only
		self.label_split = config.label_split
		device = torch.device('cuda', config.gpu)
		self.opt_accross_styles = config.opt_accross_styles
		self.output_detail = config.output_detail

		

		self.model_path = os.path.join(config.work_dir, 'model', config.dataset, 'trans', config.version)
		makedirs(self.model_path)
		self.output_path = os.path.join(config.work_dir, 'output', config.dataset, 'trans', config.version)
		makedirs(self.output_path)
		if not self.eval_only:
			self.summary_path = os.path.join(config.work_dir, 'summary', config.dataset, 'trans', config.version)
			makedirs(self.summary_path)
			self.summary_writer = SummaryWriter(self.summary_path)
		

		
		if self.clf_weight > 0 or (self.bd and self.dis_weight > 0):
			style_train_tool = torch.load(config.style_train_tool, map_location=device)['model']
			if self.bd and self.dis_weight > 0:
				self.dis_tool = copy.deepcopy(style_train_tool)
				change_output_size(self.dis_tool, 2)
				self.dis_tool.train()
				self.dis_optimizer = build_optimizer(config.optim_method, self.dis_tool, 
						config.dis_lr, config.momentum, config.weight_decay)
			if self.clf_weight > 0:
				self.style_train_tool = style_train_tool
				if self.clf_adv and self.clf_adv_mode=='ac':
					add_one_class(self.style_train_tool)
				if self.clf_adv:
					self.style_train_tool_update = copy.deepcopy(self.style_train_tool)
					self.style_train_tool_update.train()
					self.clf_adv_optimizer = build_optimizer(config.optim_method, self.style_train_tool_update, 
						config.clf_adv_lr, config.momentum, config.weight_decay)

				frozen_model(self.style_train_tool)
				if config.aux_model_eval_mode:
					self.style_train_tool.eval()
				else:
					self.style_train_tool.train()

		# load evaluators
		self.style_eval_tool = torch.load(config.style_eval_tool, map_location=device)['model']
		frozen_model(self.style_eval_tool)
		self.style_eval_tool.eval()

		self.fluency_eval_tool = torch.load(config.fluency_eval_tool, map_location=device)['model']
		frozen_model(self.fluency_eval_tool)
		self.fluency_eval_tool.eval()


		datasets = Corpus.iters_dataset(config.work_dir, config.dataset, 'trans', config.max_sen_len, config.cut_length, 
							config.min_freq, config.max_vocab_size, self.parallel, self.rec_weight>0 and not self.parallel, 
							config.noise_drop, para_data_path=config.para_data_path, shuffle_id=config.shuffle_id)
		if config.sub < 1:
			old_len = len(datasets[0])
			random.shuffle(datasets[0].examples)
			new_len = int(old_len * config.sub)
			datasets[0].examples = datasets[0].examples[:new_len]
			print('cutting the training data from {} to {}'.format(old_len, len(datasets[0])))
		self.valid_loader, self.test_loader = BucketIterator.splits(datasets[1:], 
							batch_size = config.eval_batch_size, 
							train_flags = [False]*2, device = device)
		if config.make_para_only:
			self.train_loaders = BucketIterator.splits(datasets[:1], 
							batch_size = config.eval_batch_size, 
							train_flags = [False], device = device)
		else:
			trainsets = datasets[0].stratify_split('label') if self.label_split else (datasets[0],)
			self.train_loaders = BucketIterator.splits(trainsets, 
							batch_size = config.batch_size, 
							train_flags = [True]*len(trainsets), device = device)
		

		vocab_size = len(self.train_loaders[0].dataset.fields['text'].vocab)
		num_classes = len(self.train_loaders[0].dataset.fields['label'].vocab)
		self.num_classes = num_classes
		self.dataset_stats = {'test':get_class_stats(datasets[2], 'label'), 'valid':get_class_stats(datasets[1], 'label')}
		self.human_files = get_human_files(config.work_dir, config.dataset, num_classes, config.num_ref, shuffle_id=config.shuffle_id)
		self.test_files = get_test_files(config.work_dir, config.dataset, num_classes, shuffle_id=config.shuffle_id)

		print('number of human files:', len(self.human_files))
		print('number of test files:', len(self.test_files))

		for i in range(len(self.train_loaders)):
			print('train size', i, ':', len(self.train_loaders[i].dataset))
		print('valid size:', len(self.valid_loader.dataset))
		print('test size:', len(self.test_loader.dataset))
		print('vocab size:', vocab_size)
		print('number of classes:', num_classes)

		self.itos = self.train_loaders[0].dataset.fields['text'].vocab.itos
		# self.label_itos = self.train_loaders[0].fields['label'].vocab.itos
		
		self.right1, self.right2 = config.right1, config.right2
		self.same_dir = self.right1 == self.right2
		if not self.use_transformer:
			self.model = bd_style_transfer(vocab_size, config.emb_size, config.emb_max_norm, 
				config.rnn_type, config.hid_size, True, config.dec_hid_size, config.num_layers, config.dec_num_layers, config.pooling_size,
				config.h_only, config.diff_bias, num_classes,
				config.feed_last_context, config.use_att, config.cxt_drop, self.bd, config.right1, config.right2)
		else:
			self.model = bd_style_transfer_transformer(vocab_size, config.emb_size, config.emb_max_norm, config.dropout_rate,
					config.num_heads, config.hid_size, config.num_layers, config.subseq_mask,
					config.diff_bias, num_classes, self.bd, config.right1, config.right2)
		
		if config.pretrained_emb is not None:
			text_vocab = self.train_loaders[0].dataset.fields['text'].vocab
			text_vocab.load_vectors(config.pretrained_emb, cache=os.path.join(config.work_dir, 'word_vectors'), max_vectors=config.pretrained_emb_max)
			self.model.emb.weight.data.copy_(text_vocab.vectors)
			text_vocab.vectors = None
		self.model.to(device)
		self.optimizer = build_optimizer(config.optim_method, self.model, config.lr, config.momentum, config.weight_decay, self.use_transformer)
		self.lr_scheduler = build_lr_scheduler(config.lr_use_noam, self.optimizer, config.lr_warmup_steps, 
												config.lr_decay_steps, config.lr_decay_mode, config.lr_min_factor, config.lr_decay_rate)
		self.step = 1
		if config.train_from is not None:
			check_point=torch.load(config.train_from, map_location=lambda storage, loc: storage)
			self.model.load_state_dict(check_point['model_state'])
			if self.clf_weight > 0 and self.clf_adv and 'clf_model_state' in check_point:
				self.style_train_tool_update.load_state_dict(check_point['clf_model_state'])
				self.style_train_tool.load_state_dict(check_point['clf_model_state'])
			if self.bd and self.dis_weight > 0 and 'dis_model_state' in check_point:
				self.dis_tool.load_state_dict(check_point['dis_model_state'])
			if config.load_optim:
				self.optimizer.load_state_dict(check_point['optimizer_state'])
				self.lr_scheduler.load_state_dict(check_point['lr_scheduler_state'])
				if self.clf_weight > 0 and self.clf_adv and 'clf_model_state' in check_point:
					self.clf_adv_optimizer.load_state_dict(check_point['clf_adv_optimizer_state'])
				if self.bd and self.dis_weight > 0 and 'dis_model_state' in check_point:
					self.dis_optimizer.load_state_dict(check_point['dis_optimizer_state'])
			self.step = check_point['step'] + 1
			del check_point
		if self.cd_weight > 0:
			self.model_bf = copy.deepcopy(self.model)
			frozen_model(self.model_bf)
		if self.eval_beam_search:
			self.beam_decoder = SequenceGenerator(self.beam_size, self.trans_extra_len_eval, config.min_len)


	def save_states(self, prefix = ''):
		check_point = {
			'step': self.step,
			'model_state': self.model.state_dict(),
			'optimizer_state': self.optimizer.state_dict(),
			'lr_scheduler_state': self.lr_scheduler.state_dict()
		}
		if self.clf_weight > 0 and self.clf_adv:
			check_point['clf_model_state'] = self.style_train_tool_update.state_dict()
			check_point['clf_adv_optimizer_state'] = self.clf_adv_optimizer.state_dict()
		if self.bd and self.dis_weight > 0:
			check_point['dis_model_state'] = self.dis_tool.state_dict()
			check_point['dis_optimizer_state'] = self.dis_optimizer.state_dict()
		filename = os.path.join(self.model_path, '{}model-{}'.format(prefix, self.step))
		torch.save(check_point, filename)

	def save_model(self):
		check_point = {
			'model': self.model
		}
		filename = os.path.join(self.model_path, 'full-model-{}'.format(self.step))
		torch.save(check_point, filename)
		
	def prepare_batch(self, batch):
		xc, x, enc_lens, y_out = batch.text
		if self.right2 or self.right1:
			right_x = reverse_seq(x, enc_lens, x.new_zeros(x.size(), dtype=torch.bool))
			right_y_out = reverse_seq(y_out, enc_lens, y_out==constants.EOS_ID)
		yin1, yout1 = (x, y_out) if not self.right1 else (right_x, right_y_out)
		if self.bd:
			yin2, yout2 = (x, y_out) if not self.right2 else (right_x, right_y_out)
		else:
			yin2, yout2 = None, None
		style = batch.label
		enc_padding_mask = get_padding_mask(x, enc_lens)
		enc_padding_mask_t = enc_padding_mask.t().contiguous() if self.use_transformer else None
		dec_lens = enc_lens + 1
		dec_padding_mask = get_padding_mask(y_out, dec_lens)
		dec_padding_mask_t = dec_padding_mask.t().contiguous() if self.use_transformer else None
		if self.parallel and self.model.training:
			xc, pseudo_enc_lens = batch.pseudo
			pseudo_enc_padding_mask = get_padding_mask(xc, pseudo_enc_lens, 0, 1) if self.use_transformer else get_padding_mask(xc, pseudo_enc_lens)
		elif self.parallel:
			pseudo_enc_lens, pseudo_enc_padding_mask = None, None
		else:
			pseudo_enc_lens, pseudo_enc_padding_mask = enc_lens, (enc_padding_mask_t if self.use_transformer else enc_padding_mask)
		
		return xc, pseudo_enc_lens, pseudo_enc_padding_mask, x, enc_lens, enc_padding_mask, enc_padding_mask_t, style, yin1, yout1, yin2, yout2, dec_lens, dec_padding_mask, dec_padding_mask_t

	def compute_loss(self, para_result, mono_result, x, style, yout1, yout2, enc_lens, dec_lens, enc_padding_mask, dec_padding_mask, size_average):
		loss_values = OrderedDict()

		loss_all = 0
		down_weight = rampdown(self.step, self.weight_down_start, self.weight_down_end, self.update_interval, self.down_alpha, True)
		
		if self.rec_weight > 0 and not (self.rec_down and self.step > self.weight_down_end):
			scale = (down_weight if self.rec_down else 1) * self.rec_weight
			rec1_loss = seq_ce_logits_loss(para_result['r1']['logits'], yout1, dec_lens, dec_padding_mask, size_average, smooth=self.smooth, ignore_index=constants.PAD_ID)
			loss_values['rec1'] = rec1_loss.item()
			loss_all = loss_all + scale * rec1_loss
			loss_values['rec1_acc'] = seq_acc(para_result['r1']['logits'], yout1, dec_lens, dec_padding_mask, size_average)
			if self.bd:
				rec2_loss = seq_ce_logits_loss(para_result['r2']['logits'], yout2, dec_lens, dec_padding_mask, size_average, smooth=self.smooth, ignore_index=constants.PAD_ID)
				loss_values['rec2'] = rec2_loss.item()
				loss_all = loss_all + scale * rec2_loss
				loss_values['rec2_acc'] = seq_acc(para_result['r2']['logits'], yout2, dec_lens, dec_padding_mask, size_average)
		
		if (self.weight_up_start is None or self.step > self.weight_up_start) and self.mono:
			up_weight = rampup(self.step, self.weight_up_start, self.weight_up_end, self.update_interval, self.up_alpha, True)
			if mono_result['fw1']['hard_outputs'].size(1) != yout1.size(1):
				yout1, yout2, dec_padding_mask = repeat_tensors(self.beam_size, 1, (yout1, yout2, dec_padding_mask))
				enc_lens, dec_lens = repeat_tensors(self.beam_size, 0, (enc_lens, dec_lens))
			loss_values['zl_rate1'] = mono_result['fw1']['hard_output_zl_mask'].float().mean().item() if size_average else mono_result['fw1']['hard_output_zl_mask'].float().sum().item()
			if self.bd:
				loss_values['zl_rate2'] = mono_result['fw2']['hard_output_zl_mask'].float().mean().item() if size_average else mono_result['fw2']['hard_output_zl_mask'].float().sum().item()
			loss_values['word_ent1'], loss_values['topp11'], loss_values['topp21'], loss_values['topp31'], loss_values['topp41'], loss_values['topp51'] = word_level_ent_top5(
				mono_result['fw1']['logits'], mono_result['fw1']['hard_outputs_lens'], 
				mono_result['fw1']['hard_outputs_padding_mask'], size_average)
			if self.bd:
				loss_values['word_ent2'], loss_values['topp12'], loss_values['topp22'], loss_values['topp32'], loss_values['topp42'], loss_values['topp52'] = word_level_ent_top5(
					mono_result['fw2']['logits'], mono_result['fw2']['hard_outputs_lens'], 
					mono_result['fw2']['hard_outputs_padding_mask'], size_average)
			if self.bd:
				loss_values['diversity'] = diversity(mono_result['fw1']['hard_outputs_rev' if self.right1 else 'hard_outputs'],
					mono_result['fw2']['hard_outputs_rev' if self.right2 else 'hard_outputs'],
					mono_result['fw1']['hard_outputs_lens'], mono_result['fw2']['hard_outputs_lens'])
			if self.highlight:
				tsf_w1, src_w1 = compute_change_weights(mono_result['fw1']['hard_outputs'], yout1, 
											mono_result['fw1']['hard_outputs_lens_with_eos'], dec_lens, self.delete_cxt_win, self.insert_cxt_win, 
											self.low_w_tsf, self.high_w_tsf, self.low_w_src, self.high_w_src)
				if self.bd:
					tsf_w2, src_w2 = compute_change_weights(mono_result['fw2']['hard_outputs'], yout2, 
											mono_result['fw2']['hard_outputs_lens_with_eos'], dec_lens, self.delete_cxt_win, self.insert_cxt_win, 
											self.low_w_tsf, self.high_w_tsf, self.low_w_src, self.high_w_src)
					if not self.same_dir:
						if self.bt_weight > 0 and self.bt_cross and self.enable_bt_hl:
							src_w1_rev = reverse_seq_value(src_w1, enc_lens, yout1==constants.EOS_ID)
							src_w2_rev = reverse_seq_value(src_w2, enc_lens, yout2==constants.EOS_ID)
						if self.cs_weight > 0 or self.cd_weight > 0 and (self.enable_cs_hl or self.enable_cd_hl):
							tsf_w1_rev = reverse_seq_value(tsf_w1, mono_result['fw1']['hard_outputs_lens'], mono_result['fw1']['eos_mask'])
							tsf_w2_rev = reverse_seq_value(tsf_w2, mono_result['fw2']['hard_outputs_lens'], mono_result['fw2']['eos_mask'])
			if self.bt_weight > 0:
				scale = self.bt_weight * up_weight
				if self.highlight and self.enable_bt_hl:
					if self.bd and self.bt_cross:
						reward1 = src_w2 if self.same_dir else src_w2_rev
					else:
						reward1 = src_w1
				bt1_loss = seq_ce_logits_loss(mono_result['bw1']['logits'], yout1, dec_lens, dec_padding_mask, size_average, 
											batch_mask=mono_result['fw2' if self.bd and self.bt_cross else 'fw1']['hard_output_zl_mask'], 
											rewards=reward1 if self.highlight and self.enable_bt_hl else None,
											smooth=self.smooth, ignore_index=constants.PAD_ID)
				loss_values['bt1'] = bt1_loss.item()
				loss_values['bt1_acc'] = seq_acc(mono_result['bw1']['logits'], yout1, dec_lens, dec_padding_mask, size_average)
				loss_all = loss_all + scale * bt1_loss
				if self.bd:
					if self.highlight and self.enable_bt_hl:
						if self.bt_cross:
							reward2 = src_w1 if self.same_dir else src_w1_rev
						else:
							reward2 = src_w2
					bt2_loss = seq_ce_logits_loss(mono_result['bw2']['logits'], yout2, dec_lens, dec_padding_mask, size_average, 
												batch_mask=mono_result['fw1' if self.bt_cross else 'fw2']['hard_output_zl_mask'], 
												rewards=reward2 if self.highlight and self.enable_bt_hl else None,
												smooth=self.smooth, ignore_index=constants.PAD_ID)
					loss_values['bt2'] = bt2_loss.item()
					loss_values['bt2_acc'] = seq_acc(mono_result['bw2']['logits'], yout2, dec_lens, dec_padding_mask, size_average)
					loss_all = loss_all + scale * bt2_loss
				
			if self.clf_weight > 0:
				scale = self.clf_weight * up_weight
				style_logits1 = self.style_train_tool(mono_result['fw1']['soft_outputs_rev' if self.right1 else 'soft_outputs'], 
						mono_result['fw1']['hard_outputs_padding_mask'], 
						soft_input = True)
				clf1_loss = F.cross_entropy(style_logits1, mono_result['to_style'], reduction = 'mean' if size_average else 'sum')
				loss_values['clf1'] = clf1_loss.item()
				loss_all = loss_all + scale * clf1_loss
				loss_values['style_acc1'] = unit_acc(style_logits1, mono_result['to_style'], size_average)
				if self.bd:
					style_logits2 = self.style_train_tool(mono_result['fw2']['soft_outputs_rev' if self.right2 else 'soft_outputs'], 
							mono_result['fw2']['hard_outputs_padding_mask'], 
							soft_input = True)
					clf2_loss = F.cross_entropy(style_logits2, mono_result['to_style'], reduction = 'mean' if size_average else 'sum')
					loss_values['clf2'] = clf2_loss.item()
					loss_all = loss_all + scale * clf2_loss
					loss_values['style_acc2'] = unit_acc(style_logits2, mono_result['to_style'], size_average)
				
				if self.clf_adv and self.clf_adv_scale != 0:
					src_logits = self.style_train_tool_update(x, enc_padding_mask)
					clf_adv_src_loss = F.cross_entropy(src_logits, style, reduction = 'mean' if size_average else 'sum')
					loss_values['clf_adv_src'] = clf_adv_src_loss.item()
					loss_all = loss_all + scale * self.clf_adv_scale * self.clf_adv_src_scale * clf_adv_src_loss
					loss_values['clf_adv_src_acc'] = unit_acc(src_logits, style, size_average)
					
					if self.clf_adv_mode == 'src':
						style = repeat_tensors(self.beam_size, 0, style)
					tsf_logits1 = self.style_train_tool_update(mono_result['fw1']['hard_outputs_rev' if self.right1 else 'hard_outputs'], 
						mono_result['fw1']['hard_outputs_padding_mask'])
					clf_adv_tsf1_loss = adv_loss(tsf_logits1, style, mono_result['to_style'], self.clf_adv_mode, size_average)
					loss_values['clf_adv_tsf1'] = clf_adv_tsf1_loss.item()
					loss_all = loss_all + scale * self.clf_adv_scale * clf_adv_tsf1_loss
					loss_values['clf_adv_tsf1_acc'] = unit_acc(tsf_logits1, mono_result['to_style'], size_average)
					if self.bd:
						tsf_logits2 = self.style_train_tool_update(mono_result['fw2']['hard_outputs_rev' if self.right2 else 'hard_outputs'], 
							mono_result['fw2']['hard_outputs_padding_mask'])
						clf_adv_tsf2_loss = adv_loss(tsf_logits2, style, mono_result['to_style'], self.clf_adv_mode, size_average)
						loss_values['clf_adv_tsf2'] = clf_adv_tsf2_loss.item()
						loss_all = loss_all + scale * self.clf_adv_scale * clf_adv_tsf2_loss
						loss_values['clf_adv_tsf2_acc'] = unit_acc(tsf_logits2, mono_result['to_style'], size_average)

			if self.bd and self.cs_weight > 0:
				scale = self.cs_weight * up_weight
				if self.highlight and self.enable_cs_hl:
					reward1 = tsf_w2 if self.same_dir else tsf_w2_rev
					reward2 = tsf_w1 if self.same_dir else tsf_w1_rev
				if self.cs_sentence:
					cs1_loss = seq_ce_logits_loss(mono_result['cs1']['logits'][:-1], mono_result['fw2']['hard_outputs' if self.same_dir else 'hard_outputs_rev'], 
						mono_result['fw2']['hard_outputs_lens_with_eos'], mono_result['fw2']['hard_outputs_padding_mask_with_eos'], size_average, 
						rewards=reward1 if self.highlight and self.enable_cs_hl else None,
						smooth=self.smooth, ignore_index=constants.PAD_ID)
					cs2_loss = seq_ce_logits_loss(mono_result['cs2']['logits'][:-1], mono_result['fw1']['hard_outputs' if self.same_dir else 'hard_outputs_rev'], 
						mono_result['fw1']['hard_outputs_lens_with_eos'], mono_result['fw1']['hard_outputs_padding_mask_with_eos'], size_average, 
						rewards=reward2 if self.highlight and self.enable_cs_hl else None,
						smooth=self.smooth, ignore_index=constants.PAD_ID)
				else:
					cs1_loss = seq_kl_logits_loss(mono_result['cs1']['logits'][:-1], mono_result['fw2']['logits' if self.same_dir else 'logits_rev'].detach(),
						mono_result['fw2']['hard_outputs_lens_with_eos'], mono_result['fw2']['hard_outputs_padding_mask_with_eos'], size_average,
						rewards=reward1 if self.highlight and self.enable_cs_hl else None)
					cs2_loss = seq_kl_logits_loss(mono_result['cs2']['logits'][:-1], mono_result['fw1']['logits' if self.same_dir else 'logits_rev'].detach(),
						mono_result['fw1']['hard_outputs_lens_with_eos'], mono_result['fw1']['hard_outputs_padding_mask_with_eos'], size_average,
						rewards=reward2 if self.highlight and self.enable_cs_hl else None)
				loss_values['cs1'] = cs1_loss.item()
				loss_values['cs2'] = cs2_loss.item()
				loss_all = loss_all + scale * (cs1_loss + cs2_loss)

			if self.bd and self.cd_weight > 0:
				scale = self.cd_weight * up_weight
				if self.highlight and self.enable_cd_hl:
					reward1 = tsf_w2 if self.same_dir else tsf_w2_rev
					reward2 = tsf_w1 if self.same_dir else tsf_w1_rev
				cd1_loss = - seq_ce_logits_loss(mono_result['cd1']['logits'][:-1], mono_result['fw2']['hard_outputs' if self.same_dir else 'hard_outputs_rev'],
					mono_result['fw2']['hard_outputs_lens_with_eos'], mono_result['fw2']['hard_outputs_padding_mask_with_eos'], size_average, 
					rewards=reward1 if self.highlight and self.enable_cd_hl else None)
				cd2_loss = - seq_ce_logits_loss(mono_result['cd2']['logits'][:-1], mono_result['fw1']['hard_outputs' if self.same_dir else 'hard_outputs_rev'],
					mono_result['fw1']['hard_outputs_lens_with_eos'], mono_result['fw1']['hard_outputs_padding_mask_with_eos'], size_average, 
					rewards=reward2 if self.highlight and self.enable_cd_hl else None)
				loss_values['cd1'] = cd1_loss.item()
				loss_values['cd2'] = cd2_loss.item()
				loss_all = loss_all + scale * (cd1_loss + cd2_loss)

			if self.bd and self.dis_weight > 0 and not (self.dis_down and self.step > self.weight_down_end):
				scale = self.dis_weight * up_weight * (down_weight if self.dis_down else 1)
				dis_logits1 = self.dis_tool(mono_result['fw1']['soft_outputs_rev' if self.right1 else 'soft_outputs'], 
						mono_result['fw1']['hard_outputs_padding_mask'], 
						soft_input = True)
				dis1_target = style.new_zeros(dis_logits1.size(0))
				dis1_loss = F.cross_entropy(dis_logits1, dis1_target, reduction = 'mean' if size_average else 'sum')
				loss_values['dis1'] = dis1_loss.item()
				loss_all = loss_all + scale * dis1_loss
				loss_values['dis1_acc'] = unit_acc(dis_logits1, dis1_target, size_average)

				dis_logits2 = self.dis_tool(mono_result['fw2']['soft_outputs_rev' if self.right2 else 'soft_outputs'], 
						mono_result['fw2']['hard_outputs_padding_mask'], 
						soft_input = True)
				dis2_target = style.new_ones(dis_logits2.size(0))
				dis2_loss = F.cross_entropy(dis_logits2, dis2_target, reduction = 'mean' if size_average else 'sum')
				loss_values['dis2'] = dis2_loss.item()
				loss_all = loss_all + scale * dis2_loss
				loss_values['dis2_acc'] = unit_acc(dis_logits2, dis2_target, size_average)

		loss_values['loss_total'] = loss_all.item()
		return loss_all, loss_values

	def train_batch(self, batch, para, mono, bt, cs, cd):
		if self.debug:
			b0 = time.time()
		xc, pseudo_enc_lens, pseudo_enc_padding_mask, x, enc_lens, enc_padding_mask, enc_padding_mask_t, style, yin1, yout1, yin2, yout2, dec_lens, dec_padding_mask, dec_padding_mask_t = self.prepare_batch(batch)
		# self.optimizer.zero_grad()
		if self.debug:
			b1 = time.time()
			print('preparing batch time {:.4f} s'.format(b1-b0))
		tau = update_arg(self.tau, self.tau_up, self.step, self.update_interval, self.tau_update_start, self.tau_update_end, self.up_alpha, self.down_alpha)

		# if self.model_type == 'uni':
		# 	result['s_l2r'] = self.model.para_transfer(xc, pseudo_enc_lens, pseudo_enc_padding_mask, x, style, self.parallel) if para else None
		# 	result['t_l2r'], result['t_l2r_b'] = self.model.mono_transfer(x, enc_lens, enc_padding_mask, bt, style, tau, self.greedy_train, self.beam_size, 
		# 		self.trans_extra_len) if mono else (None, None)
		
		if not self.use_transformer:
			para_result = self.model.para_transfer(xc, pseudo_enc_lens, pseudo_enc_padding_mask, yin1, None, None, None, yin2, style, self.parallel) if para else None
			mono_result = self.model.mono_transfer(x, yin1, yin2, enc_lens, enc_padding_mask, bt, cs, cd, self.model_bf if cd else None, style, tau, 
				self.greedy_train, self.beam_size, self.trans_extra_len, self.bt_cross, self.bt_sg, self.csd_detach_enc) if mono else None
		else:
			para_result = self.model.para_transfer(xc, pseudo_enc_padding_mask, dec_padding_mask_t, yin1, None, None, dec_padding_mask_t, yin2, style) if para else None
			mono_result = self.model.mono_transfer(x, yin1, yin2, enc_lens, enc_padding_mask_t, dec_padding_mask_t, bt, cs, cd, self.model_bf if cd else None, style, 
				tau, self.greedy_train, self.beam_size, self.trans_extra_len, self.bt_cross, self.bt_sg, self.csd_detach_enc) if mono else None
		if self.debug:
			b2 = time.time()
			print('forward time {:.4f} s'.format(b2-b1))
		loss, loss_values = self.compute_loss(para_result, mono_result, x, style, yout1, yout2, enc_lens, dec_lens, enc_padding_mask, dec_padding_mask, True)
		if self.debug:
			b3 = time.time()
			print('loss time {:.4f} s'.format(b3-b2))

		loss.backward()
		if self.debug:
			b4 = time.time()
			print('backward time {:.4f} s'.format(b4-b3))
		# clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
		# self.optimizer.step()
		# add_to_writer(loss_values, self.step, 'train', self.summary_writer)

		return loss_values

	def train(self):
		def prepare_optimize(clf_adv_flag, dis_flag, loss_accl):
			self.optimizer.zero_grad()
			if clf_adv_flag:
				self.clf_adv_optimizer.zero_grad()
			if dis_flag:
				self.dis_optimizer.zero_grad()
			loss_accl.clear()
		def optimize(clf_adv_flag, dis_flag, loss_accl, num_batches):
			if self.max_grad_norm is not None:
				clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
				if clf_adv_flag:
					clip_grad_norm_(self.style_train_tool_update.parameters(), self.max_grad_norm)
				if dis_flag:
					clip_grad_norm_(self.dis_tool.parameters(), self.max_grad_norm)
			
			self.optimizer.step()
			if clf_adv_flag:
				self.clf_adv_optimizer.step()
				self.style_train_tool.load_state_dict(self.style_train_tool_update.state_dict())
			if dis_flag:
				self.dis_optimizer.step()
			if self.cd_weight > 0:
				self.model_bf.load_state_dict(self.model.state_dict())
			self.lr_scheduler.step()

			if num_batches > 1:
				get_final_accl(loss_accl, num_batches)
			add_to_writer(loss_accl, self.step, 'train', self.summary_writer)
			if self.step % self.log_interval == 0:
				print('step [{}/{}] {:.2f} s elapsed'. format(self.step, self.n_iters, time.time() - start))
				print_loss(loss_accl)
			if self.step % self.eval_interval == 0:
				self.eval('valid', self.valid_loader)
				
				if (self.weight_up_start is None or self.step > self.weight_up_start):
					self.eval('test', self.test_loader, True)
				self.save_states('latest-')
				

		self.model.train()
		data_iters = [iter(tl) for tl in self.train_loaders]
		accl = OrderedDict()
		start = time.time()
		while self.step <= self.n_iters:
			if self.rec_down and self.rec_weight > 0 and self.step > self.weight_down_end:
				self.train_loaders[0].dataset.fields['text'].postprocessing.turn_off_noise()
			
			# self.optimizer.zero_grad()
			# accl.clear()
			para = self.rec_weight > 0 and not (self.rec_down and self.step > self.weight_down_end)
			up_stage = (self.weight_up_start is None or self.step > self.weight_up_start)
			mono = up_stage and self.mono
			bt = up_stage and self.bt_weight > 0
			cs = up_stage and self.cs_weight > 0
			cd = up_stage and self.cd_weight > 0
			clf_adv = up_stage and self.clf_weight > 0 and self.clf_adv and self.clf_adv_scale != 0
			dis = up_stage and self.bd and self.dis_weight > 0 and not (self.dis_down and self.step > self.weight_down_end)

			if self.opt_accross_styles:
				prepare_optimize(clf_adv, dis, accl)
			update_flag = True
			i = 0
			while i < len(data_iters):
				if not self.opt_accross_styles:
					prepare_optimize(clf_adv, dis, accl)
				b0 = debug_time_msg(self.debug)
				batch = next(data_iters[i])
				b1 = debug_time_msg(self.debug, b0, 'read data')
				try:
					loss_values = self.train_batch(batch, para, mono, bt, cs, cd)
					update_accl(accl, loss_values)
				except RuntimeError as e:
					if ('out of memory' in str(e)):
						print('step {} | WARNING: {}; skipping batch; redoing this step'.format(self.step, str(e)))
						update_flag = False
						gc.collect()
						torch.cuda.empty_cache()
						if self.opt_accross_styles:
							break
						else:
							continue
					else:
						raise e
				if not self.opt_accross_styles:
					optimize(clf_adv, dis, accl, 1)
					self.step += 1
				i += 1
			if self.opt_accross_styles and update_flag:
				optimize(clf_adv, dis, accl, len(data_iters))
				self.step += 1
		self.summary_writer.close()

	def eval(self, name, dataset_loader, with_truth = False):
		if not self.eval_dropout:
			self.model.eval()
		n_total = len(dataset_loader.dataset)
		accl = OrderedDict()
		start = time.time()
		trans_sens1 = []
		if self.bd:
			trans_sens2 = []
			select_trans_sens = []
		with_detail = self.output_detail
		if with_detail:
			details1 = []
			if self.bd:
				details2 = []
		
		with torch.no_grad():
			for i, batch in enumerate(dataset_loader):
				xc, pseudo_enc_lens, pseudo_enc_padding_mask, x, enc_lens, enc_padding_mask, enc_padding_mask_t, style, yin1, yout1, yin2, yout2, dec_lens, dec_padding_mask, dec_padding_mask_t = self.prepare_batch(batch)
				tau = update_arg(self.tau, self.tau_up, self.step, self.update_interval, self.tau_update_start, self.tau_update_end, self.up_alpha, self.down_alpha)

				# result_fw = self.model.mono_transfer(x, enc_lens, enc_padding_mask, False, None, None, style, True, 1, True, 1, self.trans_extra_len_eval, None, False, None, self.test_cat_format)[0]
				if self.eval_beam_search:
					result = self.model.mono_beam_search(x, enc_lens, enc_padding_mask_t if self.use_transformer else enc_padding_mask, style, tau, self.beam_decoder)
				elif not self.use_transformer:
					result = self.model.mono_transfer(x, None, None, enc_lens, enc_padding_mask, False, False, False, None, style, 1, 
						True, 1, self.trans_extra_len_eval, None, None, None)
				else:
					result = self.model.mono_transfer(x, None, None, enc_lens, enc_padding_mask_t, None, False, False, False, None, style, 
						1, True, 1, self.trans_extra_len_eval, None, None, None)
				word_ent1, topp11, topp21, topp31, topp41, topp51, ent_mat1, probs_mat1, top_inds1 = word_level_ent_top5(result['fw1']['logits_rev' if self.right1 else 'logits'],
					result['fw1']['hard_outputs_lens'], result['fw1']['hard_outputs_padding_mask'], False, True)
				update_accl(accl, {'word_ent1': word_ent1, 'topp11': topp11, 'topp21': topp21, 'topp31': topp31, 'topp41': topp41, 'topp51': topp51})
				if self.bd:
					word_ent2, topp12, topp22, topp32, topp42, topp52, ent_mat2, probs_mat2, top_inds2 = word_level_ent_top5(result['fw2']['logits_rev' if self.right2 else 'logits'],
						result['fw2']['hard_outputs_lens'], result['fw2']['hard_outputs_padding_mask'], False, True)
					update_accl(accl, {'word_ent2': word_ent2, 'topp12': topp12, 'topp22': topp22, 'topp32': topp32, 'topp42': topp42, 'topp52': topp52})
				start_tok = x.new_full((1, x.size(1)), constants.BOS_ID)
				style_eval_acc1 = unit_acc(self.style_eval_tool(result['fw1']['hard_outputs_rev' if self.right1 else 'hard_outputs'], 
					result['fw1']['hard_outputs_padding_mask']), 
					result['to_style'], False, False)
				fluency_eval_nll1 = seq_ce_logits_loss(
					self.fluency_eval_tool(torch.cat([start_tok, result['fw1']['hard_outputs_rev' if self.right1 else 'hard_outputs']], 0), 
						result['to_style'])[:-1], 
					result['fw1']['hard_outputs_rev' if self.right1 else 'hard_outputs'], result['fw1']['hard_outputs_lens_with_eos'], 
					result['fw1']['hard_outputs_padding_mask_with_eos'], False, reduction=False)
				update_accl(accl, {'style_eval_acc1': style_eval_acc1.sum().item(), 'fluency_eval_nll1': fluency_eval_nll1.sum().item()})
				if self.bd:
					style_eval_acc2 = unit_acc(self.style_eval_tool(result['fw2']['hard_outputs_rev' if self.right2 else 'hard_outputs'], 
						result['fw2']['hard_outputs_padding_mask']), 
						result['to_style'], False, False)
					fluency_eval_nll2 = seq_ce_logits_loss(
						self.fluency_eval_tool(torch.cat([start_tok, result['fw2']['hard_outputs_rev' if self.right2 else 'hard_outputs']], 0), 
							result['to_style'])[:-1], 
						result['fw2']['hard_outputs_rev' if self.right2 else 'hard_outputs'], result['fw2']['hard_outputs_lens_with_eos'], 
						result['fw2']['hard_outputs_padding_mask_with_eos'], False, reduction=False)
					update_accl(accl, {'style_eval_acc2': style_eval_acc2.sum().item(), 'fluency_eval_nll2': fluency_eval_nll2.sum().item()})

					if self.dis_weight > 0:
						zeros = style.new_zeros(style.size(0))
						ones = style.new_ones(style.size(0))
						
						soft_dis1_acc = unit_acc(self.dis_tool(result['fw1']['soft_outputs_rev' if self.right1 else 'soft_outputs'], 
								result['fw1']['hard_outputs_padding_mask'], 
								soft_input = True), zeros, False)
						hard_dis1_acc = unit_acc(self.dis_tool(result['fw1']['hard_outputs_rev' if self.right1 else 'hard_outputs'], 
								result['fw1']['hard_outputs_padding_mask'], 
								soft_input = False), zeros, False)
						soft_dis2_acc = unit_acc(self.dis_tool(result['fw2']['soft_outputs_rev' if self.right2 else 'soft_outputs'], 
								result['fw2']['hard_outputs_padding_mask'], 
								soft_input = True), ones, False)
						hard_dis2_acc = unit_acc(self.dis_tool(result['fw2']['hard_outputs_rev' if self.right2 else 'hard_outputs'], 
								result['fw2']['hard_outputs_padding_mask'], 
								soft_input = False), ones, False)
						update_accl(accl, {'soft_dis1_acc': soft_dis1_acc, 'hard_dis1_acc': hard_dis1_acc, 
							'soft_dis2_acc': soft_dis2_acc, 'hard_dis2_acc': hard_dis2_acc})
					
					if self.eval_beam_search:
						nlog_probs1 = -result['fw1']['scores']
						nlog_probs2 = -result['fw2']['scores']
					else:
						nlog_probs1 = seq_ce_logits_loss(result['fw1']['logits'], result['fw1']['hard_outputs'], 
							result['fw1']['hard_outputs_lens_with_eos'], result['fw1']['hard_outputs_padding_mask_with_eos'],
							False, reduction=False)
						nlog_probs2 = seq_ce_logits_loss(result['fw2']['logits'], result['fw2']['hard_outputs'], 
							result['fw2']['hard_outputs_lens_with_eos'], result['fw2']['hard_outputs_padding_mask_with_eos'],
							False, reduction=False)
					_, inds = torch.stack((nlog_probs1, nlog_probs2)).min(0, keepdim=True)
					select_style_eval_acc = torch.stack((style_eval_acc1, style_eval_acc2)).gather(0, inds)
					select_fluency_eval_nll = torch.stack((fluency_eval_nll1, fluency_eval_nll2)).gather(0, inds)
					update_accl(accl, {'style_eval_acc': select_style_eval_acc.sum().item(), 'fluency_eval_nll': select_fluency_eval_nll.sum().item()})
				
				if (self.weight_up_start is None or self.step > self.weight_up_start):
					sens1 = to_sentence_list(result['fw1']['hard_outputs_rev' if self.right1 else 'hard_outputs'], result['fw1']['hard_outputs_lens'], self.itos)
					trans_sens1.extend(sens1)
					if self.bd:
						sens2 = to_sentence_list(result['fw2']['hard_outputs_rev' if self.right2 else 'hard_outputs'], result['fw2']['hard_outputs_lens'], self.itos)
						trans_sens2.extend(sens2)
						select_trans_sens.extend(select_sentence(sens1, sens2, inds.squeeze(0)))
					if with_detail:
						to_details(ent_mat1, probs_mat1, top_inds1, result['fw1']['hard_outputs_lens'], self.itos, details1)
						if self.bd:
							to_details(ent_mat2, probs_mat2, top_inds2, result['fw2']['hard_outputs_lens'], self.itos, details2)
					
		get_final_accl(accl, n_total)
		if (self.weight_up_start is None or self.step > self.weight_up_start):
			accl['fluency_eval_nppl1'] = - math.exp(accl['fluency_eval_nll1'])
			if self.bd:
				accl['fluency_eval_nppl2'] = - math.exp(accl['fluency_eval_nll2'])
				accl['fluency_eval_nppl'] = - math.exp(accl['fluency_eval_nll'])
				accl['diversity'] = get_bleu(trans_sens1, trans_sens2)
			
			src_sens = list(dataset_loader.dataset.text)
			print('src example (index 0 out of {}): {}'.format(len(src_sens), src_sens[0]))

			trans_sens1 = reorder(dataset_loader.order, trans_sens1)
			print('trans1 example (index 0 out of {}): {}'.format(len(trans_sens1), trans_sens1[0]))
			result_file1 = os.path.join(self.output_path, '{}-result1-{}'.format(name, self.step))
			save_results(src_sens, trans_sens1, result_file1)
			if self.bd:
				trans_sens2 = reorder(dataset_loader.order, trans_sens2)
				print('trans2 example (index 0 out of {}): {}'.format(len(trans_sens2), trans_sens2[0]))
				result_file2 = os.path.join(self.output_path, '{}-result2-{}'.format(name, self.step))
				save_results(src_sens, trans_sens2, result_file2)

				select_trans_sens = reorder(dataset_loader.order, select_trans_sens)
				print('trans example (index 0 out of {}): {}'.format(len(select_trans_sens), select_trans_sens[0]))
				result_file = os.path.join(self.output_path, '{}-result-{}'.format(name, self.step))
				save_results(src_sens, select_trans_sens, result_file)
			if with_detail:
				details1 = reorder(dataset_loader.order, details1)
				detail_file1 = os.path.join(self.output_path, '{}-result1-{}.detail'.format(name, self.step))
				save_details(src_sens, trans_sens1, details1, detail_file1)
				if self.bd:
					details2 = reorder(dataset_loader.order, details2)
					detail_file2 = os.path.join(self.output_path, '{}-result2-{}.detail'.format(name, self.step))
					save_details(src_sens, trans_sens2, details2, detail_file2)

			
			accl['self_bleu1'], accl['human_bleu1'] = self.save_outputs_and_compute_bleu(name, 1, trans_sens1, with_truth)
			append_scores(result_file1, accl['style_eval_acc1'], -accl['fluency_eval_nppl1'], accl['self_bleu1'], accl['human_bleu1'])
			if self.bd:
				accl['self_bleu2'], accl['human_bleu2'] = self.save_outputs_and_compute_bleu(name, 2, trans_sens2, with_truth)
				append_scores(result_file2, accl['style_eval_acc2'], -accl['fluency_eval_nppl2'], accl['self_bleu2'], accl['human_bleu2'])

				accl['self_bleu'], accl['human_bleu'] = self.save_outputs_and_compute_bleu(name, '', select_trans_sens, with_truth)
				append_scores(result_file, accl['style_eval_acc'], -accl['fluency_eval_nppl'], accl['self_bleu'], accl['human_bleu'])

			

		if not self.eval_only:
			add_to_writer(accl, self.step, name, self.summary_writer)
		print('{} performance of model at step {}'.format(name, self.step))
		print_loss(accl)
		print('{:.2f} s for evaluation'.format(time.time() - start))
		if not self.eval_dropout:
			self.model.train()
		

		

	def save_outputs_and_compute_bleu(self, name, decoder_ind, trans_sens, with_truth):
		stats = self.dataset_stats[name]
		cur_ind = 0
		self_bleu = 0
		if with_truth:
			human_bleu = 0
		
		for i in range(self.num_classes):
			output_file = os.path.join(self.output_path, '{}-result{}-{}.{}'.format(name, decoder_ind, self.step, i))
			save_outputs(trans_sens, cur_ind, cur_ind+stats[i], output_file)
			self_bleu += compute_bleu_score(self.test_files[i], output_file)
			if with_truth:
				human_bleu += compute_bleu_score(self.human_files[i], output_file)
			cur_ind = cur_ind + stats[i]
			if self.keep_only_full:
				os.remove(output_file)
		self_bleu /= self.num_classes
		if with_truth:
			human_bleu /= self.num_classes
		return (self_bleu, human_bleu) if with_truth else (self_bleu, None) 

	def generate_parallel(self):
		# assume the decoder has only one direction
		if not self.eval_dropout:
			self.model.eval()
		dataset_loader = self.train_loaders[0]
		start = time.time()
		trans_sens = []
		with torch.no_grad():
			for i, batch in enumerate(dataset_loader):
				xc, pseudo_enc_lens, pseudo_enc_padding_mask, x, enc_lens, enc_padding_mask, enc_padding_mask_t, style, yin1, yout1, yin2, yout2, dec_lens, dec_padding_mask, dec_padding_mask_t = self.prepare_batch(batch)
				tau = update_arg(self.tau, self.tau_up, self.step, self.update_interval, self.tau_update_start, self.tau_update_end, self.up_alpha, self.down_alpha)

				if self.eval_beam_search:
					result = self.model.mono_beam_search(x, enc_lens, enc_padding_mask_t if self.use_transformer else enc_padding_mask, style, tau, self.beam_decoder)
				elif not self.use_transformer:
					result = self.model.mono_transfer(x, None, None, enc_lens, enc_padding_mask, False, False, False, None, style, 1, 
						True, 1, self.trans_extra_len_eval, None, None, None)
				else:
					result = self.model.mono_transfer(x, None, None, enc_lens, enc_padding_mask_t, None, False, False, False, None, style, 
						1, True, 1, self.trans_extra_len_eval, None, None, None)
				to_sentences(result['fw1']['hard_outputs_rev' if self.right1 else 'hard_outputs'], 
					result['fw1']['hard_outputs_lens'], self.itos, trans_sens)
					
		trans_sens = reorder(dataset_loader.order, trans_sens)
		src_sens = list(dataset_loader.dataset.text)
		src_labels = list(dataset_loader.dataset.label)
		print('trans example (index 0 out of {}): {}'.format(len(trans_sens), trans_sens[0]))
		print('src example (index 0 out of {}): {}'.format(len(src_sens), src_sens[0]))
		
		result_file = os.path.join(self.output_path, 'para-result-{}'.format(self.step))
		save_parallel_results(src_sens, src_labels, trans_sens, result_file)

		print('{:.2f} s for parallel data generation'.format(time.time() - start))
		if not self.eval_dropout:
			self.model.train()
		

		
if __name__=='__main__':

	parser=argparse.ArgumentParser()
	parser.add_argument('version', type=str)
	parser.add_argument('-seed', type=int, default=1)
	parser.add_argument('-shuffle_id', type=int, default=None, nargs='?')
	parser.add_argument('-gpu', type=int, default=0)
	parser.add_argument('-debug', type=str2bool, default=False)
	parser.add_argument('-cudnn_enabled', type=str2bool, default=True)
	parser.add_argument('-eval_only', type=str2bool, default=False)
	parser.add_argument('-eval_beam_search', type=str2bool, default=False)
	parser.add_argument('-eval_dropout', type=str2bool, default=False)
	parser.add_argument('-keep_only_full', type=str2bool, default=False)
	parser.add_argument('-make_para_only', type=str2bool, default=False)
	parser.add_argument('-label_split', type=str2bool, default=False)
	parser.add_argument('-load_optim', type=str2bool, default=False)
	parser.add_argument('-opt_accross_styles', type=str2bool, default=True, nargs='?')
	parser.add_argument('-train_from', type=str, default=None, nargs='?')
	parser.add_argument('-style_train_tool', type=str, default=None, nargs='?')
	parser.add_argument('-style_eval_tool', type=str, default=None)
	parser.add_argument('-fluency_eval_tool', type=str, default=None)
	parser.add_argument('-aux_model_eval_mode', type=str2bool, default=False, nargs='?')
	parser.add_argument('-num_ref', type=int, default=1)
	parser.add_argument('-sub', type=float, default=1)
	parser.add_argument('-pretrained_emb', type=str, default=None, nargs='?')
	parser.add_argument('-pretrained_emb_max', type=int, default=None, nargs='?')
	parser.add_argument('-parallel', type=str2bool, default=False)
	parser.add_argument('-para_data_path', type=str, default=None, nargs='?')
	parser.add_argument('-output_detail', type=str2bool, default=False, nargs='?')

	
	parser.add_argument('-work_dir', type=str, default='/sata/')
	parser.add_argument('-dataset', type=str, default='yelp')
	parser.add_argument('-max_sen_len', type=int, default=None, nargs='?')
	parser.add_argument('-cut_length', type=int, default=None, nargs='?')
	parser.add_argument('-min_freq', type=int, default=1)
	parser.add_argument('-max_vocab_size', type=int, default=None, nargs='?')
	parser.add_argument('-noise_drop', type=float, default=0.1, nargs='?')
	parser.add_argument('-batch_size', type=int, default=64)
	parser.add_argument('-eval_batch_size', type=int, default=64)
	parser.add_argument('-smooth', type=float, default=0.0)
	parser.add_argument('-low_w_src', type=float, default=1.0, nargs='?')
	parser.add_argument('-high_w_src', type=float, default=1.0, nargs='?')
	parser.add_argument('-low_w_tsf', type=float, default=1.0, nargs='?')
	parser.add_argument('-high_w_tsf', type=float, default=1.0, nargs='?')
	parser.add_argument('-highlight', type=str2bool, default=True, nargs='?')
	parser.add_argument('-enable_bt_hl', type=str2bool, default=True, nargs='?')
	parser.add_argument('-enable_cs_hl', type=str2bool, default=True, nargs='?')
	parser.add_argument('-enable_cd_hl', type=str2bool, default=True, nargs='?')
	parser.add_argument('-insert_cxt_win', type=int, default=0, nargs='?')
	parser.add_argument('-delete_cxt_win', type=int, default=1, nargs='?')
	parser.add_argument('-min_len', type=int, default=1, nargs='?')
	
	parser.add_argument('-use_transformer', type=str2bool, default=True)
	parser.add_argument('-bd', type=str2bool, default=True)
	parser.add_argument('-right1', type=str2bool, default=False)
	parser.add_argument('-right2', type=str2bool, default=True, nargs='?')
	parser.add_argument('-use_att', type=str2bool, default=True, nargs='?')
	parser.add_argument('-emb_size', type=int, default=200)
	parser.add_argument('-emb_max_norm', type=float, default=1.0)
	parser.add_argument('-pooling_size', type=int, default=5, nargs='?')
	parser.add_argument('-rnn_type', type=str, default='GRU', nargs='?')
	parser.add_argument('-hid_size', type=int, default=200)
	parser.add_argument('-dec_hid_size', type=int, default=200, nargs='?')
	parser.add_argument('-num_heads', type=int, default=8, nargs='?')
	parser.add_argument('-num_layers', type=int, default=1)
	parser.add_argument('-dec_num_layers', type=int, default=1, nargs='?')
	parser.add_argument('-h_only', type=str2bool, default=True, nargs='?')
	parser.add_argument('-diff_bias', type=str2bool, default=True)
	parser.add_argument('-feed_last_context', type=str2bool, default=True, nargs='?')
	parser.add_argument('-cxt_drop', type=float, default=0.0, nargs='?')
	parser.add_argument('-dropout_rate', type=float, default=0.0, nargs='?')
	parser.add_argument('-subseq_mask', type=str2bool, default=True, nargs='?')

	parser.add_argument('-tau', type=float, default=0.5)
	parser.add_argument('-tau_up', type=str2bool, default=True, nargs='?')
	parser.add_argument('-tau_update_start', type=int, default=None, nargs='?')
	parser.add_argument('-tau_update_end', type=int, default=None, nargs='?')
	parser.add_argument('-greedy_train', type=str2bool, default=False)
	parser.add_argument('-trans_extra_len', type=int, default=0)
	parser.add_argument('-trans_extra_len_eval', type=int, default=5)
	parser.add_argument('-beam_size', type=int, default=5, nargs='?')

	parser.add_argument('-clf_adv', type=str2bool, default=False, nargs='?')
	parser.add_argument('-clf_adv_mode', type=str, default='ac', nargs='?')
	parser.add_argument('-clf_adv_scale', type=float, default=1.0, nargs='?')
	parser.add_argument('-clf_adv_lr', type=float, default=0.001, nargs='?')
	parser.add_argument('-clf_adv_src_scale', type=float, default=1.0, nargs='?')

	
	parser.add_argument('-rec_weight', type=float, default=1.0)
	parser.add_argument('-bt_weight', type=float, default=1.0)
	parser.add_argument('-cd_weight', type=float, default=1.0)
	parser.add_argument('-cs_weight', type=float, default=1.0)
	parser.add_argument('-clf_weight', type=float, default=1.0)
	parser.add_argument('-bt_cross', type=str2bool, default=False, nargs='?')
	parser.add_argument('-csd_detach_enc', type=str2bool, default=False, nargs='?')
	parser.add_argument('-cs_sentence', type=str2bool, default=False, nargs='?')
	parser.add_argument('-bt_sg', type=str2bool, default=True, nargs='?')

	parser.add_argument('-dis_weight', type=float, default=0.0)
	parser.add_argument('-dis_lr', type=float, default=0.001, nargs='?')
	
	parser.add_argument('-weight_up_start', type=int, default=None, nargs='?')
	parser.add_argument('-weight_up_end', type=int, default=None, nargs='?')
	parser.add_argument('-weight_down_start', type=int, default=None, nargs='?')
	parser.add_argument('-weight_down_end', type=int, default=None, nargs='?')
	parser.add_argument('-rec_down', type=str2bool, default=False, nargs='?')
	parser.add_argument('-dis_down', type=str2bool, default=False, nargs='?')
	
	
	

	parser.add_argument('-max_grad_norm', type=float, default=2.0)
	parser.add_argument('-optim_method', type=str, default='adam')
	parser.add_argument('-momentum', type=float, default=None, nargs='?')
	parser.add_argument('-weight_decay', type=float, default=0)
	parser.add_argument('-lr', type=float, default=0.001)
	parser.add_argument('-update_interval', type=int, default=500, nargs='?')
	parser.add_argument('-up_alpha', type=float, default=None, nargs='?')
	parser.add_argument('-down_alpha', type=float, default=None, nargs='?')
	parser.add_argument('-lr_use_noam', type=str2bool, default=False, nargs='?')
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

	torch.backends.cudnn.enabled=config.cudnn_enabled

	random.seed(config.seed)
	np.random.seed(config.seed+1)
	torch.manual_seed(config.seed+2)
	torch.cuda.manual_seed(config.seed+3)

	print('Start time: ', time.strftime('%X %x %Z'))
	trainer = Solver(config)
	if config.eval_only:
		trainer.eval('test', trainer.test_loader, True)
	elif config.make_para_only:
		trainer.generate_parallel()
	else:
		trainer.train()
	print('Finish time: ', time.strftime('%X %x %Z'))
