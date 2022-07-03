import math
import subprocess
import re
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import check_values
# from layers import repeat_tensors

def compute_change_weights(seqs1, seqs2, lens1, lens2, delete_cxt_win=0, insert_cxt_win=1, low_w1=1, high_w1=1, low_w2=1, high_w2=1):
	# seqs2 here should use the src + eos version, otherwise the eos will be considered as a change
	bsz = seqs1.size(1)
	device = seqs1.device
	change_mask1 = torch.zeros(seqs1.size(), dtype=torch.bool, device=device)
	change_mask2 = torch.zeros(seqs2.size(), dtype=torch.bool, device=device)
	max_len1 = lens1.max().item()
	max_len2 = lens2.max().item()
	edit_dists = torch.zeros(bsz, max_len1+1, max_len2+1, dtype=torch.float, device=device)
	parent_type = torch.zeros(bsz, max_len1+1, max_len2+1, dtype=torch.long, device=device)
	# parent_i = torch.zeros(bsz, max_len1+1, max_len2+1, dtype=torch.long, device=device)
	# parent_j = torch.zeros(bsz, max_len1+1, max_len2+1, dtype=torch.long, device=device)

	# run batch version of the levenshtein algorithm
	edit_dists[:, :, 0] = torch.arange(max_len1+1, dtype=torch.float, device=device)
	edit_dists[:, 0, :] = torch.arange(max_len2+1, dtype=torch.float, device=device)
	# types: 0 - same; 1 - replace; 2 - row needs insert / column needs delete; 3 - column needs insert / row needs delete
	parent_type[:, 1:, 0] = 3
	parent_type[:, 0, 1:] = 2
	# parent_i[:, 1:, 0] -= 1
	# parent_j[:, 0, 1:] -= 1
	for i in range(1, max_len1+1):
		for j in range(1, max_len2+1):
			cost = (seqs1[i-1, :] != seqs2[j-1, :])
			min_cost, min_inds = torch.cat(((edit_dists[:, i-1, j-1] + cost.float()).unsqueeze(dim=1),
									 (edit_dists[:, i, j-1] + 1).unsqueeze(dim=1),
									 (edit_dists[:, i-1, j] + 1).unsqueeze(dim=1)), dim=1).min(dim=1)
			edit_dists[:, i, j] = min_cost
			parent_type[:, i, j].masked_scatter_(cost, min_inds.masked_select(cost)+1)

	batch_inds = torch.arange(bsz, dtype=torch.long, device=device)
	cur_i = lens1
	cur_j = lens2
	while (cur_i>0).any() or (cur_j>0).any():
		cur_type = parent_type[batch_inds, cur_i, cur_j]
		type1 = cur_type==1
		b1 = batch_inds.masked_select(type1)
		change_mask1[cur_i.masked_select(type1)-1, b1]=True
		change_mask2[cur_j.masked_select(type1)-1, b1]=True

		type2 = cur_type==2
		b2 = batch_inds.masked_select(type2)
		ilens = lens1.masked_select(type2)
		jlens = lens2.masked_select(type2)
		i2 = cur_i.masked_select(type2)-1
		i2 = torch.cat([i2 if k==0 else (torch.min(i2+k, ilens-1) if k>0 else (i2+k).clamp_(min=0)) for k in range(-insert_cxt_win+1, insert_cxt_win+1)], 0)
		j2 = cur_j.masked_select(type2)-1
		j2 = torch.cat([j2 if k==0 else (torch.min(j2+k, jlens-1) if k>0 else (j2+k).clamp_(min=0)) for k in range(-delete_cxt_win, delete_cxt_win+1)], 0)
		change_mask1[i2, b2.repeat(2*insert_cxt_win)]=True
		change_mask2[j2, b2.repeat(2*delete_cxt_win+1)]=True

		type3 = cur_type==3
		b3 = batch_inds.masked_select(type3)
		ilens = lens1.masked_select(type3)
		jlens = lens2.masked_select(type3)
		i3 = cur_i.masked_select(type3)-1
		i3 = torch.cat([i3 if k==0 else (torch.min(i3+k, ilens-1) if k>0 else (i3+k).clamp_(min=0)) for k in range(-delete_cxt_win, delete_cxt_win+1)], 0)
		j3 = cur_j.masked_select(type3)-1
		j3 = torch.cat([j3 if k==0 else (torch.min(j3+k, jlens-1) if k>0 else (j3+k).clamp_(min=0)) for k in range(-insert_cxt_win+1, insert_cxt_win+1)], 0)
		change_mask1[i3, b3.repeat(2*delete_cxt_win+1)]=True
		change_mask2[j3, b3.repeat(2*insert_cxt_win)]=True

		cur_i = torch.where((cur_i==0) | type2, cur_i, cur_i-1)
		cur_j = torch.where((cur_j==0) | type3, cur_j, cur_j-1)

	w1 = edit_dists.new_full(change_mask1.size(), low_w1)
	w1.masked_fill_(change_mask1, high_w1)
	w2 = edit_dists.new_full(change_mask2.size(), low_w2)
	w2.masked_fill_(change_mask2, high_w2)
	return w1, w2

	
def seq_ce_logits_loss(y, t, t_lens, mask, size_average, seq_dim=0, batch_mask=None, reduction=True, rewards=None, smooth=0, ignore_index=-1):
	# check_values(y, 'logits', False)
	y = F.log_softmax(y, 2)
	if smooth == 0:
		loss = -torch.gather(y, 2, t.unsqueeze(-1)).squeeze(-1)
	else:
		smooth_target = y.new_full(y.size(), smooth/y.size(-1))
		smooth_target.scatter_(2, t.unsqueeze(-1), 1-smooth)
		if ignore_index >= 0 and ignore_index < y.size(-1):
			smooth_target[:, :, ignore_index] = 0
		loss = - torch.sum(y * smooth_target, -1)

	if rewards is not None:
		loss = loss * rewards

	# if change_w is not None:
	# 	w = loss.new_full(loss.size(), low_w)
	# 	w.masked_fill_(change_mask, high_w)
	# 	loss = loss * w
	if mask is not None:
		loss = loss.masked_fill(mask, 0)
		loss = loss.sum(seq_dim) / t_lens.float()
	else:
		loss = loss.mean(seq_dim)
	if batch_mask is not None:
		loss = loss.masked_fill(batch_mask, 0)

	return (loss.mean() if size_average else loss.sum()) if reduction else loss

def seq_kl_logits_loss(y, t, t_lens, mask, size_average, seq_dim=0, batch_mask=None, reduction=True, rewards=None):
	# t should be logits here
	y = F.log_softmax(y, 2)
	loss = - y * F.softmax(t, 2)
	loss = loss.sum(2)

	if rewards is not None:
		loss = loss * rewards

	if mask is not None:
		loss = loss.masked_fill(mask, 0)
		loss = loss.sum(seq_dim) / t_lens.float()
	else:
		loss = loss.mean(seq_dim)
	if batch_mask is not None:
		loss = loss.masked_fill(batch_mask, 0)

	return (loss.mean() if size_average else loss.sum()) if reduction else loss

def seq_acc(y, t, t_lens, mask, size_average, seq_dim=0):
	y = y.detach()
	pred = torch.argmax(y, 2)
	loss = (pred == t) & (~mask)
	loss = loss.float().sum(seq_dim) / t_lens.float()

	return loss.mean().item() if size_average else loss.sum().item()

def unit_acc(y, t, size_average = True, reduction=True):
	y = y.detach()
	pred = torch.argmax(y, 1)
	loss = (pred == t).float()

	return (loss.mean().item() if size_average else loss.sum().item()) if reduction else loss

def adv_loss(y, from_style, to_style, mode, size_average):
	if mode == 'ac':
		y = F.log_softmax(y, 1)
		loss = - y[:, -1]
	elif mode == 'ent':
		logp = F.log_softmax(y, 1)
		p = F.softmax(y, 1)
		loss = torch.sum(p * logp, 1)
	elif mode == 'uni':
		logp = F.log_softmax(y, 1)
		t = y.new_full(y.size(), 1.0/y.size(1))
		loss = - torch.sum(t * logp, 1)
	elif mode == 'src':
		# if from_style.size(0) != y.size(0):
			# from_style = repeat_tensors(y.size(0)//from_style.size(0), 0, from_style)
		loss = F.cross_entropy(y, from_style, reduction='none')
	elif mode == 'mtsf':
		loss = - F.cross_entropy(y, to_style, reduction='none')
	else:
		raise ValueError('Unsupported adversarial loss mode!')

	return loss.mean() if size_average else loss.sum()

def get_style_reward(y, t, binary, smooth=1):
	if binary:
		y = torch.argmax(y, 1)
		reward = (y == t).float()
		if smooth < 1:
			zeros = reward == 0
			reward[zeros] = 1 - smooth
			reward[1-zeros] = smooth
		return reward
	else:
		y = F.softmax(y, dim=1)
		return torch.gather(y, 1, t.unsqueeze(-1)).squeeze(-1)

def get_content_reward(y, t, t_lens, mask, y_bs, t_bs, t_lens_bs, mask_bs, beam_size):
	batch_size = t_lens.size(0) // beam_size
	y_loss = seq_ce_logits_loss(y, t, t_lens, mask, None, reduction=False).view(beam_size, -1)
	if y_bs is None:
		y_bs_loss = y_loss.mean(0)
	else:
		y_bs_loss = seq_ce_logits_loss(y_bs, t_bs, t_lens_bs, mask_bs, None, reduction=False)
	reward = y_bs_loss - y_loss
	reward = torch.sigmoid(reward)
	return reward.view(-1)

def get_consistency_reward(y, y_logits, t, binary, smooth, diff, logprob, sigmoid):

	if binary:
		reward = (y == t).float()
		if smooth < 1:
			zeros = reward == 0
			reward[zeros] = 1 - smooth
			reward[1-zeros] = smooth
		return reward
	else:
		y_probs = F.log_softmax(y_logits, dim=2) if logprob else F.softmax(y_logits, dim=2)
		reward = torch.gather(y_probs, 2, t.unsqueeze(-1)).squeeze(-1)
		if diff:
			reward = reward - torch.gather(y_probs, 2, y.unsqueeze(-1)).squeeze(-1)
		if sigmoid:
			reward = torch.sigmoid(reward)
		return reward

def word_level_ent_top5(y, t_lens, mask, size_average, ret_mats=False):
	y = y.detach()
	py = F.softmax(y, 2)
	logpy = F.log_softmax(y, 2)
	ent = py * logpy
	ent = - ent.sum(2)
	ent_mat = ent

	effect_lens = t_lens.masked_fill(t_lens==0, 1).float()

	ent = ent.masked_fill(mask, 0)
	ent = ent.sum(0)/effect_lens
	ent = ent.mean() if size_average else ent.sum()

	top_probs, top_inds = torch.topk(py, 5, 2)
	probs_mat = top_probs
	top_probs = top_probs.masked_fill(mask.unsqueeze(-1), 0)
	top_probs = top_probs.sum(0) / effect_lens.unsqueeze(-1)
	top_probs = top_probs.mean(0) if size_average else top_probs.sum(0)
	if ret_mats:
		return ent, top_probs[0].item(), top_probs[1].item(), top_probs[2].item(), top_probs[3].item(), top_probs[4].item(), ent_mat, probs_mat, top_inds
	else:
		return ent, top_probs[0].item(), top_probs[1].item(), top_probs[2].item(), top_probs[3].item(), top_probs[4].item()

def diversity(out1, out2, lens1, lens2):
	out1 = out1.t().tolist()
	out2 = out2.t().tolist()
	for i in range(len(out1)):
		out1[i] = out1[i][:lens1[i].item()]
		out2[i] = out2[i][:lens2[i].item()]
	bleu = get_bleu(out1, out2)
	return bleu

def compute_bleu_score(labels_files, predictions_path):

	if not isinstance(labels_files, list):
		labels_files = [labels_files]

	try:
		cmd = 'perl %s %s < %s' % ('./multi-bleu.perl',
								   " ".join(labels_files),
								   predictions_path)
		bleu_out = subprocess.check_output(
			cmd,
			stderr=subprocess.STDOUT,
			shell=True)
		bleu_out = bleu_out.decode("utf-8")
		bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
		return float(bleu_score)
	except subprocess.CalledProcessError as error:
		if error.output is not None:
			msg = error.output.strip()
			print("bleu script returned non-zero exit code: {}".format(msg))
		return 0

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# BLEU functions from https://github.com/MaximumEntropy/Seq2Seq-PyTorch
#    (ran some comparisons, and it matches moses's multi-bleu.perl)
def bleu_stats(hypothesis, reference):
	"""Compute statistics for BLEU."""
	stats = []
	stats.append(len(hypothesis))
	stats.append(len(reference))
	for n in range(1, 5):
		s_ngrams = Counter(
			[tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
		)
		r_ngrams = Counter(
			[tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
		)
		stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
		stats.append(max([len(hypothesis) + 1 - n, 0]))
	return stats

def bleu(stats):
	"""Compute BLEU given n-gram statistics."""
	if len(list(filter(lambda x: x == 0, stats))) > 0:
		return 0
	(c, r) = stats[:2]
	log_bleu_prec = sum(
		[math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
	) / 4.
	return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)

def get_bleu(hypotheses, references):
	"""Get validation BLEU score for dev set."""
	stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
	for hyp, ref in zip(hypotheses, references):
		stats += np.array(bleu_stats(hyp, ref))
	return 100 * bleu(stats)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #