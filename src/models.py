import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as pad
from layers import *
import datalib.constants as constants
from train_utils import check_values


class language_model(nn.Module):
	"""docstring for language_model"""
	def __init__(self, vocab_size, emb_size, emb_max_norm, hid_size, rnn_type, dropout_rate, num_bias):
		super(language_model, self).__init__()
		self.vocab_size = vocab_size
		self.emb_size = emb_size
		self.hid_size = hid_size
		self.rnn_type = rnn_type
		self.emb = nn.Embedding(vocab_size, emb_size, constants.PAD_ID, max_norm = emb_max_norm)
		self.encoder = getattr(nn, rnn_type)(emb_size, hid_size)
		self.dropout = nn.Dropout(dropout_rate)
		# self.projection = nn.Linear(hid_size, vocab_size)
		self.projection = multi_bias_linear(num_bias, hid_size, vocab_size)
	
	def forward(self, x, inds=None, soft_input=False, batch_first=False):
		if batch_first:
			x = x.transpose(0, 1).contiguous()
		if soft_input:
			x = torch.matmul(x, self.emb.weight)
		else:
			x = self.emb(x)
		outputs, _ = self.encoder(x)
		outputs = self.dropout(outputs)
		logits = self.projection(outputs, inds)

		return logits

class cnn_classifier(nn.Module):
	"""docstring for cnn_classifier"""
	def __init__(self, vocab_size, emb_size, emb_max_norm, filter_sizes, n_filters, leaky, pad, dropout_rate, output_size):
		super(cnn_classifier, self).__init__()
		
		self.emb = nn.Embedding(vocab_size, emb_size, constants.PAD_ID, max_norm = emb_max_norm)
		self.cnn = cnn(emb_size, filter_sizes, n_filters, leaky, pad, dropout_rate, output_size)
	def forward(self, x, padding_mask, batch_first=False, soft_input=False):
		if not batch_first:
			x = x.transpose(0, 1).contiguous()
			if padding_mask is not None:
				padding_mask = padding_mask.transpose(0, 1).contiguous()

		if soft_input:
			x = torch.matmul(x, self.emb.weight)
		else:
			x = self.emb(x)

		logits = self.cnn(x, padding_mask)
		return logits
		
class attn_classifier(nn.Module):
	"""docstring for attn_classifier"""
	def __init__(self, vocab_size, emb_size, emb_max_norm, 
		hid_size, rnn_type, bidirectional, bilin_att, self_att, dropout_rate, output_size):
		super(attn_classifier, self).__init__()
		self.hid_size = hid_size
		self.rnn_type = rnn_type
		self.bidirectional = bidirectional
		
		self.emb = nn.Embedding(vocab_size, emb_size, constants.PAD_ID, max_norm = emb_max_norm)
		
		self.encoder = getattr(nn, rnn_type)(emb_size, hid_size, bidirectional = bidirectional)
		self.self_att = self_att
		if self_att:
			assert not bilin_att, 'bilin_att does not support self attention mode!'
		rnn_size = hid_size*2 if bidirectional else hid_size
		self.attention = bilinear_attention(rnn_size, rnn_size) if bilin_att else feedforward_attention(rnn_size, rnn_size, rnn_size, self_att)
		self.dropout = nn.Dropout(dropout_rate)
		self.projection = nn.Linear(rnn_size, output_size)

	def reshape_final_state(self, final_state):
		if self.rnn_type == 'LSTM':
			final_state = final_state[0]
		if self.bidirectional:
			return final_state.transpose(0, 1).contiguous().view(-1, 2*self.hid_size)
		else:
			return final_state.squeeze(0)
	
	def forward(self, x, enc_padding_mask, batch_first=False, soft_input = False):
		if batch_first:
			x = x.transpose(0, 1).contiguous()
			if enc_padding_mask is not None:
				enc_padding_mask = enc_padding_mask.transpose(0, 1).contiguous()
		if soft_input:
			x = torch.matmul(x, self.emb.weight)
		else:
			x = self.emb(x)
			
		outputs, final_state = self.encoder(pack(x, enc_padding_mask.bitwise_not().long().sum(0), enforce_sorted=False))
		outputs = pad(outputs, total_length=x.size(0))[0]
		# check_values(outputs, 'ac outputs', False)
		# check_values(final_state, 'ac final_state', False)
		attn_dist = self.attention(outputs, enc_padding_mask, None if self.self_att else self.reshape_final_state(final_state))
		# check_values(attn_dist, 'ac attn_dist', False)
		# if att_only:
		# 	return attn_dist

		avg_state = torch.sum(attn_dist.unsqueeze(-1) * outputs, 0)
		avg_state = self.dropout(avg_state)
		logits = self.projection(avg_state)

		return logits#, attn_dist
def add_one_class(model):
	assert isinstance(model, attn_classifier) or isinstance(model, cnn_classifier)
	if isinstance(model, attn_classifier):
		device = model.projection.weight.device
		output_size, input_size = model.projection.weight.size()
		output_size = output_size + 1
		model.projection = nn.Linear(input_size, output_size).to(device)
	else:
		device = model.cnn.linear.weight.device
		output_size, input_size = model.cnn.linear.weight.size()
		output_size = output_size + 1
		model.cnn.linear = nn.Linear(input_size, output_size).to(device)

def change_output_size(model, n_targets):
	assert isinstance(model, attn_classifier) or isinstance(model, cnn_classifier)
	if isinstance(model, attn_classifier):
		device = model.projection.weight.device
		output_size, input_size = model.projection.weight.size()
		output_size = n_targets
		model.projection = nn.Linear(input_size, output_size).to(device)
	else:
		device = model.cnn.linear.weight.device
		output_size, input_size = model.cnn.linear.weight.size()
		output_size = n_targets
		model.cnn.linear = nn.Linear(input_size, output_size).to(device)

class bd_style_transfer(nn.Module):
	"""docstring for style_transfer"""
	def __init__(self, vocab_size, emb_size, emb_max_norm, 
					rnn_type, enc_hid_size, bidirectional, dec_hid_size, enc_num_layers, dec_num_layers, pooling_size,
					h_only, diff_bias, num_styles, feed_last_context, use_att, cxt_drop, dual_decoder, right1, right2):
		super(bd_style_transfer, self).__init__()
		self.vocab_size = vocab_size
		self.emb = nn.Embedding(vocab_size, emb_size, constants.PAD_ID, max_norm = emb_max_norm)
		
		self.emb_size = emb_size
		self.rnn_type = rnn_type
		self.enc_hid_size = enc_hid_size
		self.bidirectional = bidirectional
		self.enc_num_layers = enc_num_layers
		self.dec_num_layers = dec_num_layers
		self.pooling_size = pooling_size
		self.use_att = use_att
		self.dual_decoder = dual_decoder
		self.right1, self.right2 = right1, right2
		self.same_dir = right1 == right2
		self.require_rev = right2 or right1

		self.encoder = getattr(nn, rnn_type)(emb_size, enc_hid_size, num_layers = enc_num_layers, bidirectional = bidirectional)
		enc_size = enc_hid_size * 2 if bidirectional else enc_hid_size
		self.h_only = h_only
		
		self.num_styles = num_styles
		self.style_emb = nn.Embedding(num_styles, emb_size, max_norm = emb_max_norm)
		self.decoders = nn.ModuleList([attn_decoder(emb_size, dec_hid_size, dec_num_layers, rnn_type, True, enc_size, feed_last_context, use_att=use_att, cxt_drop=cxt_drop)])
		self.out_layers1 = nn.ModuleList([nn.Linear(((enc_size + dec_hid_size) if use_att else dec_hid_size), dec_hid_size)])
		self.out_layers2 = nn.ModuleList([multi_bias_linear(num_styles if diff_bias else 1, dec_hid_size, vocab_size)])
		if self.dual_decoder:
			self.decoders.append(attn_decoder(emb_size, dec_hid_size, dec_num_layers, rnn_type, True, enc_size, feed_last_context, use_att=use_att, cxt_drop=cxt_drop))
			self.out_layers1.append(nn.Linear(((enc_size + dec_hid_size) if use_att else dec_hid_size), dec_hid_size))
			self.out_layers2.append(multi_bias_linear(num_styles if diff_bias else 1, dec_hid_size, vocab_size))
		
		# self.eps = eps
		# self.gumbel = gumbel
	def reshape_final_state(self, final_state):
		final_state = final_state.view(self.enc_num_layers, 2 if self.bidirectional else 1, -1, self.enc_hid_size)[-1]
		if self.bidirectional:
			return final_state.transpose(0, 1).contiguous().view(-1, 2*self.enc_hid_size)
		else:
			return final_state.squeeze(0)

	def prepare_dec_init_state(self, enc_last_state):
		init_h = self.reshape_final_state(enc_last_state[0] if self.rnn_type=='LSTM' else enc_last_state)
		init_h = init_h.unsqueeze(0).expand(self.dec_num_layers, init_h.size(0), init_h.size(1)).contiguous()
		if self.rnn_type=='LSTM' and not self.h_only:
			init_c = self.reshape_final_state(enc_last_state[1])
			init_c = init_c.unsqueeze(0).expand(self.dec_num_layers, init_h.size(1), init_h.size(2)).contiguous()
			return (init_h, init_c)
		elif self.rnn_type=='GRU':
			return init_h
		else:
			return (init_h, torch.zeros_like(init_h))

	def get_target_style(self, cur_style):
		if self.num_styles == 2:
			return 1 - cur_style
		else:
			probs = cur_style.new_ones((cur_style.size(0), self.num_styles), dtype = torch.float).scatter_(1, cur_style.view(-1, 1), 0)
			return torch.multinomial(probs, 1).view(-1)

	def encode(self, x, lens, enc_padding_mask, total_length=None, is_sorted=True):
		self.encoder.flatten_parameters()
		if x.dtype == torch.long:
			x = self.emb(x)
		else:
			vocab_vector = torch.arange(0, self.vocab_size, dtype=torch.long, device=x.device)
			x = torch.matmul(x, self.emb(vocab_vector))

		enc_outputs, final_state = self.encoder(pack(x, lens, enforce_sorted=is_sorted))
		enc_outputs = pad(enc_outputs, total_length=total_length)[0]
		if self.pooling_size > 1:
			enc_outputs_chunks = enc_outputs.split(self.pooling_size, dim=0)
			enc_padding_mask_chunks = enc_padding_mask.bitwise_not().float().split(self.pooling_size, dim=0)
			enc_outputs_chunks = [c.sum(0, keepdim=True) for c in enc_outputs_chunks]
			enc_padding_mask_chunks = [c.sum(0, keepdim=True) for c in enc_padding_mask_chunks]
			enc_outputs = torch.cat(enc_outputs_chunks, 0)
			enc_padding_mask = torch.cat(enc_padding_mask_chunks, 0)
			enc_outputs = enc_outputs / torch.clamp(enc_padding_mask.unsqueeze(-1), min=1e-10)
			enc_padding_mask = enc_padding_mask == 0

		return enc_outputs, final_state, enc_padding_mask

	def predict(self, module_ind, input, enc_outputs, enc_padding_mask, last_dec_state, style_inds, last_context, require_outputs=False, tau=None, greedy=None):
		result = {}
		decoder = self.decoders[module_ind]
		# decoder.flatten_parameters()
		out1, out2 = self.out_layers1[module_ind], self.out_layers2[module_ind]
		last_dec_state, attn_dist, last_context, _ = decoder(input, last_dec_state, enc_outputs, enc_padding_mask, last_context)
		# if self.use_att:
		# 	result['attn_dists'] = attn_dist

		dec_h = last_dec_state[0][-1] if self.rnn_type == 'LSTM' else last_dec_state[-1]
		if not self.use_att:
			logit = out2(out1(dec_h), style_inds)
			# prob = F.softmax(logit, dim = 1)
		else:
			logit = out2(out1(torch.cat([last_context, dec_h], 1)), style_inds)
			# prob = F.softmax(logit, dim = 1)
		# logit[:, constants.UNK_ID] = -math.inf
		result['logits'] = logit
		if require_outputs:
			result['soft_outputs'] = softmax_sample(logit, tau, not greedy)
			result['hard_outputs'] = torch.argmax(result['soft_outputs'], 1)
		return last_dec_state, last_context, result
	
	def prepare_decode(self, to_style, final_state, enc_outputs):
		style_emb = self.style_emb(to_style)
		last_dec_state = self.prepare_dec_init_state(final_state)
		last_context = torch.zeros_like(enc_outputs[0])
		return style_emb, last_dec_state, last_context

	def para_decode(self, module_ind, y, to_style, enc_result, soft_input=False):
		enc_outputs, enc_padding_mask, style_emb, last_dec_state, last_context = enc_result
		dec_len = y.size(0) + 1
		result={}
		# init_dict(result, 'logits')
		result['logits'] = torch.zeros(dec_len, y.size(1), self.vocab_size, device=to_style.device)
		# init_dict(result, 'attn_dists', self.use_att)

		# n_samples = y.size(1)//style_emb.size(0)
		# enc_outputs, enc_padding_mask, last_dec_state = repeat_tensors(n_samples, 1, (enc_outputs, enc_padding_mask, last_dec_state))
		# style_emb, to_style, last_context = repeat_tensors(n_samples, 0, (style_emb, to_style, last_context))
		if soft_input:
			vocab_vector = torch.arange(0, self.vocab_size, dtype=torch.long, device=y.device)
			dec_emb = torch.matmul(y, self.emb(vocab_vector))
		else:
			dec_emb = self.emb(y)
		dec_emb = torch.cat([style_emb.unsqueeze(0), dec_emb], 0)
		
		for j in range(dec_len):
			
			last_dec_state, last_context, step_result = self.predict(module_ind, dec_emb[j], enc_outputs, enc_padding_mask, 
																				last_dec_state, to_style, last_context)
			acc_steps_tensor(result, step_result, j)
		# steps_to_tensor(result)

		return result

	def mono_decode(self, module_ind, lens, to_style, tau, greedy, extra_len, enc_result):
		enc_outputs, enc_padding_mask, style_emb, last_dec_state, last_context = enc_result
		# if greedy:
		# 	assert n_samples == 1, 'only 1 sample can be generated for the greedy mode!'
		
		result={}
		# init_dict(result, 'logits')
		# # init_dict(result, 'attn_dists', self.use_att)
		# init_dict(result, 'soft_outputs')
		# init_dict(result, 'hard_outputs')
		dec_len = lens.max().item() + extra_len + 1
		result['logits'] = torch.zeros(dec_len, to_style.size(0), self.vocab_size, device=to_style.device)
		result['soft_outputs'] = torch.zeros(dec_len, to_style.size(0), self.vocab_size, device=to_style.device)
		result['hard_outputs'] = torch.zeros(dec_len, to_style.size(0), dtype=torch.long, device=to_style.device)
		
		next_emb = style_emb
		# enc_outputs_tile, enc_padding_mask_tile, last_dec_state_tile = repeat_tensors(n_samples, 1, (enc_outputs, enc_padding_mask, last_dec_state))
		# next_emb_tile, to_style_tile, last_context_tile = repeat_tensors(n_samples, 0, (next_emb, to_style, last_context))
		
		for j in range(dec_len):
			last_dec_state, last_context, step_result = self.predict(module_ind, next_emb, enc_outputs, enc_padding_mask, 
																				last_dec_state, to_style, last_context,
																				True, tau, greedy)
			# soft_output = softmax_sample(step_result['logits'], tau, not greedy)
			# hard_output = torch.argmax(soft_output, 1)
			next_emb = self.emb(step_result['hard_outputs'])
			# step_result['soft_outputs'] = soft_output
			# step_result['hard_outputs'] = hard_output
			acc_steps_tensor(result, step_result, j)
		# steps_to_tensor(result)

		
		inds = torch.arange(0, dec_len, dtype=torch.long, device=style_emb.device).unsqueeze(-1)
		exceed_mask = inds >= (lens.unsqueeze(0) + extra_len)
		result['hard_outputs_lens'], result['hard_outputs_lens_with_eos'], result['eos_mask'] = get_out_lens(result['hard_outputs'], exceed_mask=exceed_mask)
		result['hard_outputs_padding_mask'], result['hard_outputs_padding_mask_with_eos'] = get_padding_masks(result['hard_outputs'], result['hard_outputs_lens'], result['hard_outputs_lens_with_eos'])
		result['hard_outputs'] = result['hard_outputs'].masked_fill(result['hard_outputs_padding_mask_with_eos'], constants.PAD_ID)
		result['soft_outputs'] = result['soft_outputs'].masked_fill(result['hard_outputs_padding_mask_with_eos'].unsqueeze(-1), 0)
		result['logits'] = result['logits'].masked_fill(result['hard_outputs_padding_mask_with_eos'].unsqueeze(-1), 0)
		if self.require_rev:
			result['hard_outputs_rev'] = reverse_seq(result['hard_outputs'], result['hard_outputs_lens'], result['eos_mask'])
			result['logits_rev'] = reverse_seq_feature(result['logits'], result['hard_outputs_lens'], result['eos_mask'])
			result['soft_outputs_rev'] = reverse_seq_feature(result['soft_outputs'], result['hard_outputs_lens'], result['eos_mask'])

		result['hard_output_zl_mask'] = result['hard_outputs_lens'] == 0
		# result['to_style_tile'] = to_style_tile
		# result['enc_padding_mask_tile'] = enc_padding_mask_tile
		return result

	def para_transfer(self, x1, lens1, enc_padding_mask1, y1, x2, lens2, enc_padding_mask2, y2, to_style, require_sort=False):
		
		enc_outputs1, final_state1, enc_padding_mask1 = self.encode(x1, lens1.masked_fill(lens1==0, 1), enc_padding_mask1, total_length=x1.size(0), is_sorted=not require_sort)
		style_emb, last_dec_state1, last_context = self.prepare_decode(to_style, final_state1, enc_outputs1)
		enc_result1 = (enc_outputs1, enc_padding_mask1, style_emb, last_dec_state1, last_context)
		result = {}
		result['r1'] = self.para_decode(0, y1, to_style, enc_result1)
		if self.dual_decoder:
			if x2 is None:
				enc_result2 = enc_result1
			else:
				enc_outputs2, final_state2, enc_padding_mask2 = self.encode(x2, lens2.masked_fill(lens2==0, 1), enc_padding_mask2, total_length=x2.size(0), is_sorted=not require_sort)
				last_dec_state2 = self.prepare_dec_init_state(final_state2)
				enc_result2 = (enc_outputs2, enc_padding_mask2, style_emb, last_dec_state2, last_context)
			result['r2'] = self.para_decode(1, y2, to_style, enc_result2)
		
		return result

	def mono_transfer(self, x, x1, x2, lens, enc_padding_mask, with_bt, with_cs, with_cd, model_bf, from_style, tau, greedy, n_samples, extra_len, bt_cross, bt_sg, csd_detach_enc):
		enc_outputs, final_state, enc_padding_mask = self.encode(x, lens, enc_padding_mask, total_length=x.size(0))
		to_style = self.get_target_style(from_style)
		style_emb, last_dec_state, last_context = self.prepare_decode(to_style, final_state, enc_outputs)
		if greedy:
			n_samples = 1
		if n_samples > 1:
			enc_outputs, enc_padding_mask, last_dec_state = repeat_tensors(n_samples, 1, (enc_outputs, enc_padding_mask, last_dec_state))
			style_emb, to_style, last_context = repeat_tensors(n_samples, 0, (style_emb, to_style, last_context))
		enc_result = (enc_outputs, enc_padding_mask, style_emb, last_dec_state, last_context)
		result = {}
		result['to_style'] = to_style
		result['fw1'] = self.mono_decode(0, lens, to_style, tau, greedy, extra_len, enc_result)
		if self.dual_decoder:
			result['fw2'] = self.mono_decode(1, lens, to_style, tau, greedy, extra_len, enc_result)


		if self.dual_decoder and (with_cs or with_cd):
			if csd_detach_enc:
				enc_result = (enc_outputs.detach(), enc_padding_mask, style_emb.detach(), last_dec_state.detach(), last_context)
			if with_cs:
				result['cs1'] = self.para_decode(0, result['fw2']['hard_outputs' if self.same_dir else 'hard_outputs_rev'], to_style, enc_result)
				result['cs2'] = self.para_decode(1, result['fw1']['hard_outputs' if self.same_dir else 'hard_outputs_rev'], to_style, enc_result)
			if with_cd:
				result['cd1'] = model_bf.para_decode(0, result['fw2']['soft_outputs' if self.same_dir else 'soft_outputs_rev'], to_style, enc_result, True)
				result['cd2'] = model_bf.para_decode(1, result['fw1']['soft_outputs' if self.same_dir else 'soft_outputs_rev'], to_style, enc_result, True)

		
		if with_bt:
			if n_samples > 1:
				x1, x2 = repeat_tensors(n_samples, 1, (x1, x2))
				from_style = repeat_tensors(n_samples, 0, from_style)
			# x1 = right_x if self.right1 else x
			# if self.dual_decoder:
			# 	x2 = right_x if self.right2 else x

			if self.dual_decoder:
				inp1, inp2 = (result['fw2'], result['fw1']) if bt_cross else (result['fw1'], result['fw2'])
				rev1 = self.right2 if bt_cross else self.right1
				rev2 = self.right1 if bt_cross else self.right2
			else:
				inp1 = result['fw1']
				rev1 = self.right1
			key_name = 'hard_outputs' if bt_sg else 'soft_outputs'
			bt_result = self.para_transfer(inp1[f'{key_name}_rev' if rev1 else f'{key_name}'], inp1['hard_outputs_lens'], 
				inp1['hard_outputs_padding_mask'], x1, 
				inp2[f'{key_name}_rev' if rev2 else f'{key_name}'] if self.dual_decoder else None, 
				inp2['hard_outputs_lens'] if self.dual_decoder else None, 
				inp2['hard_outputs_padding_mask'] if self.dual_decoder else None, 
				x2 if self.dual_decoder else None, from_style, True)
			result['bw1'] = bt_result['r1']
			if self.dual_decoder:
				result['bw2'] = bt_result['r2']
				
		return result

	def mono_beam_search(self, x, lens, enc_padding_mask, from_style, tau, beam_decoder):
		enc_outputs, final_state, enc_padding_mask = self.encode(x, lens, enc_padding_mask, total_length=x.size(0))
		to_style = self.get_target_style(from_style)
		style_emb, last_dec_state, last_context = self.prepare_decode(to_style, final_state, enc_outputs)
		# enc_result = (enc_outputs, enc_padding_mask, style_emb, last_dec_state, last_context)
		result = {}
		result['to_style'] = to_style
		result['fw1'] = beam_decoder.generate(False, self, 0, tau, to_style, enc_outputs, lens, enc_padding_mask, style_emb, last_dec_state, last_context)
		if self.dual_decoder:
			result['fw2'] = beam_decoder.generate(False, self, 1, tau, to_style, enc_outputs, lens, enc_padding_mask, style_emb, last_dec_state, last_context)
				
		return result

class bd_style_transfer_transformer(nn.Module):
	"""docstring for style_transfer"""
	def __init__(self, vocab_size, emb_size, emb_max_norm, dropout_rate,
					num_heads, hid_size, num_layers, subseq_mask,
					diff_bias, num_styles, dual_decoder, right1, right2):
		super(bd_style_transfer_transformer, self).__init__()
		self.vocab_size = vocab_size
		self.dual_decoder = dual_decoder
		self.right1, self.right2 = right1, right2
		self.same_dir = right1 == right2
		self.require_rev = right2 or right1
		self.subseq_mask = subseq_mask
		
		self.emb = nn.Embedding(vocab_size, emb_size, constants.PAD_ID, max_norm = emb_max_norm)
		self.pos_emb = PositionalEncoding(emb_size, dropout_rate)
		
		self.emb_size = emb_size
		self.hid_size = hid_size
		self.num_heads = num_heads
		self.num_layers = num_layers
		encoder_layer = nn.TransformerEncoderLayer(emb_size, num_heads, hid_size, dropout_rate)
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
		
		self.num_styles = num_styles
		self.style_emb = nn.Embedding(num_styles, emb_size, max_norm = emb_max_norm)
		left_decoder_layer = nn.TransformerDecoderLayer(emb_size, num_heads, hid_size, dropout_rate)
		
		self.decoders = nn.ModuleList([nn.TransformerDecoder(left_decoder_layer, num_layers)])
		self.out_layers = nn.ModuleList([multi_bias_linear(num_styles if diff_bias else 1, emb_size, vocab_size)])
		if self.dual_decoder:
			right_decoder_layer = nn.TransformerDecoderLayer(emb_size, num_heads, hid_size, dropout_rate)
			self.decoders.append(nn.TransformerDecoder(right_decoder_layer, num_layers))
			self.out_layers.append(multi_bias_linear(num_styles if diff_bias else 1, emb_size, vocab_size))
		
		# self.eps = eps
		# self.gumbel = gumbel

	def get_target_style(self, cur_style):
		if self.num_styles == 2:
			return 1 - cur_style
		else:
			probs = cur_style.new_ones((cur_style.size(0), self.num_styles), dtype = torch.float).scatter_(1, cur_style.view(-1, 1), 0)
			return torch.multinomial(probs, 1).view(-1)

	def embed(self, x, soft_input=False, j=None):
		if soft_input:
			vocab_vector = torch.arange(0, self.vocab_size, dtype=torch.long, device=x.device)
			x = torch.matmul(x, self.emb(vocab_vector))
		else:
			x = self.emb(x) 
		x = x * math.sqrt(self.emb_size)
		x = self.pos_emb(x, j)
		return x

	def encode(self, x, enc_padding_mask):
		x = self.embed(x, soft_input=(not (x.dtype==torch.long)))
		# check_values(x, 'emb', False)
		enc_outputs = self.encoder(x, src_key_padding_mask=enc_padding_mask)
		# check_values(enc_outputs, 'enc_outputs', False)
		return enc_outputs

	def seq_predict(self, module_ind, input, enc_outputs, enc_padding_mask, dec_padding_mask, style_inds):
		result = {}
		decoder = self.decoders[module_ind]
		out = self.out_layers[module_ind]
		tgt_mask = generate_square_subsequent_mask(input.size(0), input.device)
		# check_values(input, 'input', False)
		# check_values(enc_outputs, 'enc_outputs', False)
		# check_values(tgt_mask, 'tgt_mask', False)
		# check_values(dec_padding_mask, 'dec_padding_mask', False)
		# check_values(enc_padding_mask, 'enc_padding_mask', False)
		dec_h = decoder(input, enc_outputs, tgt_mask, tgt_key_padding_mask=dec_padding_mask, memory_key_padding_mask=enc_padding_mask)
		# check_values(dec_h, 'dec_h', False)
		logit = out(dec_h, style_inds)
		# check_values(logit, 'logit', False)
		result['logits'] = logit
		return result

	def predict(self, module_ind, input, enc_outputs, enc_padding_mask, style_inds, require_outputs=False, tau=None, greedy=None):
		result = {}
		decoder = self.decoders[module_ind]
		out = self.out_layers[module_ind]
		tgt_mask = generate_square_subsequent_mask(input.size(0), input.device) if self.subseq_mask else None
		dec_h = decoder(input, enc_outputs, tgt_mask, memory_key_padding_mask=enc_padding_mask)[-1]
		logit = out(dec_h, style_inds)	
		
		result['logits'] = logit
		if require_outputs:
			result['soft_outputs'] = softmax_sample(logit, tau, not greedy)
			result['hard_outputs'] = torch.argmax(result['soft_outputs'], 1)
		return result

	def para_decode(self, module_ind, y, to_style, enc_outputs, enc_padding_mask, dec_padding_mask, style_emb, soft_input=False):
		
		# n_samples = y.size(1)//style_emb.size(0)
		# enc_outputs, enc_padding_mask = repeat_tensors(n_samples, 1, (enc_outputs, enc_padding_mask))
		# style_emb, to_style = repeat_tensors(n_samples, 0, (style_emb, to_style))
		
		dec_emb = self.embed(y, soft_input)
		dec_emb = torch.cat([style_emb.unsqueeze(0), dec_emb], 0)
		result = self.seq_predict(module_ind, dec_emb, enc_outputs, enc_padding_mask, dec_padding_mask, to_style)

		return result

	def mono_decode(self, module_ind, lens, to_style, tau, greedy, extra_len, enc_outputs, enc_padding_mask, style_emb):
		# if greedy:
		# 	assert n_samples == 1, 'only 1 sample can be generated for the greedy mode!'
		
		result={}
		dec_len = lens.max().item() + extra_len + 1
		result['logits'] = torch.zeros(dec_len, to_style.size(0), self.vocab_size, device=to_style.device)
		result['soft_outputs'] = torch.zeros(dec_len, to_style.size(0), self.vocab_size, device=to_style.device)
		result['hard_outputs'] = torch.zeros(dec_len, to_style.size(0), dtype=torch.long, device=to_style.device)
		
		
		# enc_outputs_tile, enc_padding_mask_tile = repeat_tensors(n_samples, 1, (enc_outputs, enc_padding_mask))
		# style_emb_tile, to_style_tile = repeat_tensors(n_samples, 0, (style_emb, to_style))
		next_emb = style_emb.unsqueeze(0)
		for j in range(dec_len):
			
			step_result = self.predict(module_ind, next_emb, enc_outputs, enc_padding_mask, 
										to_style, True, tau, greedy)
			next_emb = torch.cat([next_emb, self.embed(step_result['hard_outputs'], j=j).unsqueeze(0)], 0)
			acc_steps_tensor(result, step_result, j)
			# if j < dec_len - 1:
			# 	next_emb = self.embed(result['hard_outputs'][:j+1])
			# 	next_emb = torch.cat([style_emb.unsqueeze(0), next_emb], 0)
			


		inds = torch.arange(0, dec_len, dtype=torch.long, device=style_emb.device).unsqueeze(-1)
		exceed_mask = inds >= (lens.unsqueeze(0) + extra_len)
		result['hard_outputs_lens'], result['hard_outputs_lens_with_eos'], result['eos_mask'] = get_out_lens(result['hard_outputs'], exceed_mask=exceed_mask)
		result['hard_outputs_padding_mask'], result['hard_outputs_padding_mask_with_eos'] = get_padding_masks(result['hard_outputs'], result['hard_outputs_lens'], result['hard_outputs_lens_with_eos'])
		result['hard_outputs_padding_mask_t'] = result['hard_outputs_padding_mask'].t().contiguous()
		if self.require_rev:
			result['hard_outputs_rev'] = reverse_seq(result['hard_outputs'], result['hard_outputs_lens'], result['eos_mask'])
			result['logits_rev'] = reverse_seq_feature(result['logits'], result['hard_outputs_lens'], result['eos_mask'])
			result['soft_outputs_rev'] = reverse_seq_feature(result['soft_outputs'], result['hard_outputs_lens'], result['eos_mask'])

		result['hard_output_zl_mask'] = result['hard_outputs_lens'] == 0
		result['hard_outputs_padding_mask_t'].masked_fill_(result['hard_output_zl_mask'].unsqueeze(1), False)
		# print('num of zero len:',result['hard_output_zl_mask'].sum().item())
		# result['to_style_tile'] = to_style_tile
		# result['enc_padding_mask_tile'] = enc_padding_mask_tile
		return result

	def para_transfer(self, x1, enc_padding_mask1, dec_padding_mask1, y1, x2, enc_padding_mask2, dec_padding_mask2, y2, to_style):
		# check_values(x1, 'x1', False)
		# print('x1<0:', (x1<0).sum().item(), ' x1>vocab_size:', (x1>=self.emb.num_embeddings).sum().item())
		enc_outputs1 = self.encode(x1, enc_padding_mask1)
		# check_values(enc_outputs1, 'enc_outputs1', False)
		style_emb = self.style_emb(to_style)
		result = {}
		result['r1'] = self.para_decode(0, y1, to_style, enc_outputs1, enc_padding_mask1, dec_padding_mask1, style_emb)
		if self.dual_decoder:
			if x2 is None:
				enc_outputs2, enc_padding_mask2 = enc_outputs1, enc_padding_mask1
			else:
				enc_outputs2 = self.encode(x2, enc_padding_mask2)
			result['r2'] = self.para_decode(1, y2, to_style, enc_outputs2, enc_padding_mask2, dec_padding_mask2, style_emb)

		return result

	def mono_transfer(self, x, x1, x2, lens, enc_padding_mask, dec_padding_mask, with_bt, with_cs, with_cd, model_bf, from_style, tau, greedy, n_samples, extra_len, bt_cross, bt_sg, csd_detach_enc):
		enc_outputs = self.encode(x, enc_padding_mask)
		to_style = self.get_target_style(from_style)
		style_emb = self.style_emb(to_style)
		if greedy:
			n_samples = 1
		if n_samples > 1:
			enc_outputs = repeat_tensors(n_samples, 1, enc_outputs)
			style_emb, to_style, enc_padding_mask = repeat_tensors(n_samples, 0, (style_emb, to_style, enc_padding_mask))
		result = {}
		result['to_style'] = to_style
		result['fw1'] = self.mono_decode(0, lens, to_style, tau, greedy, extra_len, enc_outputs, enc_padding_mask, style_emb)
		if self.dual_decoder:
			result['fw2'] = self.mono_decode(1, lens, to_style, tau, greedy, extra_len, enc_outputs, enc_padding_mask, style_emb)

		if self.dual_decoder and (with_cs or with_cd):
			if csd_detach_enc:
				enc_outputs, style_emb = enc_outputs.detach(), style_emb.detach()
			fw1_padding_mask = torch.cat((dec_padding_mask.new_zeros((to_style.size(0), 1)), result['fw1']['hard_outputs_padding_mask_t']), 1)
			fw2_padding_mask = torch.cat((dec_padding_mask.new_zeros((to_style.size(0), 1)), result['fw2']['hard_outputs_padding_mask_t']), 1)
			if with_cs:
				result['cs1'] = self.para_decode(0, result['fw2']['hard_outputs' if self.same_dir else 'hard_outputs_rev'], to_style, enc_outputs, 
												enc_padding_mask, fw2_padding_mask, style_emb)
				result['cs2'] = self.para_decode(1, result['fw1']['hard_outputs' if self.same_dir else 'hard_outputs_rev'], to_style, enc_outputs, 
												enc_padding_mask, fw1_padding_mask, style_emb)
			if with_cd:
				result['cd1'] = model_bf.para_decode(0, result['fw2']['soft_outputs' if self.same_dir else 'soft_outputs_rev'], to_style, enc_outputs, 
												enc_padding_mask, fw2_padding_mask, style_emb, True)
				result['cd2'] = model_bf.para_decode(1, result['fw1']['soft_outputs' if self.same_dir else 'soft_outputs_rev'], to_style, enc_outputs, 
												enc_padding_mask, fw1_padding_mask, style_emb, True)

		if with_bt:
			if n_samples > 1:
				x1, x2 = repeat_tensors(n_samples, 1, (x1, x2))
				from_style, dec_padding_mask = repeat_tensors(n_samples, 0, (from_style, dec_padding_mask))
			# x1 = right_x if self.right1 else x
			# if self.dual_decoder:
			# 	x2 = right_x if self.right2 else x

			if self.dual_decoder:
				inp1, inp2 = (result['fw2'], result['fw1']) if bt_cross else (result['fw1'], result['fw2'])
				rev1 = self.right2 if bt_cross else self.right1
				rev2 = self.right1 if bt_cross else self.right2
			else:
				inp1 = result['fw1']
				rev1 = self.right1
			key_name = 'hard_outputs' if bt_sg else 'soft_outputs'
			bt_result = self.para_transfer(inp1[f'{key_name}_rev' if rev1 else f'{key_name}'], 
				inp1['hard_outputs_padding_mask_t'], dec_padding_mask, x1, 
				inp2[f'{key_name}_rev' if rev2 else f'{key_name}'] if self.dual_decoder else None, 
				inp2['hard_outputs_padding_mask_t'] if self.dual_decoder else None, dec_padding_mask, x2 if self.dual_decoder else None,
				from_style)
			result['bw1'] = bt_result['r1']
			if self.dual_decoder:
				result['bw2'] = bt_result['r2']
				
		return result

	def mono_beam_search(self, x, lens, enc_padding_mask, from_style, tau, beam_decoder):
		enc_outputs = self.encode(x, enc_padding_mask)
		to_style = self.get_target_style(from_style)
		style_emb = self.style_emb(to_style)
		result = {}
		result['to_style'] = to_style
		result['fw1'] = beam_decoder.generate(True, self, 0, tau, to_style, enc_outputs, lens, enc_padding_mask, style_emb)
		if self.dual_decoder:
			result['fw2'] = beam_decoder.generate(True, self, 1, tau, to_style, enc_outputs, lens, enc_padding_mask, style_emb)
				
		return result