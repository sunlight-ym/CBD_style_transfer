import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import datalib.constants as constants
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as pad
# import spacy
# from spacy.tokens import Doc

def acc_steps(result, step_result):
	for k in result:
		result[k].append(step_result[k])
def acc_steps_tensor(result, step_result, ind):
	for k in result:
		result[k][ind] = step_result[k]
def steps_to_tensor(result):
	for k in result:
		result[k] = torch.stack(result[k])
def init_dict(result, key, cond=True):
	if cond:
		result[key] = []
def _repeat(num, batch_dim, x):
	if x is None:
		return x
	shape = list(x.size())
	x = x.unsqueeze(batch_dim+1)
	new_shape = list(x.size())
	new_shape[batch_dim+1] = num
	x = x.expand(*new_shape).contiguous()
	shape[batch_dim] = shape[batch_dim]*num
	return x.view(shape)

def repeat_tensors(num, batch_dim, mats):
	if num == 1 or (mats is None):
		return mats
	elif torch.is_tensor(mats):
		return _repeat(num, batch_dim, mats)
	else:
		return (_repeat(num, batch_dim, m) for m in mats)



def softmax_sample(logits, tau = 1, gumbel = True):
	tau = max(tau, 1e-10)
	if gumbel:
		return F.gumbel_softmax(logits, tau=tau)
	else:
		return F.softmax(logits / tau, 1)


def init_rnn_hidden(batch_size, n_layers, n_directions, hid_size, rnn_type, input_tensor):
	size=(n_layers*n_directions, batch_size, hid_size)
	if rnn_type == 'LSTM':
		return input_tensor.new_zeros(size), input_tensor.new_zeros(size)
	else:
		return input_tensor.new_zeros(size)





def get_out_lens(x, seq_dim=0, exceed_mask=None, return_with_eos=True):
	eos_mask = (x == constants.EOS_ID)
	if exceed_mask is not None:
		eos_mask = eos_mask | exceed_mask
	ret_eos_mask = eos_mask
	eos_mask = eos_mask.int()
	mask_sum = eos_mask.cumsum(seq_dim)
	# eos_mask = eos_mask.masked_fill_(mask_sum != 1, 0)
	# lens = eos_mask.argmax(seq_dim)
	lens = torch.sum((mask_sum == 0).long(), seq_dim)
	# if count_eos:
	if not return_with_eos:
		return lens, ret_eos_mask
	lens_with_eos = lens + 1
	lens_with_eos.clamp_(max=x.size(seq_dim))
	# zl_mask = eos_mask.sum(seq_dim) == 0
	# lens[zl_mask] = x.size(seq_dim)
	return lens, lens_with_eos, ret_eos_mask




def get_pad_size(mode, k_size):
	if mode is None:
		return 0
	elif mode=='half':
		return (int((k_size-1)/2), 0)
	elif mode=='full':
		return (k_size-1, 0)
	else:
		raise ValueError('unsupported padding mode')

def get_padding_mask(x, lens, batch_dim=1, seq_dim=0):
	max_len = x.size(seq_dim)
	mask = torch.arange(0, max_len, dtype = torch.long, device = x.device).unsqueeze(batch_dim)#.expand(max_len, lens.size(0))
	mask = mask >= lens.unsqueeze(seq_dim)
	return mask

def get_padding_masks(x, lens, lens_with_eos, batch_dim=1, seq_dim=0):
	max_len = x.size(seq_dim)
	inds = torch.arange(0, max_len, dtype = torch.long, device = x.device).unsqueeze(batch_dim)#.expand(max_len, lens.size(0))
	mask = inds >= lens.unsqueeze(seq_dim)
	mask_with_eos = inds >= lens_with_eos.unsqueeze(seq_dim)
	return mask, mask_with_eos

def reverse_seq(x, lens, eos_mask, batch_dim=1, seq_dim=0):
	#lens should be without eos
	inds = lens.unsqueeze(seq_dim)
	inds = inds - torch.arange(1, x.size(seq_dim)+1, dtype = torch.long, device = x.device).unsqueeze(batch_dim)
	inds.masked_fill_(inds < 0, 0)
	rev_x = x.gather(seq_dim, inds)
	rev_x = rev_x.masked_fill(eos_mask, constants.EOS_ID)
	return rev_x

def reverse_seq_feature(x, lens, eos_mask, batch_dim=1, seq_dim=0):
	inds = lens.unsqueeze(seq_dim)
	inds = inds - torch.arange(1, x.size(seq_dim)+1, dtype = torch.long, device = x.device).unsqueeze(batch_dim)
	inds.masked_fill_(inds < 0, 0)
	rev_x = x.gather(seq_dim, inds.unsqueeze(-1).expand(inds.size(0), inds.size(1), x.size(2)))
	eos_feature = x.masked_select(eos_mask.unsqueeze(-1))
	rev_x = rev_x.masked_scatter(eos_mask.unsqueeze(-1), eos_feature)
	return rev_x

def reverse_seq_value(x, lens, eos_mask, batch_dim=1, seq_dim=0):
	inds = lens.unsqueeze(seq_dim)
	inds = inds - torch.arange(1, x.size(seq_dim)+1, dtype = torch.long, device = x.device).unsqueeze(batch_dim)
	inds.masked_fill_(inds < 0, 0)
	rev_x = x.gather(seq_dim, inds)
	eos_value = x.masked_select(eos_mask)
	rev_x = rev_x.masked_scatter(eos_mask, eos_value)
	return rev_x
		
def conv_mask(x, conv_pad_num, padding_mask):
	if conv_pad_num > 0:
		prefix = padding_mask.new_full((x.size(0), conv_pad_num), False)
		padding_mask = torch.cat([prefix, padding_mask], 1)
	mask = padding_mask[:,:x.size(2)]
	mask = mask.unsqueeze(1)
	low_value = x.detach().min() - 1
	# print('x', x.size())
	# print('low_values', low_values.size())
	# print('mask', mask.size())
	# low_values = low_values.expand_as(x).masked_select(mask)
	return x.masked_fill(mask, low_value)

class cnn(nn.Module):
	
	def __init__(self, input_size, filter_sizes, n_filters, leaky, pad, dropout_rate, output_size):
		super(cnn, self).__init__()

		self.conv_pad_nums = [w-1 if pad else 0 for w in filter_sizes]

		self.convs = nn.ModuleList([
			nn.Conv2d(1, n_filters, (w, input_size), padding = (self.conv_pad_nums[i], 0)) 
			for i, w in enumerate(filter_sizes)])
		self.leaky = leaky
		self.dropout = nn.Dropout(dropout_rate)
		self.linear = nn.Linear(len(filter_sizes)*n_filters, output_size)
	
	def forward(self, x, padding_mask):
		# x is the embedding matrix: seq * batch * emb_size
		
		x = x.unsqueeze(1)
		conv_outs=[]
		for i, conv in enumerate(self.convs):
			conv_out = conv(x)
			conv_out = F.leaky_relu(conv_out) if self.leaky else F.relu(conv_out)
			conv_out = conv_out.squeeze(-1)
			if padding_mask is not None:
				conv_out = conv_mask(conv_out, self.conv_pad_nums[i], padding_mask)
			conv_out = conv_out.max(2)[0]
			conv_outs.append(conv_out)
		conv_outs = torch.cat(conv_outs, 1)
		conv_outs = self.dropout(conv_outs)
		logits = self.linear(conv_outs)

		return logits

class feedforward_attention(nn.Module):
	"""docstring for feedforward_attention"""
	def __init__(self, input_size, hid_size, att_size, self_att = False, use_coverage = False, eps = 1e-10):
		super(feedforward_attention, self).__init__()
		self.input_proj = nn.Linear(input_size, att_size, bias = self_att)
		self.self_att = self_att
		if not self_att:
			self.hid_proj = nn.Linear(hid_size, att_size)
		self.use_coverage = use_coverage
		if use_coverage:
			self.cov_proj = nn.Linear(1, att_size)
		self.v = nn.Linear(att_size, 1, bias = False)
		self.eps = eps
	
	def forward(self, enc_outputs, enc_padding_mask, hidden = None, coverage = None):
		att_fea = self.input_proj(enc_outputs) # seq * b * att_size
		if not self.self_att:
			hid_fea = self.hid_proj(hidden) # b * att_size
			att_fea = att_fea + hid_fea.unsqueeze(0)
		if self.use_coverage:
			cov_fea = self.cov_proj(coverage.unsqueeze(-1))
			att_fea = att_fea + cov_fea

		scores = self.v(torch.tanh(att_fea)).squeeze(-1) # seq * b

		attn_dist = F.softmax(scores, dim=0)
		if enc_padding_mask is not None:
			attn_dist = attn_dist.masked_fill(enc_padding_mask, 0)
			normalization_factor = attn_dist.sum(0, keepdim = True)
			attn_dist = attn_dist / (normalization_factor + self.eps)

		return attn_dist

class bilinear_attention(nn.Module):
	"""docstring for bilinear_attention"""
	def __init__(self, input_size, hid_size, eps = 1e-10):
		super(bilinear_attention, self).__init__()
		self.proj = nn.Linear(hid_size, input_size, bias=False)
		self.eps = eps
	
	def forward(self, enc_outputs, enc_padding_mask, hidden, dummy = None):
		hid_fea = self.proj(hidden).unsqueeze(0)
		scores = torch.sum(hid_fea * enc_outputs, -1)

		attn_dist = F.softmax(scores, dim=0)
		if enc_padding_mask is not None:
			attn_dist = attn_dist.masked_fill(enc_padding_mask, 0)
			normalization_factor = attn_dist.sum(0, keepdim = True)
			attn_dist = attn_dist / (normalization_factor + self.eps)
		
		return attn_dist

class attn_decoder(nn.Module):
	"""docstring for attn_decoder"""
	def __init__(self, emb_size, hid_size, num_layers, rnn_type, bilin_att, enc_size, feed_last_context, att_coverage = False, eps = 1e-10, use_att=True, use_copy=False, cxt_drop=0):
		super(attn_decoder, self).__init__()
		self.rnn_type = rnn_type
		self.use_att = use_att
		self.use_copy = use_copy
		self.feed_last_context = feed_last_context
		self.cxt_drop = cxt_drop
		if att_coverage:
			assert not bilin_att, 'bilin_att does not support using coverage!'
		self.rnn = getattr(nn, rnn_type)(emb_size+enc_size if feed_last_context else emb_size, hid_size, num_layers = num_layers)
		if use_att:
			self.attention = bilinear_attention(enc_size, hid_size, eps=eps) if bilin_att else feedforward_attention(enc_size, hid_size, hid_size, use_coverage=att_coverage, eps=eps)
			if use_copy:
				self.copy = nn.Linear((enc_size * 2 if feed_last_context else enc_size) + hid_size + emb_size, 1)
		# self.out1 = nn.Linear(enc_size + hid_size, hid_size)
		# self.out2 = nn.Linear(hid_size, vocab_size)

	def forward(self, input_emb, last_state, enc_outputs, enc_padding_mask, last_context, coverage = None):
		if self.feed_last_context:
			input_emb = torch.cat([last_context, input_emb], 1)
		self.rnn.flatten_parameters()
		dec_out, state = self.rnn(input_emb.unsqueeze(0), last_state)
		if self.use_att:
			att_hid = state[0][-1] if self.rnn_type == 'LSTM' else state[-1]
			attn_dist = self.attention(enc_outputs, enc_padding_mask, att_hid, coverage)
			context = torch.sum(attn_dist.unsqueeze(-1) * enc_outputs, 0)
			if self.cxt_drop > 0:
				context = F.dropout(context, self.cxt_drop, training=self.training)

			if self.use_copy:
				p_copy_input = torch.cat([context, att_hid, input_emb], 1)
				p_copy = self.copy(p_copy_input)
				p_copy = torch.sigmoid(p_copy.squeeze(-1))
		
		if not self.use_att:
			attn_dist, context, p_copy = None, None, None
		if self.use_att and not self.use_copy:
			p_copy = None

		# pred_input = torch.cat([context, att_hid], 1)
		# pred = self.out2(self.out1(pred_input))
		# vocab_dist = F.softmax(pred, dim = 1)

		return state, attn_dist, context, p_copy

class multi_bias_linear(nn.Module):
	"""docstring for multi_bias_linear"""
	def __init__(self, num_bias, input_size, output_size):
		super(multi_bias_linear, self).__init__()
		self.num_bias = num_bias
		self.input_size = input_size
		self.output_size = output_size
		if num_bias == 1:
			self.linear = nn.Linear(input_size, output_size)
		else:
			self.linear = nn.Linear(input_size, output_size, bias = False)
			self.biases = nn.Parameter(torch.Tensor(num_bias, output_size))
			self.init_bias()
	def init_bias(self):
		fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.linear.weight)
		bound = 1 / math.sqrt(fan_in)
		nn.init.uniform_(self.biases, -bound, bound)

	def forward(self, x, inds=None, soft_inds=False):
		x = self.linear(x)
		if self.num_bias == 1:
			return x
		elif soft_inds:
			return x + torch.mm(inds, self.biases)
		else:
			return x + self.biases[inds]

class PositionalEncoding(nn.Module):
	r"""Inject some information about the relative or absolute position of the tokens
		in the sequence. The positional encodings have the same dimension as
		the embeddings, so that the two can be summed. Here, we use sine and cosine
		functions of different frequencies.
	.. math::
		\text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
		\text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
		\text{where pos is the word position and i is the embed idx)
	Args:
		d_model: the embed dim (required).
		dropout: the dropout value (default=0.1).
		max_len: the max. length of the incoming sequence (default=5000).
	Examples:
		>>> pos_encoder = PositionalEncoding(d_model)
	"""

	def __init__(self, d_model, dropout=0.1, max_len=50):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / (d_model-2)))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x, j=None):
		r"""Inputs of forward function
		Args:
			x: the sequence fed to the positional encoder model (required).
		Shape:
			x: [sequence length, batch size, embed dim]
			output: [sequence length, batch size, embed dim]
		Examples:
			>>> output = pos_encoder(x)
		"""
		if j is None:
			assert x.dim() == 3
			x = x + self.pe[:x.size(0), :]
		else:
			assert x.dim() == 2
			x = x + self.pe[j, :]
		return self.dropout(x)
def generate_square_subsequent_mask(sz, device):
	r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
		Unmasked positions are filled with float(0.0).
	"""
	mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
	mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
	return mask