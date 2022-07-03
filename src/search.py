import math
import torch
import torch.nn.functional as F
from layers import *
import datalib.constants as constants

class BeamSearch(object):

	def __init__(self):
		self.scores_buf = None
		self.indices_buf = None
		self.beams_buf = None

	def _init_buffers(self, device):
		if self.scores_buf is None:
			self.scores_buf = torch.FloatTensor().to(device=device)
			self.indices_buf = torch.LongTensor().to(device=device)
			self.beams_buf = torch.LongTensor().to(device=device)


	def step(self, step, lprobs, scores):
		"""Take a single search step.
		Args:
			step: the current search step, starting at 0
			lprobs: (bsz x input_beam_size x vocab_size)
				the model's log-probabilities over the vocabulary at the current step
			scores: (bsz x input_beam_size x step)
				the historical model scores of each hypothesis up to this point
		Return: A tuple of (scores, indices, beams) where:
			scores: (bsz x output_beam_size)
				the scores of the chosen elements; output_beam_size can be
				larger than input_beam_size, e.g., we may return
				2*input_beam_size to account for EOS
			indices: (bsz x output_beam_size)
				the indices of the chosen elements
			beams: (bsz x output_beam_size)
				the hypothesis ids of the chosen elements, in the range [0, input_beam_size)
		"""
		self._init_buffers(lprobs.device)
		bsz, beam_size, vocab_size = lprobs.size()

		if step == 0:
			# at the first step all hypotheses are equally likely, so use
			# only the first beam
			lprobs = lprobs[:, ::beam_size, :].contiguous()
		else:
			# make probs contain cumulative scores for each hypothesis
			lprobs.add_(scores[:, :, step - 1].unsqueeze(-1))

		torch.topk(
			lprobs.view(bsz, -1),
			k=beam_size * 2,# Take the best 2 x beam_size predictions. We'll choose the first beam_size of these which don't predict eos to continue with.
			out=(self.scores_buf, self.indices_buf),
		)
		torch.div(self.indices_buf, vocab_size, out=self.beams_buf)
		self.indices_buf.fmod_(vocab_size)
		return self.scores_buf, self.indices_buf, self.beams_buf

class SequenceGenerator(object):
	def __init__(
		self,
		beam_size=1,
		max_len_b=200,
		min_len=1,
		normalize_scores=True,
		len_penalty=1.,
		unk_penalty=0.,
	):
		
		self.beam_size = beam_size
		self.max_len_b = max_len_b
		self.min_len = min_len
		self.normalize_scores = normalize_scores
		self.len_penalty = len_penalty
		self.unk_penalty = unk_penalty
		self.search = BeamSearch()

	

	@torch.no_grad()
	def generate(self, use_transformer, model, module_ind, tau, to_style, encoder_outs, src_lengths, enc_padding_mask, style_emb, last_dec_state=None, last_context=None):
		
		device = encoder_outs.device

		# batch dimension goes first followed by source lengths
		bsz = src_lengths.size(0)
		beam_size = self.beam_size

		max_len = src_lengths.max().item() + self.max_len_b

		# compute the encoder output for each beam
		result = {}
		encoder_outs, last_dec_state = repeat_tensors(beam_size, 1, (encoder_outs, last_dec_state))
		style_emb, to_style, last_context = repeat_tensors(beam_size, 0, (style_emb, to_style, last_context))
		enc_padding_mask = repeat_tensors(beam_size, 0 if use_transformer else 1, enc_padding_mask)

		next_emb = style_emb

		# initialize buffers
		scores = encoder_outs.new_zeros((bsz * beam_size, max_len + 1))
		scores_buf = scores.clone()
		tokens = torch.full((bsz * beam_size, max_len + 1), constants.PAD_ID, dtype=torch.long, device=device)
		tokens_buf = tokens.clone()
		# tokens[:, 0] = constants.EOS_ID if bos_token is None else bos_token

		# The blacklist indicates candidates that should be ignored.
		# For example, suppose we're sampling and have already finalized 2/5
		# samples. Then the blacklist would mark 2 positions as being ignored,
		# so that we only finalize the remaining 3 samples.
		blacklist = torch.zeros(bsz, beam_size, dtype=torch.bool, device=device)

		# list of completed sentences
		finalized = [[] for i in range(bsz)]
		finished = [False for i in range(bsz)]
		num_remaining_sent = bsz

		# number of candidate hypos per step
		cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

		# offset arrays for converting between different indexing schemes
		bbsz_offsets = (torch.arange(0, bsz, dtype=torch.long, device=device) * beam_size).unsqueeze(1)
		cand_offsets = torch.arange(0, cand_size, dtype=torch.long, device=device)

		# helper function for allocating buffers on the fly
		buffers = {}
		def buffer(name, type_of=tokens):  # noqa
			if name not in buffers:
				buffers[name] = type_of.new()
			return buffers[name]

		def is_finished(sent, step, unfin_idx):
			"""
			Check whether we've finished generation for a given sentence, by
			comparing the worst score among finalized hypotheses to the best
			possible score among unfinalized hypotheses.
			"""
			assert len(finalized[sent]) <= beam_size
			if len(finalized[sent]) == beam_size or step == max_len:
				return True
			return False

		def finalize_hypos(step, bbsz_idx, eos_scores):
			"""
			Finalize the given hypotheses at this step, while keeping the total
			number of finalized hypotheses per sentence <= beam_size.
			Note: the input must be in the desired finalization order, so that
			hypotheses that appear earlier in the input are preferred to those
			that appear later.
			Args:
				step: current time step
				bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
					indicating which hypotheses to finalize
				eos_scores: A vector of the same size as bbsz_idx containing
					scores for each hypothesis
			"""
			assert bbsz_idx.numel() == eos_scores.numel()

			# clone relevant token and attention tensors
			# tokens_clone = tokens.index_select(0, bbsz_idx)[:, :step + 1]
			tokens_clone = tokens.index_select(0, bbsz_idx)
			assert not tokens_clone.eq(constants.EOS_ID).any()
			tokens_clone[:, step] = constants.EOS_ID

			# normalize sentence-level scores
			if self.normalize_scores:
				eos_scores /= (step + 1) ** self.len_penalty

			cum_unfin = []
			prev = 0
			for f in finished:
				if f:
					prev += 1
				else:
					cum_unfin.append(prev)

			sents_seen = set()
			for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores.tolist())):
				unfin_idx = idx // beam_size
				sent = unfin_idx + cum_unfin[unfin_idx]
				sents_seen.add((sent, unfin_idx))

				# if step > src_lengths[unfin_idx] + self.max_len_b:
				# 	score = -math.inf

				if len(finalized[sent]) < beam_size:
					finalized[sent].append({'tokens': tokens_clone[i], 'score': score})

			newly_finished = []
			for sent, unfin_idx in sents_seen:
				# check termination conditions for this sentence
				if not finished[sent] and is_finished(sent, step, unfin_idx):
					finished[sent] = True
					newly_finished.append(unfin_idx)
			return newly_finished

		reorder_state = None
		batch_idxs = None
		for step in range(max_len + 1):  # one extra step for EOS marker
			if not use_transformer:
				# reorder decoder internal states based on the prev choice of beams
				if reorder_state is not None:
					if isinstance(last_dec_state, tuple):
						last_dec_state = (last_dec_state[0].index_select(1, reorder_state), last_dec_state[1].index_select(1, reorder_state))
					else:
						last_dec_state = last_dec_state.index_select(1, reorder_state)
					last_context = last_context.index_select(0, reorder_state)

				last_dec_state, last_context, pred_result = model.predict(module_ind, next_emb, encoder_outs, enc_padding_mask,
														last_dec_state, to_style, last_context)
			else:
				pred_result = model.predict(module_ind, next_emb, encoder_outs, enc_padding_mask, to_style)
			lprobs = F.log_softmax(pred_result['logits']/max(tau, 1e-10), -1)

			lprobs[:, constants.PAD_ID] = -math.inf  # never select pad
			lprobs[:, constants.UNK_ID] -= self.unk_penalty  # apply unk penalty

			# handle max length constraint
			exceed_len_mask = step >= src_lengths + self.max_len_b
			if exceed_len_mask.any():
				exceed_len_mask = repeat_tensors(beam_size, 0, exceed_len_mask)
				lprobs[exceed_len_mask, :constants.EOS_ID] = -math.inf
				lprobs[exceed_len_mask, constants.EOS_ID + 1:] = -math.inf

			# handle prefix tokens (possibly with different lengths)
			if step < self.min_len:
				lprobs[:, constants.EOS_ID] = -math.inf


			eos_bbsz_idx = buffer('eos_bbsz_idx')
			eos_scores = buffer('eos_scores', type_of=scores)

			cand_scores, cand_indices, cand_beams = self.search.step(
				step,
				lprobs.view(bsz, -1, lprobs.size(-1)),
				scores.view(bsz, beam_size, -1)[:, :, :step],
			)

			# cand_bbsz_idx contains beam indices for the top candidate
			# hypotheses, with a range of values: [0, bsz*beam_size),
			# and dimensions: [bsz, cand_size]
			cand_bbsz_idx = cand_beams.add(bbsz_offsets)

			# finalize hypotheses that end in eos, except for blacklisted ones
			# or candidates with a score of -inf
			eos_mask = cand_indices.eq(constants.EOS_ID) & cand_scores.ne(-math.inf)
			eos_mask[:, :beam_size][blacklist] = 0

			# only consider eos when it's among the top beam_size indices
			torch.masked_select(
				cand_bbsz_idx[:, :beam_size],
				mask=eos_mask[:, :beam_size],
				out=eos_bbsz_idx,
			)

			finalized_sents = set()
			if eos_bbsz_idx.numel() > 0:
				torch.masked_select(
					cand_scores[:, :beam_size],
					mask=eos_mask[:, :beam_size],
					out=eos_scores,
				)
				finalized_sents = finalize_hypos(step, eos_bbsz_idx, eos_scores)
				num_remaining_sent -= len(finalized_sents)

			assert num_remaining_sent >= 0
			if num_remaining_sent == 0:
				break
			assert step < max_len

			if len(finalized_sents) > 0:
				new_bsz = bsz - len(finalized_sents)

				# construct batch_idxs which holds indices of batches to keep for the next pass
				batch_mask = cand_indices.new_ones(bsz)
				batch_mask[cand_indices.new_tensor(finalized_sents)] = 0
				batch_idxs = batch_mask.nonzero().squeeze(-1)

				eos_mask = eos_mask[batch_idxs]
				cand_beams = cand_beams[batch_idxs]
				bbsz_offsets.resize_(new_bsz, 1)
				cand_bbsz_idx = cand_beams.add(bbsz_offsets)
				cand_scores = cand_scores[batch_idxs]
				cand_indices = cand_indices[batch_idxs]
				
				src_lengths = src_lengths[batch_idxs]
				encoder_outs = encoder_outs.view(encoder_outs.size(0), bsz, beam_size, encoder_outs.size(-1))[:, batch_idxs].view(encoder_outs.size(0), -1, encoder_outs.size(-1))
				# enc_padding_mask = enc_padding_mask[batch_idxs] if use_transformer else enc_padding_mask[:, batch_idxs]
				to_style = to_style.view(bsz, beam_size)[batch_idxs].view(-1)
				style_emb = style_emb.view(bsz, beam_size, style_emb.size(-1))[batch_idxs].view(-1, style_emb.size(-1))
				if not use_transformer:
					enc_padding_mask = enc_padding_mask.view(enc_padding_mask.size(0), bsz, beam_size)[:, batch_idxs].view(enc_padding_mask.size(0), -1)
					last_dec_state = last_dec_state.view(last_dec_state.size(0), bsz, beam_size, last_dec_state.size(-1))[:, batch_idxs].view(last_dec_state.size(0), -1, last_dec_state.size(-1))
					last_context = last_context.view(bsz, beam_size, last_context.size(-1))[batch_idxs].view(-1, last_context.size(-1))
				else:
					enc_padding_mask = enc_padding_mask.view(bsz, beam_size, enc_padding_mask.size(-1))[batch_idxs].view(-1, enc_padding_mask.size(-1))
					


				blacklist = blacklist[batch_idxs]

				scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
				scores_buf.resize_as_(scores)
				tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
				tokens_buf.resize_as_(tokens)
				
				bsz = new_bsz
			else:
				batch_idxs = None

			# Set active_mask so that values > cand_size indicate eos or
			# blacklisted hypos and values < cand_size indicate candidate
			# active hypos. After this, the min values per row are the top
			# candidate active hypos.
			active_mask = buffer('active_mask')
			eos_mask[:, :beam_size] |= blacklist
			torch.add(
				eos_mask.type_as(cand_offsets) * cand_size,
				cand_offsets[:eos_mask.size(1)],
				out=active_mask,
			)

			# get the top beam_size active hypotheses, which are just the hypos
			# with the smallest values in active_mask
			active_hypos, new_blacklist = buffer('active_hypos'), buffer('new_blacklist')
			torch.topk(
				active_mask, k=beam_size, dim=1, largest=False,
				out=(new_blacklist, active_hypos)
			)

			# update blacklist to ignore any finalized hypos: there can be true values in 
			# the first beam_size choices because only eos in the first beam_size positions 
			# are finalized, but the last beam_size can also contain eos
			blacklist = new_blacklist.ge(cand_size)[:, :beam_size]
			assert (~blacklist).any(dim=1).all()

			active_bbsz_idx = buffer('active_bbsz_idx')
			torch.gather(
				cand_bbsz_idx, dim=1, index=active_hypos,
				out=active_bbsz_idx,
			)
			torch.gather(
				cand_scores, dim=1, index=active_hypos,
				out=scores[:, step].view(bsz, beam_size),
			)

			active_bbsz_idx = active_bbsz_idx.view(-1)

			# copy tokens and scores for active hypotheses
			if step > 0:
				torch.index_select(
					tokens[:, :step], dim=0, index=active_bbsz_idx,
					out=tokens_buf[:, :step],
				)
			torch.gather(
				cand_indices, dim=1, index=active_hypos,
				out=tokens_buf.view(bsz, beam_size, -1)[:, :, step],
			)
			if step > 0:
				torch.index_select(
					scores[:, :step], dim=0, index=active_bbsz_idx,
					out=scores_buf[:, :step],
				)
			torch.gather(
				cand_scores, dim=1, index=active_hypos,
				out=scores_buf.view(bsz, beam_size, -1)[:, :, step],
			)

			# swap buffers
			tokens, tokens_buf = tokens_buf, tokens
			scores, scores_buf = scores_buf, scores
			
			# reorder incremental state in decoder
			reorder_state = active_bbsz_idx
			if not use_transformer:
				next_emb = model.emb(tokens[:, step])
			else:
				next_emb = model.embed(tokens[:, :step+1].t())
				next_emb = torch.cat((style_emb.unsqueeze(0), next_emb), 0)

		# sort by score descending
		for sent in range(len(finalized)):
			finalized[sent] = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)

		
		result['hard_outputs'] = torch.stack([finalized[sent][0]['tokens'] for sent in range(len(finalized))]).t_()

		result['scores'] = torch.tensor([finalized[sent][0]['score'] for sent in range(len(finalized))], dtype=torch.float, device=device)
		result['hard_outputs_padding_mask_with_eos'] = result['hard_outputs'].eq(constants.PAD_ID)
		eos_mask = result['hard_outputs'].eq(constants.EOS_ID)
		result['hard_outputs_padding_mask'] = eos_mask | result['hard_outputs_padding_mask_with_eos']
		result['hard_outputs_lens'] = (~result['hard_outputs_padding_mask']).long().sum(dim=0)
		result['hard_outputs_lens_with_eos'] = (~result['hard_outputs_padding_mask_with_eos']).long().sum(dim=0)
		if getattr(model, 'right{}'.format(module_ind+1)):
			result['hard_outputs_rev'] = reverse_seq(result['hard_outputs'], result['hard_outputs_lens'], eos_mask)
		return result

