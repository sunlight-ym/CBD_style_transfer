# coding: utf8
from collections import Counter, OrderedDict
from itertools import chain
import six
import torch

from .dataset import Dataset
from .utils import dtype_to_attr
from .vocab import Vocab
from . import constants

class RawField(object):
	""" Defines a general datatype.

	Every dataset consists of one or more types of data. For instance, a text
	classification dataset contains sentences and their classes, while a
	machine translation dataset contains paired examples of text in two
	languages. Each of these types of data is represented by a RawField object.
	A RawField object does not assume any property of the data type and
	it holds parameters relating to how a datatype should be processed.

	Attributes:
		preprocessing: The Pipeline that will be applied to examples
			using this field before creating an example.
			Default: None.
		postprocessing: A Pipeline that will be applied to a list of examples
			using this field before assigning to a batch.
			Function signature: (batch(list)) -> object
			Default: None.
		is_target: Whether this field is a target variable.
			Affects iteration over batches. Default: False
	"""

	def __init__(self, preprocessing=None, postprocessing=None):
		self.preprocessing = preprocessing
		self.postprocessing = postprocessing

	def preprocess(self, x):
		""" Preprocess an example if the `preprocessing` Pipeline is provided. """
		if self.preprocessing is not None:
			return self.preprocessing(x)
		else:
			return x

	def process(self, batch, *args, **kwargs):
		""" Process a list of examples to create a batch.

		Postprocess the batch with user-provided Pipeline.

		Args:
			batch (list(object)): A list of object from a batch of examples.
		Returns:
			object: Processed object given the input and custom
			postprocessing Pipeline.
		"""
		if self.postprocessing is not None:
			batch = self.postprocessing(batch, *args, **kwargs)
		return batch


class Field(RawField):
	"""Defines a datatype together with instructions for converting to Tensor.

	Field class models common text processing datatypes that can be represented
	by tensors.  It holds a Vocab object that defines the set of possible values
	for elements of the field and their corresponding numerical representations.
	The Field object also holds other parameters relating to how a datatype
	should be numericalized, such as a tokenization method and the kind of
	Tensor that should be produced.

	If a Field is shared between two columns in a dataset (e.g., question and
	answer in a QA dataset), then they will have a shared vocabulary.

	Attributes:
		sequential: Whether the datatype represents sequential data. If False,
			no tokenization is applied. Default: True.
		use_vocab: Whether to use a Vocab object. If False, the data in this
			field should already be numerical. Default: True.
		init_token: A token that will be prepended to every example using this
			field, or None for no initial token. Default: None.
		eos_token: A token that will be appended to every example using this
			field, or None for no end-of-sentence token. Default: None.
		fix_length: A fixed length that all examples using this field will be
			padded to, or None for flexible sequence lengths. Default: None.
		dtype: The torch.dtype class that represents a batch of examples
			of this kind of data. Default: torch.long.
		preprocessing: The Pipeline that will be applied to examples
			using this field after tokenizing but before numericalizing. Many
			Datasets replace this attribute with a custom preprocessor.
			Default: None.
		postprocessing: A Pipeline that will be applied to examples using
			this field after numericalizing but before the numbers are turned
			into a Tensor. The pipeline function takes the batch as a list, and
			the field's Vocab.
			Default: None.
		lower: Whether to lowercase the text in this field. Default: False.
		tokenize: The function used to tokenize strings using this field into
			sequential examples. If "spacy", the SpaCy tokenizer is
			used. If a non-serializable function is passed as an argument,
			the field will not be able to be serialized. Default: string.split.
		tokenizer_language: The language of the tokenizer to be constructed.
			Various languages currently supported only in SpaCy.
		include_lengths: Whether to return a tuple of a padded minibatch and
			a list containing the lengths of each examples, or just a padded
			minibatch. Default: False.
		batch_first: Whether to produce tensors with the batch dimension first.
			Default: False.
		pad_token: The string token used as padding. Default: "<pad>".
		unk_token: The string token used to represent OOV words. Default: "<unk>".
		pad_first: Do the padding of the sequence at the beginning. Default: False.
		truncate_first: Do the truncating of the sequence at the beginning. Default: False
		stop_words: Tokens to discard during the preprocessing step. Default: None
		is_target: Whether this field is a target variable.
			Affects iteration over batches. Default: False
	"""

	vocab_cls = Vocab
	# Dictionary mapping PyTorch tensor dtypes to the appropriate Python
	# numeric type.
	dtypes = {
		torch.float32: float,
		torch.float: float,
		torch.float64: float,
		torch.double: float,
		torch.float16: float,
		torch.half: float,

		torch.uint8: int,
		torch.int8: int,
		torch.int16: int,
		torch.short: int,
		torch.int32: int,
		torch.int: int,
		torch.int64: int,
		torch.long: int,
	}

	ignore = ['dtype']

	def __init__(self, sequential=True, use_vocab=True, 
					unk_token=constants.UNK_WORD, pad_token=constants.PAD_WORD,
					init_token=constants.BOS_WORD, eos_token=constants.EOS_WORD, 
					preprocessing=None, postprocessing=None,
					dtype=torch.long, batch_first=False):
		self.sequential = sequential
		self.use_vocab = use_vocab
		self.unk_token = unk_token
		self.pad_token = pad_token
		self.init_token = init_token
		self.eos_token = eos_token
		
		self.preprocessing = preprocessing
		self.postprocessing = postprocessing
		self.dtype = dtype
		self.batch_first = batch_first
		
		
	def __getstate__(self):
		str_type = dtype_to_attr(self.dtype)
		attrs = {k: v for k, v in self.__dict__.items() if k not in self.ignore}
		attrs['dtype'] = str_type

		return attrs

	def __setstate__(self, state):
		state['dtype'] = getattr(torch, state['dtype'])
		self.__dict__.update(state)

	def __hash__(self):
		# we don't expect this to be called often
		return 42

	def __eq__(self, other):
		if not isinstance(other, RawField):
			return False

		return self.__dict__ == other.__dict__

	def process(self, batch, device=None):
		""" Process a list of examples to create a torch.Tensor.

		Pad, numericalize, and postprocess a batch and create a tensor.

		Args:
			batch (list(object)): A list of object from a batch of examples.
		Returns:
			torch.autograd.Variable: Processed object given the input
			and custom postprocessing Pipeline.
		"""

		batch = self.str2index(batch)
		if self.postprocessing is not None:
			batch = self.postprocessing(self, batch)
		tensor = self.numericalize(batch, device=device)
		return tensor

	def pad(self, minibatch, fix_length=None, truncate_first=False, use_init=False, 
		use_eos=False, pad_first=False, reverse=False, offset=0, include_lengths=False,
		pad_value=None, init_value=None, eos_value=None):
		"""Pad a batch of examples using this field.

		Pads to self.fix_length if provided, otherwise pads to the length of
		the longest example in the batch. Prepends self.init_token and appends
		self.eos_token if those attributes are not None. Returns a tuple of the
		padded list and a list containing lengths of each example if
		`self.include_lengths` is `True` and `self.sequential` is `True`, else just
		returns the padded list. If `self.sequential` is `False`, no padding is applied.
		"""
		minibatch = list(minibatch)
		if not self.sequential:
			return minibatch

		assert (self.pad_token is not None) or (pad_value is not None)
		if use_init:
			assert (self.init_token is not None) or (init_value is not None)
		if use_eos:
			assert (self.eos_token is not None) or (eos_value is not None)
		if offset != 0:
			assert abs(offset) < min(len(x) for x in minibatch)

		if fix_length is None:
			max_len = max(len(x) for x in minibatch) - abs(offset)
		else:
			max_len = fix_length - (use_init, use_eos).count(True)
		
		padded, lengths = [], []
		pad_idx = [self.vocab.stoi[self.pad_token]] if pad_value is None else [pad_value]
		init_idx = ([self.vocab.stoi[self.init_token]] if init_value is None else [init_value]) if use_init else []
		eos_idx = ([self.vocab.stoi[self.eos_token]] if eos_value is None else [eos_value]) if use_eos else []

		for x in minibatch:
			t = list(x[-(max_len+abs(offset)):] if truncate_first else x[:(max_len+abs(offset))])
			if offset != 0:
				t = t[offset:] if offset>0 else t[:len(t)+offset]
			p = pad_idx * (max_len - len(t))
			t = init_idx+t+eos_idx
			if reverse:
				t = t[::-1]
			padded.append(p+t if pad_first else t+p)

			lengths.append(len(padded[-1]) - len(p))
		
		if include_lengths:
			return (padded, lengths)
		return padded

	def build_vocab(self, *args, **kwargs):
		"""Construct the Vocab object for this field from one or more datasets.

		Arguments:
			Positional arguments: Dataset objects or other iterable data
				sources from which to construct the Vocab object that
				represents the set of possible values for this field. If
				a Dataset object is provided, all columns corresponding
				to this field are used; individual columns can also be
				provided directly.
			Remaining keyword arguments: Passed to the constructor of Vocab.
		"""
		counter = Counter()
		sources = []
		for arg in args:
			if isinstance(arg, Dataset):
				sources += [getattr(arg, name) for name, field in
							arg.fields.items() if field is self]
			else:
				sources.append(arg)
		for data in sources:
			for x in data:
				if not self.sequential:
					x = [x]
				try:
					counter.update(x)
				except TypeError:
					counter.update(chain.from_iterable(x))
		specials = [tok for tok in [self.pad_token, self.init_token,
										self.eos_token] if tok is not None]
		self.vocab = self.vocab_cls(counter, unk_token=self.unk_token, specials=specials, **kwargs)

	def str2index(self, arr):
		if self.use_vocab:
			if self.sequential:
				arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]
			else:
				arr = [self.vocab.stoi[x] for x in arr]

		else:
			if self.dtype not in self.dtypes:
				raise ValueError(
					"Specified Field dtype {} can not be used with "
					"use_vocab=False because we do not know how to numericalize it. "
					"Please raise an issue at "
					"https://github.com/pytorch/text/issues".format(self.dtype))
			numericalization_func = self.dtypes[self.dtype]
			# It doesn't make sense to explicitly coerce to a numeric type if
			# the data is sequential, since it's unclear how to coerce padding tokens
			# to a numeric type.
			if self.sequential:
				arr = [[numericalization_func(x) if isinstance(x, six.string_types)
									   else x for x in ex] for ex in arr]
			else:
				arr = [numericalization_func(x) if isinstance(x, six.string_types)
					   else x for x in arr]

		return arr
	
	def _to_tensor(self, arr, device=None):
		
		var = torch.tensor(arr, dtype=self.dtype, device=device)
		if self.sequential and not self.batch_first and var.dim()==2:
			var.t_()
		if self.sequential:
			var = var.contiguous()
		return var

	def numericalize(self, arr, device=None):
		"""Turn a batch of examples that use this field into a Variable.

		If the field has include_lengths=True, a tensor of lengths will be
		included in the return value.

		Arguments:
			arr (List[List[str]], or tuple of (List[List[str]], List[int])):
				List of tokenized and padded examples, or tuple of List of
				tokenized and padded examples and List of lengths of each
				example if self.include_lengths is True.
			device (str or torch.device): A string or instance of `torch.device`
				specifying which device the Variables are going to be created on.
				If left as default, the tensors will be created on cpu. Default: None.
		"""
		
		if isinstance(arr, tuple):
			ret = (self._to_tensor(x, device) if x is not None else None for x in arr)
		else:
			ret = self._to_tensor(arr, device)

		return ret


class LabelField(Field):
	"""A Label field.

	A label field is a shallow wrapper around a standard field designed to hold labels
	for a classification task. Its only use is to set the unk_token and sequential to
	`None` by default.
	"""

	def __init__(self, **kwargs):
		# whichever value is set for sequential, unk_token, and is_target
		# will be overwritten
		kwargs['sequential'] = False
		kwargs['unk_token'] = None
		kwargs['pad_token'] = None
		kwargs['init_token'] = None
		kwargs['eos_token'] = None
		
		super(LabelField, self).__init__(**kwargs)

class NumField(Field):
	"""docstring for NumField"""
	def __init__(self, **kwargs):
		kwargs['use_vocab'] = False
		kwargs['unk_token'] = None
		kwargs['pad_token'] = None
		kwargs['init_token'] = None
		kwargs['eos_token'] = None
		super(NumField, self).__init__(**kwargs)
		