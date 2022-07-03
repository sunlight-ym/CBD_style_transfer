import six
import random
import functools
import numpy as np
from .utils import get_tokenizer
from . import constants

def _cut_list(x, n, cut_mode='first'):
    if cut_mode is None:
        cut_mode = 'first'
    if cut_mode == 'first':
        return x[:n]
    elif cut_mode == 'last':
        return x[-n:]
    elif cut_mode == 'random':
        p=random.random()
        return x[-n:] if p>0.5 else x[:n]
    else:
        raise ValueError('the provided cut_mode {} is not supported, only effective with: first, last, and random'.format(cut_mode))


class Preprocessor(object):
    def __init__(self, lower=False, cut_length=None, cut_mode=None,
                 tokenize=None, tokenizer_language='en', stop_words=None):
        super(Preprocessor, self).__init__()
        self.lower=lower
        self.cut_length=cut_length
        self.cut_mode=cut_mode
        self.tokenizer_args = (tokenize, tokenizer_language)
        self.tokenize = get_tokenizer(tokenize, tokenizer_language)
        try:
            self.stop_words = set(stop_words) if stop_words is not None else None
        except TypeError:
            raise ValueError("Stop words must be convertible to a set")

    def __call__(self, x):
        """Load a single example using this field, tokenizing if necessary.

        If the input is a Python 2 `str`, it will be converted to Unicode
        first. If `sequential=True`, it will be tokenized. Then the input
        will be optionally lowercased and passed to the user-provided
        `preprocessing` Pipeline."""
        if (six.PY2 and isinstance(x, six.string_types) and
                not isinstance(x, six.text_type)):
            x = six.text_type(x, encoding='utf-8')
        assert isinstance(x, six.text_type)
        x = self.tokenize(x.rstrip('\n'))
        if self.lower:
            # x = Pipeline(six.text_type.lower)(x)
            x = [six.text_type.lower(w) for w in x]
        if self.stop_words is not None:
            x = [w for w in x if w not in self.stop_words]
        if self.cut_length is not None:
            x = _cut_list(x, self.cut_length, self.cut_mode)
        return x

# noise model from paper "Unsupervised Machine Translation Using Monolingual Corpora Only"
def noise_input(x, unk, word_drop=0.0, k=3):
    x = x[:]
    n = len(x)
    for i in range(n):
        if random.random() < word_drop:
            x[i] = unk

    # slight shuffle such that |sigma[i]-i| <= k
    sigma = (np.arange(n) + (k+1) * np.random.rand(n)).argsort()
    return [x[sigma[i]] for i in range(n)]

def prepare_simple(field, minibatch):
    x = field.pad(minibatch)
    return x

def prepare_value(field, minibatch):
    x = field.pad(minibatch, pad_value=0.0)
    return x
def prepare_class(field, minibatch):
    x, lens = field.pad(minibatch, include_lengths=True)
    return x, lens

def prepare_lm(field, minibatch):
    x, lens = field.pad(minibatch, use_init=True, include_lengths=True)
    y = field.pad(minibatch, use_eos=True)
    return x, lens, y

def prepare_dec(field, minibatch, with_input):
    x = field.pad(minibatch) if with_input else None
    y = field.pad(minibatch, use_eos=True)
    return x, y

def prepare_trans(field, minibatch, noisy, noise_drop=None):
    if noisy:
        noisy_minibatch = [noise_input(x, constants.UNK_ID, noise_drop) for x in minibatch]
        xc = field.pad(noisy_minibatch)
    else:
        xc = None
    x, lens = field.pad(minibatch, include_lengths=True)
    y_out = field.pad(minibatch, use_eos=True)
    return xc, x, lens, y_out


def prepare_eval(field, minibatch):
    x = field.pad(minibatch)
    y_in, lens = field.pad(minibatch, use_init=True, include_lengths=True)
    y_out = field.pad(minibatch, use_eos=True)
    return x, y_in, y_out, lens

class Postprocessor(object):
    """docstring for Postprocessor"""
    def __init__(self, mode, noisy=None, noise_drop=None, usr_func=None, with_input=False):
        super(Postprocessor, self).__init__()
        self.mode = mode
        # self.noisy = noisy
        # self.noise_drop = noise_drop
        # self.require_clean = require_clean
        if mode == 'lm':
            self.post_func = prepare_lm
        elif mode == 'dec':
            self.post_func = functools.partial(prepare_dec, with_input=with_input)
        elif mode == 'class':
            self.post_func = prepare_class
        elif mode == 'trans':
            self.post_func = functools.partial(prepare_trans, noisy=noisy, noise_drop=noise_drop)
        elif mode == 'eval':
            self.post_func = prepare_eval
        elif mode == 'simple':
            self.post_func = prepare_simple
        elif mode == 'value':
            self.post_func = prepare_value
        elif mode == 'usr':
            self.post_func = usr_func
        else:
            raise ValueError('unsupported mode for postprocessing function!')
    def turn_off_noise(self):
        assert self.mode == 'trans'
        self.post_func = functools.partial(prepare_trans, noisy=False)

    def __call__(self, field, minibatch):
        return self.post_func(field, minibatch)