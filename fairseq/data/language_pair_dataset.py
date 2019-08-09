# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from . import data_utils, FairseqDataset

def add_plain(sample, src_plain, trg_plain):
    sample['src_plain'] = src_plain
    sample['trg_plain'] = trg_plain

    return sample

def combine_samples(src_sample, trg_sample):
    return {
        'id': src_sample['id'],
        'ntokens': src_sample['ntokens'],
        'target': trg_sample['source'],
        'source': src_sample['source'],
        'src_plain': src_sample,
        'trg_plain': trg_sample,
    }

def collate(samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    source = merge('source', left_pad=left_pad_target)
    source_prev_output_tokens = merge(
        'source',
        left_pad=left_pad_target,
        move_eos_to_beginning=True,
    )

    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel()+1 for s in samples]) # +1 for lang_idx
    src_lengths, sort_order = src_lengths.sort(descending=True)

    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)
    source = source.index_select(0, sort_order)
    source_prev_output_tokens = source_prev_output_tokens.index_select(0, sort_order)

    target_prev_output_tokens = None
    target = None
    trg_lengths = None
    trg_tokens = None
    if samples[0].get('target', None) is not None:
        trg_tokens = merge('target', left_pad=left_pad_source)
        target = merge('target', left_pad=left_pad_target)
        trg_lengths = torch.LongTensor([s['target'].numel()+1 for s in samples])
        # we create a shifted version of targets for feeding the
        # previous output token(s) into the next decoder step
        target_prev_output_tokens = merge(
            'target',
            left_pad=left_pad_target,
            move_eos_to_beginning=True,
        )
        target_prev_output_tokens = target_prev_output_tokens.index_select(0, sort_order)
        target = target.index_select(0, sort_order)
        trg_lengths = trg_lengths.index_select(0, sort_order)
        trg_tokens = trg_tokens.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    return {
        'id': id,
        'ntokens': ntokens,
        'net_input': {
            'source_prev_output_tokens':source_prev_output_tokens,
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'target_prev_output_tokens': target_prev_output_tokens,
            'trg_tokens': trg_tokens,
            'trg_lengths': trg_lengths,
        },
        'target': target,
        'source': source,
    }


class LanguagePairDataset(FairseqDataset):
    """A pair of torch.utils.data.Datasets."""

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle

    def __getitem__(self, index):
        return {
            'id': index,
            'source': self.src[index],
            'target': self.tgt[index] if self.tgt is not None else None,
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128):
        max_source_positions, max_target_positions = self._get_max_positions(max_positions)
        src_len, tgt_len = min(src_len, max_source_positions), min(tgt_len, max_target_positions)
        bsz = num_tokens // max(src_len, tgt_len)
        return self.collater([
            {
                'id': i,
                'source': self.src_dict.dummy_sentence(src_len),
                'target': self.tgt_dict.dummy_sentence(tgt_len) if self.tgt_dict is not None else \
                          self.src_dict.dummy_sentence(tgt_len),
            }
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return an example's length (number of tokens), used for batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Ordered indices for batching."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    def valid_size(self, index, max_positions):
        """Check if an example's size is valid according to max_positions."""
        max_source_positions, max_target_positions = self._get_max_positions(max_positions)
        return (
            self.src_sizes[index] <= max_source_positions
            and (self.tgt_sizes is None or self.tgt_sizes[index] <= max_target_positions)
        )

    def _get_max_positions(self, max_positions):
        if max_positions is None:
            return self.max_source_positions, self.max_target_positions
        assert len(max_positions) == 2
        max_src_pos, max_tgt_pos = max_positions
        return min(self.max_source_positions, max_src_pos), min(self.max_target_positions, max_tgt_pos)
