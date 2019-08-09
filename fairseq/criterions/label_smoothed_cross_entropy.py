# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

from fairseq import utils

from . import FairseqCriterion, register_criterion
from fairseq.data.data_utils import insert_lang_code

@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')

    def forward(self, model, sample, reduce=True, **kwargs):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # MT loss
        mt_loss = 0.0
        nll_loss = 0.0
        for inp_tokens, inp_length, lang_code, prev_output_tokens, target in zip(
            [sample['net_input']['src_tokens']],
            [sample['net_input']['src_lengths']],
            [model.src_dict.trg_lang_index],
            [sample['net_input']['target_prev_output_tokens']],
            [sample['target']],
        ):

            net_output = model(src_tokens=insert_lang_code(inp_tokens, True, lang_code), src_lengths=inp_length,
                            prev_output_tokens=prev_output_tokens)
            lprobs = model.get_normalized_probs(net_output, log_probs=True)
            lprobs = lprobs.view(-1, lprobs.size(-1))
            target_flat = target.view(-1, 1)
            non_pad_mask = target_flat.ne(self.padding_idx)
            nll_loss_tmp = -lprobs.gather(dim=-1, index=target_flat)[non_pad_mask]
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
            if reduce:
                nll_loss_tmp = nll_loss_tmp.sum()
                smooth_loss = smooth_loss.sum()
            eps_i = self.eps / lprobs.size(-1)

            nll_loss += nll_loss_tmp
            mt_loss += (1. - self.eps) * nll_loss + eps_i * smooth_loss

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']

        logging_output = {
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'loss': utils.item(mt_loss.data) if reduce else mt_loss.data,
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
        }

        return mt_loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'sample_size': sample_size,
        }
