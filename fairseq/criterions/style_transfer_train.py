# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch

from fairseq import utils

from . import FairseqCriterion, register_criterion
from fairseq.data.data_utils import insert_lang_code
from fairseq.utils import anneal_temp

SHOW_TRANS = False

@register_criterion('style_transfer_train')
class StyTransTrainCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.mt_w = args.mt_loss_weight
        self.cls_w = args.classify_loss_weight
        self.slf_rcnstrct_w = args.self_recon_loss_weight
        self.cyc_rcnstrct_w = args.cycle_recon_loss_weight
        self.max_source_positions = args.max_source_positions
        self.len_plus = args.max_train_gen_len_plus
        self.left_pad_source = args.left_pad_source
        self.anneal_max_step = args.temp_anneal_max_step
        self.anneal_step = args.temp_anneal_step
        self.anneal_speed = args.temp_anneal_speed

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--mt-loss-weight', default=1., type=float,
                            help='weight for MT loss')
        parser.add_argument('--classify-loss-weight', default=1., type=float,
                            help='weight for classification loss')
        parser.add_argument('--self-recon-loss-weight', default=1., type=float,
                            help='weight for self-reconstruction loss')
        parser.add_argument('--cycle-recon-loss-weight', default=1., type=float,
                            help='weight for cycled reconstruction loss')
        parser.add_argument('--temp-anneal-max-step', default=15000, type=int,
                            help='stop anneal tempurature after this step')
        parser.add_argument('--temp-anneal-step', default=8000, type=int,
                            help='tempurature anneal step size for soft generation')
        parser.add_argument('--temp-anneal-speed', default=8000, type=int,
                            help='tempurature anneal speed for soft generation')
        parser.add_argument('--max-train-gen-len-plus', default=10, type=int,
                            help='generate to encoder length plus this number')

    def forward(self, model, sample, reduce=True, **kwargs):
        """Compute the loss for the given sample.

        sample['src_plain'] = src_plain_sample_obj
        sample['trg_plain'] = trg_plain_sample_obj

        each sample obj has the following structure:
        {
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

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        num_updates = kwargs['num_updates']
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        temp = anneal_temp(min(self.anneal_max_step, num_updates), self.anneal_step, self.anneal_speed)

        loss = 0

        # MT loss
        mt_loss = 0.0
        if self.mt_w > 0:
            for inp_tokens, inp_length, lang_code, prev_output_tokens, target in zip(
                [sample['net_input']['src_tokens'], sample['net_input']['trg_tokens']],
                [sample['net_input']['src_lengths'], sample['net_input']['trg_lengths']],
                [model.src_dict.trg_lang_index, model.src_dict.src_lang_index],
                [sample['net_input']['target_prev_output_tokens'], sample['net_input']['source_prev_output_tokens']],
                [sample['target'], sample['source']],
            ):

                net_output = model(src_tokens=insert_lang_code(inp_tokens, True, lang_code), src_lengths=inp_length,
                                prev_output_tokens=prev_output_tokens)
                lprobs = model.get_normalized_probs(net_output, log_probs=True)
                lprobs = lprobs.view(-1, lprobs.size(-1))
                target_flat = target.view(-1, 1)
                non_pad_mask = target_flat.ne(self.padding_idx)
                nll_loss = -lprobs.gather(dim=-1, index=target_flat)[non_pad_mask]
                smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
                if reduce:
                    nll_loss = nll_loss.sum()
                    smooth_loss = smooth_loss.sum()
                eps_i = self.eps / lprobs.size(-1)

                mt_loss += (1. - self.eps) * nll_loss + eps_i * smooth_loss
                loss += self.mt_w * ( (1. - self.eps) * nll_loss + eps_i * smooth_loss )
            mt_loss /= 2

        # classification loss
        cls_loss = 0.0
        if self.cls_w > 0:
            for inp_tokens, inp_length, lang_code, label in zip(
                [sample['src_plain']['net_input']['src_tokens'], sample['trg_plain']['net_input']['src_tokens']],
                [sample['src_plain']['net_input']['src_lengths'], sample['trg_plain']['net_input']['src_lengths']],
                [model.src_dict.trg_lang_index, model.src_dict.src_lang_index],
                [1.0, 0.0],
            ):
                # print("label", label)
                cls_loss_tmp = model.soft_sample_classify_loss(
                            insert_lang_code(inp_tokens, True, lang_code),
                            inp_length,
                            label,
                            temp=temp,
                            plus_length=self.len_plus,
                            maxlen=self.max_source_positions,
                        )
                cls_loss += cls_loss_tmp
                loss += self.cls_w * cls_loss_tmp * sample_size
            cls_loss /= 2

        # self-reconstruction loss
        slf_rcnstrct_loss = 0.0
        if self.slf_rcnstrct_w > 0:
            for inp_tokens, inp_length, lang_code, prev_output_tokens, target in zip(
                [sample['src_plain']['net_input']['src_tokens'], sample['trg_plain']['net_input']['src_tokens']],
                [sample['src_plain']['net_input']['src_lengths'], sample['trg_plain']['net_input']['src_lengths']],
                [model.src_dict.src_lang_index, model.src_dict.trg_lang_index],
                [sample['src_plain']['net_input']['source_prev_output_tokens'], sample['trg_plain']['net_input']['source_prev_output_tokens']],
                [sample['src_plain']['source'], sample['trg_plain']['source']],
            ):
                net_output = model(
                        src_tokens=insert_lang_code(inp_tokens, True, lang_code),
                        src_lengths=inp_length,
                        prev_output_tokens=prev_output_tokens,
                    )
                lprobs = model.get_normalized_probs(net_output, log_probs=True)
                lprobs = lprobs.view(-1, lprobs.size(-1))
                target_flat = target.view(-1, 1)
                non_pad_mask = target_flat.ne(self.padding_idx)
                nll_loss = -lprobs.gather(dim=-1, index=target_flat)[non_pad_mask]
                smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
                if reduce:
                    nll_loss = nll_loss.sum()
                    smooth_loss = smooth_loss.sum()
                eps_i = self.eps / lprobs.size(-1)
                slf_rcnstrct_loss += (1. - self.eps) * nll_loss + eps_i * smooth_loss
                loss += self.slf_rcnstrct_w * ( (1. - self.eps) * nll_loss + eps_i * smooth_loss )
            slf_rcnstrct_loss /= 2

        # cycled-reconstruction loss
        cyc_rcnstrct_loss = 0.0
        if self.cyc_rcnstrct_w > 0:
            for inp_tokens, inp_length, lang_code, rev_lang_code, prev_output_tokens, target in zip(
                [sample['src_plain']['net_input']['src_tokens'], sample['trg_plain']['net_input']['src_tokens']],
                [sample['src_plain']['net_input']['src_lengths'], sample['trg_plain']['net_input']['src_lengths']],
                [model.src_dict.trg_lang_index, model.src_dict.src_lang_index],
                [model.src_dict.src_lang_index, model.src_dict.trg_lang_index],
                [sample['src_plain']['net_input']['source_prev_output_tokens'], sample['trg_plain']['net_input']['source_prev_output_tokens']],
                [sample['src_plain']['source'], sample['trg_plain']['source']],
            ):
                # print("inp_tokens.size", inp_tokens.size())
                # print("inp_length.max", torch.max(inp_length))

                trans_tokens, trans_length = model.translate(
                            src_tokens=insert_lang_code(inp_tokens, True, lang_code),
                            src_lengths=inp_length,
                            temp=1.0,
                            plus_length=self.len_plus,
                            maxlen=self.max_source_positions,
                            left_pad=self.left_pad_source,
                            )

                if SHOW_TRANS:
                    print("inp_tokens:", model.src_dict.string(inp_tokens))
                    print("trans_tokens:", model.src_dict.string(trans_tokens))

                # print("trans_tokens.max", torch.max(trans_tokens))
                # print("trans_length.max", torch.max(trans_length))
                # print("trans_tokens.size", trans_tokens.size())

                net_output = model(
                        src_tokens=insert_lang_code(trans_tokens, True, rev_lang_code),
                        src_lengths=trans_length,
                        prev_output_tokens=prev_output_tokens)

                lprobs = model.get_normalized_probs(net_output, log_probs=True)
                lprobs = lprobs.view(-1, lprobs.size(-1))
                target_flat = target.view(-1, 1)
                non_pad_mask = target_flat.ne(self.padding_idx)
                nll_loss = -lprobs.gather(dim=-1, index=target_flat)[non_pad_mask]
                smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
                if reduce:
                    nll_loss = nll_loss.sum()
                    smooth_loss = smooth_loss.sum()
                eps_i = self.eps / lprobs.size(-1)

                cyc_rcnstrct_loss += ( (1. - self.eps) * nll_loss + eps_i * smooth_loss )
                loss += self.cyc_rcnstrct_w * ( (1. - self.eps) * nll_loss + eps_i * smooth_loss )
            cyc_rcnstrct_loss /= 2

        loss /= 2;
        loss /= max((self.mt_w + self.cls_w + self.cyc_rcnstrct_w + self.slf_rcnstrct_w), 1.0);

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            # 'loss': utils.item(loss.data) if reduce else loss.data,
            # 'mt_loss': utils.item(mt_loss.data) if reduce else mt_loss.data,
            # 'cyc_rcnstrct_loss': utils.item(cyc_rcnstrct_loss.data) if reduce else cyc_rcnstrct_loss.data,
            # 'slf_rcnstrct_loss': utils.item(slf_rcnstrct_loss.data) if reduce else slf_rcnstrct_loss.data,
            # 'cls_loss': utils.item(cls_loss.data),
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
            'tempurature': temp,
            'loss': utils.item(loss.data),
        }
        for w, loss_tensor, key in zip([self.mt_w, self.cls_w, self.slf_rcnstrct_w, self.cyc_rcnstrct_w],
                                  [mt_loss, cls_loss, slf_rcnstrct_loss, cyc_rcnstrct_loss],
                                  ['mt_loss', 'cls_loss', 'slf_rcnstrct_loss', 'cyc_rcnstrct_loss']):
            if w > 0:
                logging_output[key] = utils.item(loss_tensor.data)
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        num_updates = len(logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'mt_loss': sum(log.get('mt_loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'cyc_rcnstrct_loss': sum(log.get('cyc_rcnstrct_loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'slf_rcnstrct_loss': sum(log.get('slf_rcnstrct_loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'cls_loss': sum(log.get('cls_loss', 0) for log in logging_outputs) / num_updates,
            'sample_size': sample_size,
            'tempurature': sum(log.get('tempurature', 0) for log in logging_outputs) / num_updates,
        }
