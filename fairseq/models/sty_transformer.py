# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from fairseq import options
from fairseq import utils
from fairseq.data import data_utils

from fairseq.modules import (
    AdaptiveSoftmax, LearnedPositionalEmbedding, MultiheadAttention, SinusoidalPositionalEmbedding
)

from . import (
    FairseqIncrementalDecoder, FairseqEncoder, FairseqLanguageModel, FairseqModel, register_model,
    register_model_architecture,
)


@register_model('sty_transformer')
class StyleTransformerModel(FairseqModel):
    def __init__(self, encoder, decoder, discriminator, src_dict, tgt_dict):
        super().__init__(encoder, decoder)
        self.discriminator = discriminator
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        # discriminator args
        parser.add_argument('--cnn-filters', type=int, metavar='N',
                            help='num of CNN filters')
        parser.add_argument('--cnn-dropout', type=float, metavar='D',
                            help='cnn dropout rate')


    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise RuntimeError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise RuntimeError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise RuntimeError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
        else:
            raise NotImplementedError

        discriminator_embed_tokens = build_embedding(
            src_dict, args.encoder_embed_dim, args.encoder_embed_path
        )

        encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
        decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)
        discriminator = CNNDiscriminator(args, discriminator_embed_tokens)
        return StyleTransformerModel(encoder, decoder, discriminator, src_dict, tgt_dict)

    def soft_sample_classify_loss(self, src_tokens, src_lengths, label, temp, plus_length, maxlen):
        encoder_out = self.encoder(src_tokens, src_lengths)
        soft_embed = self.decoder.generate_soft_embed(
                encoder_out, self.discriminator.embedding, temp,
                self.src_dict.eos(), self.src_dict.pad(), plus_length, maxlen)
        return self.discriminator.forward_loss_embed(soft_embed, label)

    def translate(self, src_tokens, src_lengths, temp, plus_length, maxlen, left_pad):
        encoder_out = self.encoder(src_tokens, src_lengths)
        return self.decoder.generate_hard_idx(encoder_out, temp, plus_length,
                maxlen, self.src_dict.eos(), self.src_dict.pad(), left_pad)

    def fix_discriminator(self):
        for p in self.discriminator.embedding.parameters():
            p.requires_grad = False
        for p in self.discriminator.parameters():
            p.requires_grad = False

class CNNDiscriminator(nn.Module):
    """
    A CNN based binary classifier, use default filter size as [3,4,5]
    """
    def __init__(self, args, embed_tokens, gpu=True):
        super().__init__()

        self.embedding = embed_tokens
        self.emb_dim = embed_tokens.embedding_dim

        self.conv3 = nn.Conv2d(1, args.cnn_filters, (3, self.emb_dim))
        self.conv4 = nn.Conv2d(1, args.cnn_filters, (4, self.emb_dim))
        self.conv5 = nn.Conv2d(1, args.cnn_filters, (5, self.emb_dim))

        self.disc_fc = nn.Sequential(
            nn.Dropout(args.cnn_dropout),
            nn.Linear(3*args.cnn_filters, 2)
        )

        self.gpu = gpu
        if self.gpu:
            self.cuda()

    def pred(self, inputs):
        """
        Inputs: batch of sentences: mbsize x seq_len
            ## we cannot have char input for discriminator because of the
            ## soft generation of decoder is on word-level and cannot be
            ## converted to chars
        Outputs: tuple of batches of predictions: (mbsize x 2, mbsize)
        """

        self.eval()

        inputs = self.embedding(inputs)
        logits = self.forward_embed(inputs).view(-1, 2)
        y = F.softmax(logits, dim=1) # mbsize x 2
        pred = torch.argmax(logits, dim=1) # mbsize

        self.train()

        return y, pred

    def forward_embed(self, inputs):
        """
        Inputs: batch of sentences: mbsize x seq_len x emb_dim
        Outputs: batch of predictions: mbsize x 2
        """

        # print("inputs.size()", inputs.size())

        inputs = inputs.unsqueeze(1)  # mbsize x 1 x seq_len x emb_dim

        # Conv
        x3 = F.relu(self.conv3(inputs)).squeeze(dim=3)
        x4 = F.relu(self.conv4(inputs)).squeeze(dim=3)
        x5 = F.relu(self.conv5(inputs)).squeeze(dim=3)

        # Max-over-time-pool
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(dim=2)
        x4 = F.max_pool1d(x4, x4.size(2)).squeeze(dim=2)
        x5 = F.max_pool1d(x5, x5.size(2)).squeeze(dim=2)

        x = torch.cat([x3, x4, x5], dim=1)
        y = self.disc_fc(x)

        return y

    def forward(self, inputs, targets):
        """
        Inputs: batch of sentences: mbsize x seq_len
            ## we cannot have char input for discriminator because of the
            ## soft generation of decoder is on word-level and cannot be
            ## converted to chars
        Ouput: cross entropy between prediction and targets
        """
        # print("inputs.size", inputs.size())
        inputs = self.embedding(inputs)
        return self.forward_loss_embed(inputs, targets)

    def forward_loss_embed(self, inputs, targets):
        """
        Inputs:
            inputs: mbsize x seq_len x emb_dim
            targets: mbsize
        Outputs: batch of predictions: mbsize x 2
        """

        if isinstance(targets, float):
            mbsize = inputs.size(0)
            targets = Variable(torch.ones([mbsize], dtype=torch.int64)*targets).cuda()

        y = self.forward_embed(inputs)

        cross_entropy_loss = F.cross_entropy(
            y.view(-1, 2), targets.view(-1), size_average=True
        )

        return cross_entropy_loss

class TransformerEncoder(FairseqEncoder):
    """Transformer encoder."""

    def __init__(self, args, dictionary, embed_tokens, left_pad=True):
        super().__init__(dictionary)
        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])

    def forward(self, src_tokens, src_lengths):
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        return self.forward_embed(x, encoder_padding_mask)

    def forward_embed(self, x, encoder_padding_mask=None):
        # input: B x T x C

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def upgrade_state_dict(self, state_dict):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if 'encoder.embed_positions.weights' in state_dict:
                del state_dict['encoder.embed_positions.weights']
            state_dict['encoder.embed_positions._float_tensor'] = torch.FloatTensor(1)
        return state_dict


class TransformerDecoder(FairseqIncrementalDecoder):
    """Transformer decoder."""

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, left_pad=False):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        embed_dim = embed_tokens.embedding_dim
        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            left_pad=left_pad,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])

        self.adaptive_softmax = None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary), args.decoder_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.dropout
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=embed_dim ** -0.5)

    def tokens_to_embed(self, prev_output_tokens, incremental_state=None):
        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None):
        # get embed
        x = self.tokens_to_embed(prev_output_tokens, incremental_state)
        return self.forward_embed(x, encoder_out, incremental_state)

    def forward_embed(self, x, encoder_out, incremental_state):

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
            )

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = F.linear(x, self.embed_out)

        return x, attn

    def generate_hard_idx(self, encoder_out, temp, plus_length, maxlen,
            eos_idx, pad_idx, left_pad=True):
        self.eval()

        mbsize = encoder_out['encoder_out'].size(1)
        encoder_length = encoder_out['encoder_out'].size(0)
        maxlen = min(maxlen, encoder_length + plus_length)

        prev_output_tokens = torch.LongTensor([eos_idx]).repeat(mbsize, 1)
        prev_output_tokens = prev_output_tokens.cuda()
        incremental_state = {}

        emb_t = self.tokens_to_embed(prev_output_tokens, incremental_state)
        outputs = []

        for i in range(maxlen):
            o = self.forward_embed(emb_t, encoder_out, incremental_state)[0].view(mbsize, -1) # mbsize x n_vocab

            # Sample softmax with temperature
            word = torch.argmax(o, 1).cuda().view(mbsize, 1) # mbsize

            emb_t = self.embed_tokens(word)
            emb_t = self.embed_scale * emb_t.view(mbsize, 1, -1)

            # if positions is not None:
            emb_t += self.embed_positions.one_step_forward(mbsize, i+1, incremental_state)
            emb_t = F.dropout(emb_t, p=self.dropout, training=self.training)

            # Save resulting index
            outputs.append(word)

        outputs = torch.cat(outputs, dim=1)
        trans = []
        for i in range(outputs.size(0)):
            o = outputs[i,:]
            idx = (o == eos_idx).nonzero()
            if idx.size(0) > 0: # find <eos>
                o = o[:idx[0]+1]
            else:
                o[-1] = eos_idx
            trans.append(o)

        # Back to default state: train
        self.train()

        trans_tokens = data_utils.collate_tokens(trans, pad_idx=pad_idx,
                eos_idx=eos_idx, left_pad=left_pad,
                move_eos_to_beginning=False)

        trans_length = torch.LongTensor([t.size(0)+1 for t in trans])

        return trans_tokens, trans_length

    def generate_soft_embed(self, encoder_out, embedding_out, temp,
            eos_idx, pad_idx, plus_length, maxlen):
        mbsize = encoder_out['encoder_out'].size(1)
        encoder_length = encoder_out['encoder_out'].size(0)
        # print('mbsize', mbsize)
        # print('encoder_length', encoder_length)
        maxlen = min(maxlen, encoder_length + plus_length)

        prev_output_tokens = torch.LongTensor([eos_idx]).repeat(mbsize, 1)
        prev_output_tokens = prev_output_tokens.cuda()
        incremental_state = {}

        emb_t = self.tokens_to_embed(prev_output_tokens, incremental_state)
        outputs = []

        for i in range(maxlen):
            o = self.forward_embed(emb_t, encoder_out, incremental_state)[0].view(mbsize, -1) # mbsize x n_vocab

            # Sample softmax with temperature
            y = F.softmax(o / temp, dim=1) # mbsize x n_vocab

            emb_t = y.unsqueeze(0) @ self.embed_tokens.weight
            emb_t =  self.embed_scale * emb_t.view(mbsize, 1, -1)

            # if positions is not None:
            emb_t += self.embed_positions.one_step_forward(mbsize, i+1, incremental_state)
            emb_t = F.dropout(emb_t, p=self.dropout, training=self.training)

            # Save resulting soft embedding
            emb = y.unsqueeze(0) @ embedding_out.weight
            outputs.append(emb.view(mbsize, 1, -1)) # mbsize x 1 x emb_dim

        # mbsize x seq_len x emb_dim
        outputs = torch.cat(outputs, dim=1) # mbsize x seqlen x emb_dim

        return outputs.cuda()

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def upgrade_state_dict(self, state_dict):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if 'decoder.embed_positions.weights' in state_dict:
                del state_dict['decoder.embed_positions.weights']
            state_dict['decoder.embed_positions._float_tensor'] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = 'decoder.layers.{}.layer_norms.{}.{}'.format(i, old, m)
                    if k in state_dict:
                        state_dict['decoder.layers.{}.{}.{}'.format(i, new, m)] = state_dict[k]
                        del state_dict[k]

        return state_dict


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: dropout -> add residual -> layernorm.
    In the tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    dropout -> add residual.
    We default to the approach in the paper, but the tensor2tensor approach can
    be enabled by setting `normalize_before=True`.
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(2)])

    def forward(self, x, encoder_padding_mask):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block."""

    def __init__(self, args, no_encoder_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.decoder_normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = MultiheadAttention(
                self.embed_dim, args.decoder_attention_heads,
                dropout=args.attention_dropout,
            )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

    def forward(self, x, encoder_out, encoder_padding_mask, incremental_state):
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            mask_future_timesteps=True,
            incremental_state=incremental_state,
            need_weights=False,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        attn = None
        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.bias, 0.)
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad, learned=False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings + padding_idx + 1, embedding_dim, padding_idx, left_pad)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad, num_embeddings + padding_idx + 1)
    return m

@register_model_architecture('sty_transformer', 'sty_transformer')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', True)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    # CNN args
    args.cnn_filters = getattr(args, 'cnn_filters', 100)
    args.cnn_dropout = getattr(args, 'cnn_dropout', 0.5)
