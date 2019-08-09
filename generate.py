#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch

from fairseq import bleu, data, options, progress_bar, tasks, tokenizer, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_generator import SequenceGenerator
from fairseq.sequence_scorer import SequenceScorer
from fairseq.data.data_utils import collate_tokens

from nltk.translate.bleu_score import sentence_bleu

import json

def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset, aligned=False)
    print('| {} {} {} examples'.format(args.data, args.gen_subset, len(task.dataset(args.gen_subset))))

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _ = utils.load_ensemble_for_inference(args.path.split(':'), task, model_arg_overrides=eval(args.model_overrides))
    first_model = models[0]

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Initialize generator
    gen_timer = StopwatchMeter()
    if args.score_reference:
        translator = SequenceScorer(models, task.target_dictionary)
    else:
        translator = SequenceGenerator(
            models, task.target_dictionary, beam_size=args.beam,
            stop_early=(not args.no_early_stop), normalize_scores=(not args.unnormalized),
            len_penalty=args.lenpen, unk_penalty=args.unkpen,
            sampling=args.sampling, sampling_topk=args.sampling_topk, minlen=args.min_len,
        )

    if use_cuda:
        translator.cuda()

    for data_idx in [0, 1]:

        # Load dataset (possibly sharded)
        itr = data.EpochBatchIterator(
            dataset=task.dataset(args.gen_subset)[data_idx],
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences,
            max_positions=models[0].max_positions(),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=8,
            num_shards=args.num_shards,
            shard_id=args.shard_id,
        ).next_epoch_itr(shuffle=False)

        # Generate and compute BLEU score
        scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
        num_sentences = 0
        has_target = True
        res = []
        out_obj = []
        with progress_bar.build_progress_bar(args, itr) as t:
            if args.score_reference:
                translations = translator.score_batched_itr(t, cuda=use_cuda, timer=gen_timer)
            else:
                translations = translator.generate_batched_itr(
                    t, maxlen_a=args.max_len_a, maxlen_b=args.max_len_b,
                    cuda=use_cuda, timer=gen_timer, prefix_size=args.prefix_size,
                    to_trg=(data_idx==0),
                )

            wps_meter = TimeMeter()
            for sample_id, src_tokens, target_tokens, hypos in translations:

                # sample out dict
                sample_out_dict = {}

                # Process input and ground truth
                has_target = target_tokens is not None
                target_tokens = target_tokens.int().cpu() if has_target else None

                # Either retrieve the original sentences or regenerate them from tokens.
                if align_dict is not None:
                    src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                    target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
                else:
                    src_str = src_dict.string(src_tokens, args.remove_bpe)
                    if has_target:
                        target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

                sample_out_dict['source'] = src_str
                if has_target:
                    sample_out_dict['target'] = target_str

                if not args.quiet:
                    print('S-{}\t{}'.format(sample_id, src_str))
                    if has_target:
                        print('T-{}\t{}'.format(sample_id, target_str))


                # Process top predictions
                preds = []

                sample_out_dict['translations'] = []
                sample_out_dict['gen_scores'] = []
                sample_out_dict['class_scores'] = []
                sample_out_dict['oracle_scores'] = []

                for i, hypo in enumerate(hypos[:min(len(hypos), args.nbest)]):
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo['tokens'].int().cpu(),
                        src_str=src_str,
                        alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.remove_bpe,
                    )
                    sample_out_dict['translations'].append(hypo_str)
                    sample_out_dict['gen_scores'].append(hypo['score'])

                    # res.append((sample_id.item(), hypo_str, hypo['score']))
                    preds.append([hypo['score'], hypo_str, sample_id.item()])

                    # oracle_score
                    # oracle_score = sentence_bleu([target_str.split()], hypo_str.split())
                    # sample_out_dict['oracle_scores'].append(oracle_score)
                    # if args.oracle_score:
                    #     if has_target: # score the prediction
                    #         # replace the hypo score with the testing one
                    #         preds[-1][0] = oracle_score
                    #     else:
                    #         print("# WARNING: Not target to compute oracle")

                    # disc_score
                    padded_hypo_tokens = collate_tokens([hypo['tokens']],
                                pad_idx=first_model.src_dict.pad(),
                                eos_idx=first_model.src_dict.eos(),
                                left_pad=False, min_size=5,
                                )
                    # print("padded_hypo_tokens.size", padded_hypo_tokens.size())
                    # print(models[0].discriminator.pred(padded_hypo_tokens)[0].size())
                    disc_score = models[0].discriminator.pred(padded_hypo_tokens)[0][0][1-data_idx].item()
                    sample_out_dict['class_scores'].append(disc_score)
                    if args.disc_score:
                        if hasattr(first_model, 'discriminator'):

                            preds[-1][0] = -float("inf") if disc_score < 0.5 else preds[-1][0]
                            # print("{}:{}".format(hypo_str, preds[-1][0]))
                        else:
                            print("# WARNING: No discriminator to score")

                    if not args.quiet:
                        print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str))
                        print('P-{}\t{}'.format(
                            sample_id,
                            ' '.join(map(
                                lambda x: '{:.4f}'.format(x),
                                hypo['positional_scores'].tolist(),
                            ))
                        ))

                        if args.print_alignment:
                            print('A-{}\t{}'.format(
                                sample_id,
                                ' '.join(map(lambda x: str(utils.item(x)), alignment))
                            ))

                    # Score only the top hypothesis
                    if has_target and i == 0:
                        if align_dict is not None or args.remove_bpe is not None:
                            # Convert back to tokens for evaluation with unk replacement and/or without BPE
                            target_tokens = tokenizer.Tokenizer.tokenize(
                                target_str, tgt_dict, add_if_not_exist=True)
                        scorer.add(target_tokens, hypo_tokens)

                preds = sorted(preds, reverse=True)
                res.append((preds[0][2], preds[0][1], preds[0][0]))

                wps_meter.update(src_tokens.size(0))
                t.log({'wps': round(wps_meter.avg)})
                num_sentences += 1

                out_obj.append(sample_out_dict)

        if args.output_path is not None:
            if data_idx == 0:
                output_suffix = '.' + args.source_lang + '-' + args.target_lang
            else:
                output_suffix = '.' + args.target_lang + '-' + args.source_lang
            out = open(args.output_path+output_suffix, 'w')
            res = sorted(res)
            for r in res:
                if args.score_reference:
                    out.write("{} ||| {:.4f}\n".format(r[1], r[2]))
                else:
                    out.write(r[1] + '\n')

            with open(args.output_path+output_suffix+'.json', 'w') as f_out:
                f_out.write(json.dumps(out_obj, ensure_ascii=False, sort_keys=False, indent=4))

    print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if has_target:
        print('| Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))


if __name__ == '__main__':
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)