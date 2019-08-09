# Overview
This repo contains source code for "Formality Style Transfer with Hybrid Textual Annotations"

# Install
The software is built upon `fairseq`, follow the instruction on this [link](https://github.com/pytorch/fairseq) for installation

# Data
The preprocessed data could be downloaded from [this link]().

# Usage

To replicate our experiment on the GYAFC dataset, see the `pipeline.sh` script for a working example.

The general usage is similar to [fairseq](https://github.com/pytorch/fairseq)
with the same syntax for arguments. The only difference is to make sure the `--task`
is *style-transfer* and `arch` is *sty-transformer*.

Training with supervised paired data: `train.py`

```
usage: train.py [-h] DIR
                --task style_transfer
                --arch sty_transformer [--criterion CRIT] [--max-epoch N]
                [--max-update N] [--clip-norm NORM] [--sentence-avg]
                [--mt-loss-weight] [--classify-loss-weight]
                [--self-recon-loss-weight] [--cycle-recon-loss-weight]
                [--max-source-positions] [--max-target-positions]
                [--encoder-embed-dim] [--encoder-ffn-embed-dim]
                [--decoder-embed-dim] [--decoder-ffn-embed-dim]
                [--encoder-attention-heads] [--decoder-attention-heads]
                [--encoder-layers] [--decoder-layers]
                [--optimizer OPT] [--lr LR_1,LR_2,...,LR_N]
                [--momentum M] [--weight-decay WD]
                [--lr-scheduler LR_SCHEDULER] [--lr-shrink LS] [--min-lr LR]
                [--save-dir DIR]
                [--restore-file RESTORE_FILE] [--save-interval N]
                [--save-interval-updates N] [--keep-interval-updates N]
                [--no-save] [--no-epoch-checkpoints] [--validate-interval N]
```

Generate with trained model: `generate.py`

```
usage: generate.py [-h] DIR
                   --task style_transfer  
                   [--no-progress-bar] [--log-interval N]
                   [--remove-bpe [REMOVE_BPE]] [--beam N]
                   [--nbest N]
                   [--output-path OUTPUT_PATH]
```
