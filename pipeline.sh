# Example script to train and generate

PRE_EPOCH=30;
NUM_UPDATES=60000;
ANNEAL_MAX_STEPS=15000;
ANNEAL_STEPS=8000;
ANNEAL_SPEED=8000;
MAX_TOKENS=3000;
# SUFFIX="";
MAX_POSITIONS=400;
DOM=Entertainment_Music_comb;

CLS_W=0.2;
SLF_RECON_W=0.2;
CYC_RECON_W=1.0;

PROCESSED_DATA_PATH=processed/"$DOM"
EXP_PATH="exp/$DOM";
SAVED_MODEL_PATH="$EXP_PATH"/model;

mkdir -p "$EXP_PATH";

# python train.py $PROCESSED_DATA_PATH \
#         --raw-text \
#         --distributed-world-size 1 \
#         --source-lang informal --target-lang formal \
#         --arch sty_transformer \
#         --task style_transfer \
#         --mt-loss-weight 1.0 --classify-loss-weight "$CLS_W" \
#         --self-recon-loss-weight "$SLF_RECON_W" --cycle-recon-loss-weight "$CYC_RECON_W" \
#         --max-source-positions "$MAX_POSITIONS" --max-target-positions "$MAX_POSITIONS" \
#         --encoder-embed-dim 256 --encoder-ffn-embed-dim 1024 \
#         --decoder-embed-dim 256 --decoder-ffn-embed-dim 1024 \
#         --encoder-attention-heads 4 --decoder-attention-heads 4 \
#         --encoder-layers 2 --decoder-layers 2 \
#         --left-pad-source True \
#         --attention-dropout 0.2 --relu-dropout 0.2 --dropout 0.2 \
#         --share-all-embeddings \
#         --criterion style_transfer_train --label-smoothing 0.1 \
#         --optimizer adam --adam-betas '(0.9, 0.997)' --max-tokens "$MAX_TOKENS" \
#         --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 8000 \
#         --lr 0.0008 --min-lr 1e-09 --pre-train-max-epoch "$PRE_EPOCH" --max-update "$NUM_UPDATES" \
#         --clip-norm 5.0 \
#         --temp-anneal-step "$ANNEAL_STEPS" --temp-anneal-speed "$ANNEAL_SPEED" \
#         --temp-anneal-max-step "$ANNEAL_MAX_STEPS" \
#         --log-format simple --no-epoch-checkpoints \
#         --restore-file 'checkpoint_last.pt' \
#         --save-dir $SAVED_MODEL_PATH \
#         | tee "$EXP_PATH"/run.log.txt;
        
BEAM_SIZE=12;
TEST_OUT_PATH="$EXP_PATH"/pred.bs_"$BEAM_SIZE".discr.txt;
# rm $TEST_OUT_PATH;
SAVED_MODEL_PATH="$EXP_PATH"/model
if [ ! -f "$TEST_OUT_PATH.informal-formal" ]; then
    python generate.py \
            $PROCESSED_DATA_PATH \
            --task style_transfer \
            --nbest "$BEAM_SIZE" \
            --raw-text --disc-score \
            --path "$SAVED_MODEL_PATH"/checkpoint_best.pt \
            --batch-size 128 --beam "$BEAM_SIZE" --remove-bpe \
            --output-path "$TEST_OUT_PATH" | tee "$EXP_PATH"/generate.log.txt;

else
    echo "Prediction exists. Skipping predict...";
fi
