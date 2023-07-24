WARMUP_UPDATES=4000     
LR=0.0005
MAX_TOKENS=15000
UPDATE_FREQ=3
ARCH_NAME=model05-jsdiv
SAVE_PATH=./model/$ARCH_NAME
LOG_PATH=./logs/$ARCH_NAME.log.txt
BIN_PATH=../data/BiNews/mix_data-bin
GPU=6,7
# CUDA_VISIBLE_DEVICES=$GPU python -m debugpy --listen 127.0.0.1:5678 --wait-for-client train.py $BIN_PATH \
CUDA_VISIBLE_DEVICES=$GPU python train.py $BIN_PATH \
    --save-dir $SAVE_PATH \
    --user-dir ./my-module \
    --restore-file ${SAVE_PATH}/checkpoint_last.pt \
    --max-tokens $MAX_TOKENS \
    --task cls_translation \
    --source-lang source --target-lang target \
    --max-source-positions 512 \
    --max-target-positions 150 \
    --truncate-source \
    --share-decoder-input-output-embed \
    --share-all-embeddings \
    --required-batch-size-multiple 1 \
    --arch  cls_transformer \
    --criterion label_smoothed_cross_entropy_with_jsdiv \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.998)" --adam-eps 1e-09 \
    --clip-norm 2 \
    --fp16 \
    --patience 5 \
    --lr-scheduler inverse_sqrt --lr $LR  --warmup-updates $WARMUP_UPDATES \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters  2>&1 | tee -a $LOG_PATH