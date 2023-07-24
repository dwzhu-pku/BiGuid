DATA_PATH="../data/BiNews/mix_data-bin"
LEN_PEN=$1
GPU=$2
NUM=$3
ARCH_NAME=model05-jsdiv
MODEl_PATH=./model/$ARCH_NAME/checkpoint${NUM}.pt
MAX_TOKENS=50000
RESULT_PATH=./result/$ARCH_NAME/${NUM}/lenpen_${LEN_PEN}

mkdir $RESULT_PATH

# CUDA_VISIBLE_DEVICES=$GPU python generate.py $DATA_PATH --gen-subset "test" \
#     --user-dir ./my-module \
#     --task cls_translation \
#     --source-lang source --target-lang target \
#     --path $MODEl_PATH \
#     --no-repeat-ngram-size 3 \
#     --max-len-b 150 \
#     --min-len 10 \
#     --lenpen $LEN_PEN \
#     --beam 5 \
#     --max-source-positions 512 \
#     --truncate-source \
#     --results-path ${RESULT_PATH} \
#     --max-tokens $MAX_TOKENS \
#     --skip-invalid-size-inputs-valid-test

python ../ZTools/zhMergeChar.py $RESULT_PATH
python ../ZTools/map.py $RESULT_PATH
files2rouge $RESULT_PATH/target.char.en $RESULT_PATH/hypo.char.en  > $RESULT_PATH/rougescore