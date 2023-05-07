#!/bin/bash

#SBATCH --job-name=megatron
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

DATASET="${DIR}/../transformers-benchmarks/data/gpt2-sample_text_document"


options=" \
	--tensor-model-parallel-size 8 \
	--pipeline-model-parallel-size 1 \
        --num-layers 48 \
        --hidden-size 6144 \
        --num-attention-heads 64 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
	--micro-batch-size 4 \
	--global-batch-size 4 \
	--train-iters 50 \
        --lr 6.0e-6 \
	--lr-warmup-fraction 0.5 \
	--min-lr 6.0e-7 \
        --lr-decay-style cosine \
        --log-interval 10 \
        --eval-iters 100 \
        --eval-interval 1000 \
	--data-path ${DATASET} \
	--vocab-file "${DIR}/examples/gpt2-vocab.json" \
	--merge-file "${DIR}/examples/gpt2-merges.txt" \
	--save-interval 1000 \
        --split 98,2,0 \
        --clip-grad 1.0 \
	--weight-decay 0.1 \
	--adam-beta1 0.9 \
	--adam-beta2 0.95 \
	--init-method-std 0.006 \
        --bf16 \
	--use-flash-attn \
	--transformer-impl transformer_engine \
	--sequence-parallel "

run_cmd="python3 -u ${DIR}/pretrain_gpt.py $@ ${options}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN

#export MASTER_ADDR=127.0.0.1
#export MASTER_PORT=9783 
#export WORLD_SIZE=8

srun -l \
     --container-image /run/enroot/pt.sqsh \
     --container-mounts /run:/fsx \
     --container-mount-home \
     --output=$DIR/logs/%x_%j_$DATETIME.log bash -c "${run_cmd}"

set +x

