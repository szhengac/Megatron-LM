#!/bin/bash


DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs


DATASET="../transformers-benchmarks/data/gpt2-sample_text_document"


options=" \
	--tensor-model-parallel-size 4 \
	--pipeline-model-parallel-size 2 \
        --num-layers 50 \
        --hidden-size 7168 \
        --num-attention-heads 56 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
	--micro-batch-size 1 \
	--global-batch-size 32 \
	--train-iters 50 \
        --lr 6.0e-6 \
	--lr-warmup-fraction 0.5 \
	--min-lr 6.0e-7 \
        --lr-decay-style cosine \
        --log-interval 10 \
        --eval-iters 100 \
        --eval-interval 1000 \
	--data-path ${DATASET} \
	--vocab-file "examples/gpt2-vocab.json" \
	--merge-file "examples/gpt2-merges.txt" \
	--save-interval 1000 \
        --split 98,2,0 \
        --clip-grad 1.0 \
	--weight-decay 0.1 \
	--adam-beta1 0.9 \
	--adam-beta2 0.95 \
	--init-method-std 0.006 \
        --bf16 \
	--use-flash-attn \
	--transformer-impl local \
	--sequence-parallel "

#deepspeed ${DIR}/pretrain_gpt.py $@ ${options}
mpirun --allow-run-as-root -n 8 -N 8 \
--mca pml ^cm --mca btl tcp,self \
--mca btl_tcp_if_exclude lo,docker0 \
--bind-to none \
-x NCCL_DEBUG=WARN \
-x MASTER_ADDR=127.0.0.1 \
-x MASTER_PORT=9783 \
-x WORLD_SIZE=8 \
-x CUDA_DEVICE_MAX_CONNECTIONS=1 \
-x NVTE_FLASH_ATTN=1 \
python3 ${DIR}/pretrain_gpt.py $@ ${options}


set +x

