#!/bin/bash

#SBATCH --job-name=megatron
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

DATASET="${DIR}/../transformers-benchmarks/data/gpt2-sample_text_document"


options=" \
	--tensor-model-parallel-size 8 \
	--pipeline-model-parallel-size 1 \
        --num-layers 24 \
        --hidden-size 6144 \
        --num-attention-heads 48 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
	--micro-batch-size 4 \
	--global-batch-size 8 \
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
	--sequence-parallel "

run_cmd="python3 -u ${DIR}/pretrain_gpt.py $@ ${options}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN

var_UCX_NET_DEVICES=mlx5_0:1
var_NCCL_IB_HCA="=mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_14,mlx5_15,mlx5_16,mlx5_17,mlx5_9,mlx5_10,mlx5_11,mlx5_12"

export UCX_TLS=ud,self,sm \
       NCCL_DEBUG=INFO \
       NCCL_IB_CUDA_SUPPORT=1 \
       NCCL_IB_SL=0 \
       NCCL_IB_TC=41 \
       NCCL_IB_QPS_PER_CONNECTION=4 \
       UCX_NET_DEVICES=${var_UCX_NET_DEVICES} \
       HCOLL_ENABLE_MCAST_ALL=0 \
       coll_hcoll_enable=0 \
       NCCL_IB_GID_INDEX=3 \
       NCCL_IB_HCA="${var_NCCL_IB_HCA}" \
       NCCL_ALGO=Ring \
       OMPI_MCA_coll=^hcoll \
       OMPI_MCA_pml=ucx \
       ENROOT_RESTRICT_DEV=y

DEVICE_MOUNT="/dev/infiniband/rdma_cm:/dev/infiniband/rdma_cm"

for i in {0..17}
do
  DEVICE_MOUNT="$DEVICE_MOUNT,/dev/infiniband/uverbs$i:/dev/infiniband/uverbs$i"
done

echo $DEVICE_MOUNT

srun -l \
     --container-image /fsx/enroot_images/nvcr.io+ea-bignlp+bignlp-training+23.03-py3.sqsh \
     --container-mounts /fsx:/fsx,$DEVICE_MOUNT \
     --exclusive \
     --output=$DIR/logs/%x_%j_$DATETIME.log bash -c "${run_cmd}"

set +x

