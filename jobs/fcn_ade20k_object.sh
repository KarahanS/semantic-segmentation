#!/bin/bash
#SBATCH --job-name=obj-ade20k-linear
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --partition=gpu-h200-141g-short,gpu-h100-80g-short,gpu-h100-80g,gpu-v100-32g,gpu-a100-80g,gpu-v100-16g
#SBATCH --mem=64GB
#SBATCH --output=slurm_outputs/fit/%x.%A_%a.out
#SBATCH --error=slurm_outputs/fit/%x.%A_%a.err
#SBATCH --array=0              # ← update to 0‑$(( ${#CKPTS[@]} - 1 ))

set -euo pipefail
GPUS=${SLURM_GPUS_ON_NODE:-1}

# -----------------------------------------------------------------------
# 1)  CHECKPOINTS (+ matching encoder names, if they differ)
# -----------------------------------------------------------------------
CKPTS=(
    # /scratch/work/saritak1/checkpoints/coco-v137/checkpoint.pth.timm
  # /scratch/work/saritak1/checkpoints/odis-coco-idea_C_both-obj-thr:20-loss:Lp_Lo-gc-lc-border:0.15/checkpoint.pth.timm
/scratch/work/saritak1/checkpoints/coco_cribo_v137_Lnn_Lo_Lp/checkpoint.timm.pth
  #/scratch/work/saritak1/checkpoints/vidor_odis_Lp_Lo_Lot_1sec_bs32/checkpoint.pth.timm
  #/scratch/work/saritak1/checkpoints/vidor_odis_Lp_Lo_1sec_bs32/checkpoint.pth.timm
)

# If every checkpoint uses the *same* encoder (e.g. vit_small_patch16_224)
# you can leave ENCODERS with a single entry; it will be reused.
ENCODERS=(
 vit_small_patch16_224
)

# -----------------------------------------------------------------------
# 2)  SELECT COMBINATION FOR THIS ARRAY TASK
# -----------------------------------------------------------------------
NUM_CKPTS=${#CKPTS[@]}

if (( SLURM_ARRAY_TASK_ID >= NUM_CKPTS )); then
  echo "Array‑index $SLURM_ARRAY_TASK_ID ≥ $NUM_CKPTS"; exit 1; fi

CKPT=${CKPTS[$SLURM_ARRAY_TASK_ID]}
ENCODER=${ENCODERS[$SLURM_ARRAY_TASK_ID]:-${ENCODERS[0]}}

echo "─── task $SLURM_ARRAY_TASK_ID / $((NUM_CKPTS-1))"
echo "     → checkpoint: $CKPT"
echo "     → encoder   : $ENCODER"

# -----------------------------------------------------------------------
# 3)  OUTPUT LOCATION
# -----------------------------------------------------------------------
CKPT_FOLDER=$(basename "$(dirname "$CKPT")")
RUN_DIR=/scratch/work/saritak1/segmentation/output_linear/${CKPT_FOLDER}
mkdir -p "$RUN_DIR"

# -----------------------------------------------------------------------
# 4)  ENVIRONMENT
# -----------------------------------------------------------------------
module load mamba
eval "$(conda shell.bash hook)"
conda activate /scratch/work/saritak1/conda/miniconda3/envs/vfm

export WANDB_API_KEY=$(cat ~/.wandb_key)       # set in ~/.bashrc or via sbatch --export
export WANDB_PROJECT="ade20k-linear"
export WANDB_ENTITY="agape"
export WANDB_NAME="${CKPT_FOLDER}-run${SLURM_ARRAY_TASK_ID}"
export WANDB_DIR="$RUN_DIR/wandb"               # logs go inside your run folder
export WANDB_RESUME="allow"                     # so requeues don’t start a new run


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
PORT=$(( 29500 + SLURM_ARRAY_TASK_ID ))
nvidia-smi || true

# -----------------------------------------------------------------------
# 5)  TRAIN
# -----------------------------------------------------------------------
srun python main.py fit \
  -c configs/ade20k_linear_semantic.yaml \
  --model.network.encoder_name "$ENCODER" \
  --model.network.ckpt_path    "$CKPT" \
  --model.freeze_encoder       True \
  --root                       /scratch/work/saritak1/datasets \
  --data.num_workers           8 \
  --trainer.default_root_dir   "$RUN_DIR" \
  --trainer.logger.tensorboard False \
  --trainer.logger.save_dir    "$RUN_DIR"/tb \

echo "✓ Done (ckpt=$CKPT)"
