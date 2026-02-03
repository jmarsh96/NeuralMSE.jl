#!/bin/bash
#SBATCH --job-name=neuralmse-train
#SBATCH --output=logs/neuralmse_%j.out
#SBATCH --error=logs/neuralmse_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -e

module purge; module load bluebear
module load bear-apps/2023a
module load Julia/1.11.4-linux-x86_64

mkdir -p logs

# Print job info
echo "=============================================="
echo "NeuralMSE Training Job"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_JOB_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "=============================================="

# Set models directory (required for development/training)
export NEURALMSE_MODELS_PATH="${NEURALMSE_MODELS_PATH:-$(pwd)/trained_models}"
mkdir -p "$NEURALMSE_MODELS_PATH"
echo "Models directory: $NEURALMSE_MODELS_PATH"

echo ""
echo "Checking dependencies..."
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run the training script
echo ""
echo "Starting distributed training..."
echo "=============================================="

julia --project=. scripts/train_models_distributed.jl

echo ""
echo "=============================================="
echo "Training complete!"
echo "End time: $(date)"
echo "=============================================="
