#!/bin/bash
#SBATCH --job-name=ageing_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --chdir=/hpc/dla_lti/jvanmeenen/ageingM
#SBATCH --output=logs/ageing_general2050_%j.log

set -euo pipefail

echo "Job started at: $(date)"
echo "Running on: $(hostname)"
echo "Working directory: $(pwd)"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK}"

export OPENBLAS_NUM_THREADS=1
export JULIA_CPU_TARGET=generic
export JULIA_NUM_PRECOMPILE_TASKS=1

echo "Julia version:"
"$HOME/.juliaup/bin/julia" --version

echo "Preparing project dependencies..."
srun "$HOME/.juliaup/bin/julia" \
    --startup-file=no \
    --project=. \
    --threads="${SLURM_CPUS_PER_TASK}" \
    -e 'import Pkg; Pkg.instantiate(); Pkg.precompile()'

echo "Starting simulation at: $(date)"
srun "$HOME/.juliaup/bin/julia" \
    --startup-file=no \
    --project=. \
    --threads="${SLURM_CPUS_PER_TASK}" \
    scripts/runs/ageing_general_2050.jl

echo "Job finished at: $(date)"