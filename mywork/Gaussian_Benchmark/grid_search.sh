#!/bin/bash

#SBATCH --job-name=scan_3
#SBATCH -p GPU
# requesting one node
# SBATCH -N1
# requesting 12 cpus
# SBATCH -n12
#SBATCH --ntasks=1                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --cpus-per-task=12           # nombre de coeurs CPU par tache (un quart du noeud ici)
#SBATCH --gres=gpu:v100:1            # requesting GPUs
#SBATCH --mail-use=henry.aldridge.23@ucl.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --output=/share/gpu1/haldridge/data/proxnest/logs/scan/mh_false/%x_%j.out
#SBATCH --error=/share/gpu1/haldridge/data/proxnest/logs/scan/mh_false/%x_%j.err



# Load conda and activate environment
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate proxnest

set -x

cd "/share/gpu1/haldridge/repositories/proxnest/mywork/Gaussian Benchmark"
git checkout u/henry-ald/gaussian_benchmark

srun python "/share/gpu1/haldridge/repositories/proxnest/mywork/Gaussian Benchmark/gaussian_benchmark.py" -d 20 200 10 -l "Param Testing/MH_False" --seed 10 -p lv_thinning_init lv_thinning