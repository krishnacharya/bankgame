#!/bin/bash
#SBATCH -J Puf_2gam
#SBATCH --array=1-2
#SBATCH -A gts-jziani3
#SBATCH --mem-per-cpu=2G
#SBATCH --time=2:00:00
#SBATCH --output=./Sbatch-puf/Puf2gam.%A\_%a.out
#SBATCH --error=./Sbatch-puf/Puf2gam.%A\_%a.error

mkdir -p ./Sbatch-puf

module load anaconda3/2022.05.0.1
conda activate bankgame

n=$SLURM_ARRAY_TASK_ID
iteration=`sed -n "${n} p" puf_instances.csv`  # Get n-th line (1-indexed) of the file
echo "parameters for iteration: ${iteration}"

gamma_l=$(echo ${iteration} | cut -d "," -f 1)
gamma_h=$(echo ${iteration} | cut -d "," -f 2)
horizon=10000
eta=0.1
num_startprofiles=5

python mainrun_instance_puf.py --gamma_l ${gamma_l} --gamma_h ${gamma_h} --horizon ${horizon} --eta ${eta} --num_startprofiles ${num_startprofiles}
