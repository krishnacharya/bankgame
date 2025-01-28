#!/bin/bash
#SBATCH -J TG_2gam
#SBATCH --array=1-240
#SBATCH -A gts-jziani3
#SBATCH --mem-per-cpu=2G
#SBATCH --time=2:00:00
#SBATCH --output=./Sbatch-tg/tg2gam.%A\_%a.out
#SBATCH --error=./Sbatch-tg/tg2gam.%A\_%a.error

mkdir -p ./Sbatch-tg

module load anaconda3/2022.05.0.1
conda activate bankgame

n=$SLURM_ARRAY_TASK_ID
iteration=`sed -n "${n} p" tg_instances_slurm.csv`  # Get n-th line (1-indexed) of the file
echo "parameters for iteration: ${iteration}"

mu=$(echo ${iteration} | cut -d "," -f 1)
sigma=$(echo ${iteration} | cut -d "," -f 2)
gamma_l=$(echo ${iteration} | cut -d "," -f 3)
gamma_h=$(echo ${iteration} | cut -d "," -f 4)
horizon=100000
eta=0.1
num_startprofiles=5

python mainrun_instance_tg.py --mu ${mu} --sigma ${sigma} --gamma_l ${gamma_l} --gamma_h ${gamma_h} --horizon ${horizon} --eta ${eta} --num_startprofiles ${num_startprofiles}
