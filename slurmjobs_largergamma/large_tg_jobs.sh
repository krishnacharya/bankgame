#!/bin/bash
#SBATCH -J tglarge3
#SBATCH --array=1-320
#SBATCH -A gts-jziani3
#SBATCH --mem-per-cpu=2G
#SBATCH --time=2:00:00
#SBATCH --output=./Sbatch-tg/tglarge3.%A\_%a.out
#SBATCH --error=./Sbatch-tg/tglarge3.%A\_%a.error

mkdir -p ./Sbatch-tg

module load anaconda3/2022.05.0.1
conda activate bankgame

n=$SLURM_ARRAY_TASK_ID
iteration=$(sed -n "${n}p" tg_3gammas.csv)  # Get the nth line (1-indexed) of the file
echo "Parameters for iteration: ${iteration}"

mu=$(echo ${iteration} | cut -d "," -f 1)
sigma=$(echo ${iteration} | cut -d "," -f 2)

# Convert comma-separated gamma values into space-separated values
gamma_values=$(echo ${iteration} | cut -d "," -f 3- | tr ',' ' ')

horizon=200000
eta=0.1
num_startprofiles=5

# Pass gamma values as space-separated arguments
python tg_larger_run.py --mu ${mu} --sigma ${sigma} --gamma ${gamma_values} --horizon ${horizon} --eta ${eta} --num_startprofiles ${num_startprofiles}
