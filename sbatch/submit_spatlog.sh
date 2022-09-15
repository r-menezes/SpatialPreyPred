#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH -p cpu_small
#SBATCH -J spatlog
#SBATCH --time=00:20:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user="menezes.santos.rafael@gmail.com"

# Exibe os nos alocados para o Job
echo $SLURM_JOB_NODELIST

# Export input/output variables
export INPUT="spatlog_results.py"
export OUTPUT="results.zip"

# load anaconda module
module load anaconda3
source activate spatlog

# execute jobnanny
job-nanny python spatlog_results.py
