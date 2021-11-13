#!/usr/bin/env bash
#PBS -P w35
#PBS -l walltime=4:30:00
#PBS -l mem=170GB
#PBS -q normal
#PBS -l ncpus=16
#PBS -l wd
#PBS -l storage=gdata/w35+gdata/hh5
#PBS -M l.teckentrup@student.unsw.edu.au
#PBS -m abe
#PBS -l jobfs=1gb

module purge
module load pbs
module load python3/3.9.2
module load cuda/11.4.1
module load cudnn/8.2.2-cuda11.4
module load nccl/2.10.3-cuda11.4
module load openmpi/4.1.1
export LD_PRELOAD=/apps/openmpi/4.1.1/lib/libmpi_cxx.so
cd /g/data/w35/lt0205/research/Australia/RF_tensorflow/
source my-venv/bin/activate

for nattr in {5..10}; do
    python3 RF_LPJ.py --ntrees 350 --nattr ${nattr} --fname 'cpool' --var 'SoilC'
