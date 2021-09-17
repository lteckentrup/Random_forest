#!/usr/bin/env bash
#PBS -P w35
#PBS -l walltime=1:30:00
#PBS -l mem=190GB
#PBS -q gpuvolta
#PBS -l ncpus=12
#PBS -l wd
#PBS -l storage=gdata/w35+gdata/hh5
#PBS -M l.teckentrup@student.unsw.edu.au
#PBS -m abe
#PBS -l ngpus=1
#PBS -l jobfs=100GB
#PBS -l wd
#PBS -N hc
#PBS -W umask=027


module purge

module load cuda/11.4.1
module load cudnn/8.2.2-cuda11.4
source tf-df/bin/activate

python3 pixel_test.py
