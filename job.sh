#!/bin/sh
#PBS -N cnn
#PBS -q gpu
#PBS -e cnn.err
#PBS -o cnn.out
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=2:ngpus=1

cd $PBS_O_WORKDIR

echo "Current Directory: $(pwd)"
echo "Environment:" 

source /etc/profile.d/modules.sh

conda init
conda activate gw_lens

python3 modified_cnn.py
