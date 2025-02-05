#!/bin/sh
#PBS -N cnn
#PBS -q workq
#PBS -e cnn.err
#PBS -o cnn.out
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=20

cd $PBS_O_WORKDIR

echo "Current Directory: $(pwd)"
echo "Environment:" 

source /etc/profile.d/modules.sh

conda activate gw_lens

python3 generate_training_data.py 20 20000
