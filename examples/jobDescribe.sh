#!/bin/bash
#SBATCH -n 1             # request 1 node
#SBATCH -p gpu_short     # use gpu
#SBATCH --mem=32.5G       
#SBATCH -t 01:00:00    # set the wall clock time
python /home/lwc16308/BayesArctic/DLACs/examples/map_BayesConvLSTM_SIC_Barents_param_verifySIC.py
