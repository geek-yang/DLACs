#!/bin/bash
#SBATCH -N 1             # request 1 node
#SBATCH -p gpu          # use gpu     
#SBATCH -t 3-01:00:00    # set the wall clock time
python /home/lwc16308/BayesArctic/DLACs/examples/map_BayesConvLSTM_SIC_Barents_param_verifySIC_train.py
