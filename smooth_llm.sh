#!/bin/bash

perturbation_type='RandomSwapPerturbation'
perturbation_percentage=10
num_smoothing_copies=10

python main.py \
    --config=configs/vicuna.py \
    --config.logfile=data/GCG/vicuna_behaviors.json \
    --config.perturbation_type=$perturbation_type \
    --config.perturbation_percentage=$perturbation_percentage \
    --config.num_smoothing_copies=$num_smoothing_copies