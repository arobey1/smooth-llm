#!/bin/bash

perturbation_type='RandomSwapPerturbation'
perturbation_percentage=10
num_smoothing_copies=10

python main.py \
    --results_dir ./results \
    --target_model vicuna \
    --attack GCG \
    --attack_logfile data/GCG/vicuna_behaviors.json \
    --smoothllm_pert_type $perturbation_type \
    --smoothllm_pert_pct $perturbation_percentage \
    --smoothllm_num_copies $num_smoothing_copies