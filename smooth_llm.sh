#!/bin/bash

if [ $1 == "vicuna" ]; then
    config=configs/vicuna.py
    logfile=data/vicuna_behaviors.json
elif [ $1 == "llama" ]; then
    config=configs/llama2.py
    logfile=data/llama2_behaviors.json
else
    echo "Invalid argument $1.  Please choose from ['vicuna', 'llama']."
    exit 1
fi

perturbation_type='RandomSwapPerturbation'
perturbation_percentage=10
num_smoothing_copies=10
results_dir='results'

python main.py \
    --config=$config \
    --config.logfile=$logfile \
    --config.perturbation_type=$perturbation_type \
    --config.perturbation_percentage=$perturbation_percentage \
    --config.num_smoothing_copies=$num_smoothing_copies \
    --config.smoothing_results_dir=$results_dir