#!/bin/bash

types=('RandomSwapPerturbation' 'RandomInsertPerturbation' 'RandomPatchPerturbation')
pcts=(5 10 15 20)
num_copies=(2 4 6 8 10)
trials=(1 2 3 4)


for trial in "${trials[@]}"; do
    for type in "${types[@]}"; do
        for pct in "${pcts[@]}"; do
            for n in "${num_copies[@]}"; do

                python main.py \
                    --results_dir ./results \
                    --target_model vicuna \
                    --attack GCG \
                    --attack_logfile data/GCG/vicuna_behaviors.json \
                    --smoothllm_pert_type $type \
                    --smoothllm_pert_pct $pct \
                    --smoothllm_num_copies $n \
                    --trial $trial

            done     
        done
    done
done
                