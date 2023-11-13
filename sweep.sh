#!/bin/bash

# types=('RandomSwapPerturbation' 'RandomInsertPerturbation' 'RandomPatchPerturbation')
# pcts=(5 10 15 20)
# num_copies=(2 4 6 8 10)
# trials=(1 2 3 4)
types=('RandomSwapPerturbation')
pcts=(5 10)
num_copies=(10)
trials=(1)
target_model=vicuna
results_root=./results

for trial in "${trials[@]}"; do
    for type in "${types[@]}"; do
        for pct in "${pcts[@]}"; do
            for n in "${num_copies[@]}"; do

                dir=$results_root/$target_model/trial-$trial/n-$n-type-$type-pct-$pct
                echo $dir

                python main.py \
                    --results_dir $dir \
                    --target_model $target_model \
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
                