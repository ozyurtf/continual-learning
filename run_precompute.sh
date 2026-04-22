#!/bin/bash
mkdir -p logs

for i in 0; do
    CUDA_VISIBLE_DEVICES=$i python precompute.py \
        --split train --img_h 160 --img_w 240 \
        --num_workers 1 --worker_id $i \
        > logs/precompute_worker$i.log 2>&1 &
    echo "Started worker $i on GPU $i (PID $!)"
done

wait
echo "All precompute workers done."
