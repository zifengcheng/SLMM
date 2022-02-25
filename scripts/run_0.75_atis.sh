#!/usr/bin bash

for s in 0 1 2 3 4 5 6 7 8 9
do 
    python SLMM.py \
        --dataset atis \
        --known_cls_ratio 0.75 \
        --labeled_ratio 1.0 \
        --seed $s \
        --freeze_bert_parameters \
        --beta 0.3 \
        --gamma 0.9 \
        --lr_boundary 2e-5
done
