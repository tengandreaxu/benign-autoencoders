#!/bin/bash

set -x;
set -e;

# *************
# Simulated Data 
# *************
python3 experiments/simulated_data.py;


# *************
# MNIST
# *************
python3 experiments/bae_on_the_two_nist_cnn.py \
    -n 0.0 -n 0.25 -n 0.5 -n 0.75 \
    -dw 0.9 \
    -nw 0.1 \
    -lr 0.001 \
    -tp 0 -tp 1 -tp 2 -tp 3 \
    -e 20 \
    --dataset mnist;

# *************
# FMNIST
# *************
python3 experiments/bae_on_the_two_nist_cnn.py \
    -n 0.0 -n 0.25 -n 0.5 -n 0.75 \
    -dw 0.9 \
    -nw 0.1 \
    -lr 0.001 \
    -tp 0 -tp 1 -tp 2 -tp 3 \
    -e 20 \
    --dataset fmnist;