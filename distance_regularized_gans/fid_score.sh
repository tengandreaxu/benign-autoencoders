#!/bin/bash

set -x;

python3 -m pytorch_fid results/distance_regularized_gan_d1/z_dim=1/generated data/resized_celebA/celebA
python3 -m pytorch_fid results/distance_regularized_gan_d1/z_dim=10/generated data/resized_celebA/celebA
python3 -m pytorch_fid results/distance_regularized_gan_d1/z_dim=50/generated data/resized_celebA/celebA
python3 -m pytorch_fid results/distance_regularized_gan_d1/z_dim=100/generated data/resized_celebA/celebA
python3 -m pytorch_fid results/distance_regularized_gan_d1/z_dim=500/generated data/resized_celebA/celebA;
python3 -m pytorch_fid results/distance_regularized_gan_d1/z_dim=1000/generated data/resized_celebA/celebA