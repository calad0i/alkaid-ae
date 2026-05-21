#!/bin/sh

cd openml
micromamba run -n alkaid-ae ./convert.sh
cd ..

cd linear_ml4ps_models
micromamba run -n alkaid-ae ./convert.sh
cd ..

cd mha_ml4ps_models
micromamba run -n alkaid-ae ./convert.sh
cd ..

cd qbdt
KERAS_BACKEND=jax JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 micromamba run -n alkaid-ae python mnist.py
KERAS_BACKEND=jax JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 micromamba run -n alkaid-ae python jsc.py
cd ..

cd jedi
micromamba run -n legacy-hls4ml python convert.py