#!/bin/sh
ls *.keras | parallel -j5 'KERAS_BACKEND=jax JAX_PLATFORMS=cpu alkaid convert -unc 0 -lc 2.1 -c 1.5 {} rtl/{.} --no-shreg'
