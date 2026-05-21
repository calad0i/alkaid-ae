#!/bin/sh
ls *.keras | parallel 'KERAS_BACKEND=jax JAX_PLATFORMS=cpu alkaid convert -unc 0 -lc 6.5 -c 5 {} rtl/{.} --no-shreg -xopt'
for f in rtl/*/build_vivado_prj.tcl; do sed -i 's/-global_retiming on/-global_retiming on -max_dsp 0/' $f; done
