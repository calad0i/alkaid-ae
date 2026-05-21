# Artifact Evaluation for Alkaid

Requirements:
- micromamba (Can be replaced with any venv manager, modify the commands accordingly in that case)
- gnu_parallel (not to be confused with parallel from moreutils. Check with `parallel --version`)
- Reasonably updated Linux distribution

Commands to reproduce major results are as follows. The scripts shall be run from the root of the repository.

Creating environment:
```bash
micromamba create --file env1.yml # Main environment, with HGQ and QXGBoost
micromamba create --file env2.yml # For qkeras and hls4ml
```

Converting to RTL and functional testing:
```bash
bash convert_models.sh
```

Run synthesis. Make `$NUMBER_OF_PROCESS` is not too large to cause OOM.
```bash
ls -d */rtl/* | parallel -j$NUMBER_OF_PROCESS 'cd {} && vivado -mode batch -source build_vivado_prj.tcl > synth.log'
```

Report:
```bash
bash report.sh
```
