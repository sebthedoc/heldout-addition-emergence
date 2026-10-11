#!/bin/zsh
# Kierros 2: rakenteelliset pois jätöt vakautettuna.
cd "${0:A:h}"
for mode in diag row commut block oneshot; do for seed in 0 1 2; do
  out=../tulokset/v2_${mode}_s${seed}
  [ -f $out.json ] || ../.venv/bin/python lisays.py --holdout $mode --seed $seed --steps 30000 --warmup 500 --clip 1.0 --out $out
done; done
