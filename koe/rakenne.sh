#!/bin/zsh
# Rakenteelliset pois jätöt (3 siementä), ajetaan kun satunnaisen osuuden sweep on valmis.
cd "${0:A:h}"
for mode in diag row commut block newsum oneshot; do
  for seed in 0 1 2; do
    out=../tulokset/${mode}_s${seed}
    [ -f $out.json ] || ../.venv/bin/python lisays.py --holdout $mode --seed $seed --steps 10000 --out $out
  done
done
