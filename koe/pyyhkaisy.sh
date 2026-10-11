#!/bin/zsh
# Vaihekaavio: koulutusosuus x siemen, satunnainen pois jättö. Ajetaan peräkkäin.
cd "${0:A:h}"
for frac in 0.5 0.3 0.2 0.1 0.05; do
  for seed in 0 1 2; do
    out=../tulokset/random_f${frac}_s${seed}
    [ -f $out.json ] || ../.venv/bin/python lisays.py --frac $frac --seed $seed --steps 10000 --out $out
  done
done
