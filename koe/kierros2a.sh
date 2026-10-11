#!/bin/zsh
# Kierros 2: vakautus (warmup+leikkaus) ja pidempi koulutus. Satunnaiset osuudet.
cd "${0:A:h}"
for frac in 0.2 0.15 0.1 0.075; do for seed in 0 1 2; do
  out=../tulokset/v2_random_f${frac}_s${seed}
  [ -f $out.json ] || ../.venv/bin/python lisays.py --frac $frac --seed $seed --steps 40000 --warmup 500 --clip 1.0 --out $out
done; done
