#!/bin/zsh
# Kierros 3: tarkenna faasiraja 10–15 % ja testaa onko 10 % vain hidas (100 k askelta).
cd "${0:A:h}"
for frac in 0.12 0.13 0.14; do for seed in 0 1 2; do
  out=../tulokset/v2_random_f${frac}_s${seed}
  [ -f $out.json ] || ../.venv/bin/python lisays.py --frac $frac --seed $seed --steps 40000 --warmup 500 --clip 1.0 --out $out
done; done &
for seed in 0 1; do
  out=../tulokset/v3_random_f0.1_pitka_s${seed}
  [ -f $out.json ] || ../.venv/bin/python lisays.py --frac 0.1 --seed $seed --steps 100000 --warmup 500 --clip 1.0 --out $out
done
wait
