#!/bin/zsh
# Testi 11: tallennetut mallit esitysanalyysiin (yleistyvä 20 % vs. muistava 10 %).
cd "${0:A:h}"
for frac in 0.2 0.1; do for seed in 0 1; do
  out=../tulokset/v5_malli_f${frac}_s${seed}
  [ -f $out.pt ] || ../.venv/bin/python lisays.py --frac $frac --seed $seed --steps 40000 --warmup 500 --clip 1.0 --save --out $out
done; done
