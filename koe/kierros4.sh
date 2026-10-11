#!/bin/zsh
# Testi 10: mallin koko (d) x koulutusosuus. Weiin skaala-akseli.
cd "${0:A:h}"
run() { for frac in 0.15 0.2 0.3; do for seed in 0 1; do
  out=../tulokset/v4_d$1_f${frac}_s${seed}
  [ -f $out.json ] || ../.venv/bin/python lisays.py --d $1 --frac $frac --seed $seed --steps 40000 --warmup 500 --clip 1.0 --out $out
done; done; }
(run 32; run 256) & (run 64; run 512) & wait
