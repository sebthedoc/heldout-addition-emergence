#!/bin/zsh
# Rakenteen synty ajassa: summainvarianssi ja Fourier-keskittyneisyys koulutuksen aikana.
cd "${0:A:h}"
run() { for seed in 0 1 2; do
  out=../tulokset/v6_track_f$1_s${seed}
  [ -f $out.json ] || ../.venv/bin/python lisays.py --frac $1 --seed $seed --steps 40000 --warmup 500 --clip 1.0 --track --out $out
done; }
(run 0.2) & (run 0.15) & (run 0.1) & wait
