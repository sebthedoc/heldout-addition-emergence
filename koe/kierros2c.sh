#!/bin/zsh
cd "${0:A:h}"
for seed in 0 1; do
  out=../tulokset/v2_kompositio_s${seed}
  [ -f $out.json ] || ../.venv/bin/python kompositio.py --seed $seed --steps 60000 --warmup 500 --clip 1.0 --wd 1.0 --out $out
done
