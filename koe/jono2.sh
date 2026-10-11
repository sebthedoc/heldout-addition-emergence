#!/bin/zsh
# Numeroittainen carry-koe ja kompositio-koe; odottaa edelliset jonot.
cd "${0:A:h}"
for seed in 0 1 2; do
  [ -f ../tulokset/numerot_s$seed.json ] || ../.venv/bin/python numerot.py --seed $seed --steps 8000 --out ../tulokset/numerot_s$seed
  [ -f ../tulokset/kompositio_s$seed.json ] || ../.venv/bin/python kompositio.py --seed $seed --steps 30000 --out ../tulokset/kompositio_s$seed
done
