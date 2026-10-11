#!/bin/zsh
# Arvio 1 / R2a: onko 13–14 %:n "stokastinen raja" budjettiartefakti? 40 k askeleella
# yleistymättömät raja-ajot jatketaan 100 k askeleeseen (sama siemen = sama maski).
cd "${0:A:h}"
run() { out=../tulokset/v7_pitka_f$1_s$2
  [ -f $out.json ] || ../.venv/bin/python lisays.py --frac $1 --seed $2 --steps 100000 --warmup 500 --clip 1.0 --out $out; }
(run 0.13 0; run 0.12 0) & (run 0.13 2; run 0.12 1) & (run 0.14 0; run 0.14 2) & wait
echo VALMIS $(date)
