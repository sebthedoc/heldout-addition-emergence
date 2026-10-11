#!/bin/zsh
# Odottaa kierros 4:n (testi 10) ja ajaa skaala-analyysin; irrallaan Claude-istunnosta.
cd "${0:A:h}"
while pgrep -f kierros4.sh >/dev/null; do sleep 60; done
../.venv/bin/python skaala.py && cd .. && git add tulokset/taulukko_skaala.md tulokset/skaalakuva.png tulokset/v4_* && git commit -qm "Kierros 4 valmis: mallin koko × koulutusosuus" && echo VALMIS
