#!/bin/zsh
# Odottaa kierros 6:n loppuun, ajaa ajoitusanalyysin ja kirjoittaa VALMIS lokiin.
cd "${0:A:h}"
while pgrep -f 'kierros6.sh' >/dev/null; do sleep 30; done
../.venv/bin/python ajoitus.py && echo "VALMIS $(date)"
