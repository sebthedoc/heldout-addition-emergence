#!/bin/zsh
# Odottaa kierros 6:n ja 4:n loppuun ja ajaa analyysit; ei riipu Claude-istunnosta.
cd "${0:A:h}"
while pgrep -f 'kierros6.sh|kierros4.sh' >/dev/null; do sleep 60; done
../.venv/bin/python ajoitus.py && ../.venv/bin/python analysoi.py > /dev/null && echo "VALMIS $(date)"
