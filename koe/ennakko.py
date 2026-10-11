"""Schaeffer-testi ajon sisällä: laskeeko pois jätettyjen häviö (jatkuva mittari)
ennen kuin tarkkuus (epäjatkuva) hyppää? Tulostaa askeleet per yleistyvä ajo."""
import csv, glob
T = "../tulokset/"
print("| ajo | häviön huippu (askel) | häviö puoleen | tarkkuus +10 pp | tarkkuus ≥ 90 % | ennakko (askelta) |\n|---|---|---|---|---|---|")
for f in sorted(glob.glob(T + "v2_random_f*_s?.csv")):
    r = list(csv.DictReader(open(f))); st = [int(x["step"]) for x in r]
    acc = [float(x["heldout"]) for x in r]; L = [float(x["heldout_loss"]) for x in r]
    g = next((s for s, a in zip(st, acc) if a >= .9), None)
    if not g: continue
    pk = max((i for i in range(len(r)) if st[i] < g), key=lambda i: L[i])
    half = next(s for s, l in zip(st, L) if s > st[pk] and l <= L[pk] / 2)
    a10 = next(s for s, a in zip(st, acc) if s > st[pk] and a >= acc[pk] + .1)
    print(f"| {f.split('/')[-1][:-4]} | {L[pk]:.1f} ({st[pk]}) | {half} | {a10} | {g} | {a10 - half} |")
