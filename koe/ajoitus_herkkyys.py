"""Arvio 3 / kohta 5: ajoituksen herkkyys kynnykselle (v6_track_*). Nousu = ensimmäinen askel,
jolla 5 pisteen liukuva mediaani ylittää arvonsa askeleella 4000 kynnyksen d verran (kuten ajoitus.py).
Johto = pois jätettyjen tarkkuuden nousu − aikaisempi rakennemittarin nousu (positiivinen: rakenne ensin).
Lisäksi pois jätettyjen häviön puolittuminen (vs. askel 4000) ja kiinteä kynnys 0,5.
Kirjoittaa ../tulokset/taulukko_ajoitus_herkkyys.md."""
import csv, glob, statistics
T = "../tulokset/"
med5 = lambda v: [statistics.median(v[max(0, i - 2): i + 3]) for i in range(len(v))]
def rise(st, v, d, base=4000):
    v = med5(v); b = v[st.index(base)]; return next((s for s, x in zip(st, v) if s > base and x >= b + d), None)
first = lambda st, v, thr: next((s for s, x in zip(st, v) if x >= thr), None)
out = ["| ajo | johto d=0,05 | johto d=0,1 | johto d=0,2 | häviö puolittuu | tarkkuus +0,1 | kiinteä 0,5: Fourier − tarkkuus |", "|---|---|---|---|---|---|---|"]
for f in sorted(glob.glob(T + "v6_track_f0.1[5]_s?.csv") + glob.glob(T + "v6_track_f0.2_s?.csv")):
    r = list(csv.DictReader(open(f))); st = [int(x["step"]) for x in r]
    v = {k: [float(x[k]) for x in r] for k in ("heldout", "summainvarianssi", "fourier_top5", "heldout_loss")}
    j = []
    for d in (0.05, 0.1, 0.2):
        s_ = [x for x in (rise(st, v["summainvarianssi"], d), rise(st, v["fourier_top5"], d)) if x]
        h = rise(st, v["heldout"], d); j.append(f"{h - min(s_)}" if h and s_ else "–")
    l = med5(v["heldout_loss"]); b = l[st.index(4000)]
    half = next((s for s, x in zip(st, l) if s > 4000 and x <= b / 2), None)
    h1 = rise(st, v["heldout"], 0.1)
    fx, hx = first(st, v["fourier_top5"], .5), first(st, v["heldout"], .5)
    out.append(f"| {f.split('/')[-1][:-4]} | {j[0]} | {j[1]} | {j[2]} | {half or '–'} | {h1 or '–'} | {fx - hx if fx and hx else '–'} |")
open(T + "taulukko_ajoitus_herkkyys.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
