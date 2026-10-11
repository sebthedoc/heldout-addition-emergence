"""Rakenteen synty ajassa (v6_track_*): milloin summainvarianssi ja Fourier-keskittyneisyys
nousevat suhteessa pois jätettyjen tarkkuuteen. Taulukko + kuva."""
import csv, glob
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T = "../tulokset/"
first = lambda st, v, thr: next((s for s, x in zip(st, v) if x >= thr), None)
import statistics
med5 = lambda v: [statistics.median(v[max(0, i - 2): i + 3]) for i in range(len(v))]  # 5 pisteen liukuva mediaani (1250 askelta)
def rise(st, v, base_step=4000, d=0.1):
    v = med5(v)  # ensimmäinen askel, jolla mittari on noussut d yli tasanteensa (arvo askeleella base_step)
    b = v[st.index(base_step)]; return next((s for s, x in zip(st, v) if s > base_step and x >= b + d), None)
fs = sorted(glob.glob(T + "v6_track_f*_s?.csv")); fr = sorted({f.split("_f")[1].split("_s")[0] for f in fs})
fig, ax = plt.subplots(1, len(fr), figsize=(5 * len(fr), 4), squeeze=False)
out = ["| ajo | tasanne @4000: summainv. / Fourier / pois jätetyt | summainv. +0,1 | Fourier +0,1 | pois jätetyt +0,1 | pois jätetyt ≥ 90 % |", "|---|---|---|---|---|---|"]
for f in fs:
    r = list(csv.DictReader(open(f))); st = [int(x["step"]) for x in r]
    v = {k: [float(x[k]) for x in r] for k in ("heldout", "summainvarianssi", "fourier_top5")}
    i4 = st.index(4000); pl = " / ".join(f"{v[k][i4]:.2f}" for k in ("summainvarianssi", "fourier_top5", "heldout"))
    out.append(f"| {f.split('/')[-1][:-4]} | {pl} | {rise(st, v['summainvarianssi']) or '–'} | {rise(st, v['fourier_top5']) or '–'} | {rise(st, v['heldout']) or '–'} | {first(st, v['heldout'], .9) or '–'} |")
    a_ = ax[0][fr.index(f.split("_f")[1].split("_s")[0])]
    for k, c in (("heldout", "C0"), ("summainvarianssi", "C1"), ("fourier_top5", "C2")):
        a_.plot(st, v[k], c=c, alpha=.7, label=k if f.endswith("_s0.csv") else None)
for i, x in enumerate(fr):
    ax[0][i].set(title=f"koulutusosuus {x}", xlabel="askel", ylim=(0, 1.02)); ax[0][i].grid(alpha=.3)
ax[0][0].legend(["pois jätetyt oikein", "summainvarianssi", "Fourier top-5"], loc="lower right")
plt.tight_layout(); plt.savefig(T + "ajoitus.png", dpi=130)
open(T + "taulukko_ajoitus.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
