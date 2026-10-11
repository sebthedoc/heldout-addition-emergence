"""Testi 10: mallin koko (d) × koulutusosuus. Lukee ../tulokset/v4_d*_f*_s*.json ja
v2_random_f{0.15,0.2}_s*.json (d=128). Taulukko: yleistyneet siemenet / kaikki, ja kuva."""
import glob, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T = "../tulokset/"
runs = {}
for f in glob.glob(T + "v4_d*_f*_s*.json") + glob.glob(T + "v2_random_f0.15_s?.json") + glob.glob(T + "v2_random_f0.2_s?.json"):
    d = json.load(open(f)); runs.setdefault((d["d"], d["frac"]), []).append(d["final"]["heldout"])
ds = sorted({k[0] for k in runs}); fs = sorted({k[1] for k in runs})
out = ["| d \\ osuus | " + " | ".join(f"{x:.0%}" for x in fs) + " |", "|---|" + "---|" * len(fs)]
for d in ds:
    cells = []
    for fr in fs:
        r = runs.get((d, fr))
        cells.append("–" if not r else f"{sum(a >= .9 for a in r)}/{len(r)} ({min(r):.2f}–{max(r):.2f})")
    out.append(f"| {d} | " + " | ".join(cells) + " |")
fig, ax = plt.subplots(figsize=(6, 4))
for fr, c in zip(fs, ("C3", "C1", "C0")):
    for d in ds:
        r = runs.get((d, fr), [])
        ax.scatter([d] * len(r), r, c=c, alpha=.7, label=f"{fr:.0%}" if d == ds[0] else None)
ax.set(xscale="log", xlabel="mallin leveys d", ylabel="pois jätettyjen tarkkuus (lopussa)", title="Mallin koko × koulutusosuus"); ax.set_xticks(ds); ax.set_xticklabels(ds)
ax.grid(alpha=.3); ax.legend(title="osuus"); plt.tight_layout(); plt.savefig(T + "skaalakuva.png", dpi=130)
open(T + "taulukko_skaala.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
