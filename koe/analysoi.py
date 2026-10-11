"""Koostaa tulokset: taulukko + faasikuva. Lukee ../tulokset/v2_random_*.csv jne."""
import csv, glob, json, re, statistics as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T = "../tulokset/"
def grok_step(rows, thr=0.9):
    for r in rows:
        if float(r["heldout"]) >= thr: return int(r["step"])
runs = {}
for f in sorted(glob.glob(T + "v2_random_f*_s*.json") + glob.glob(T + "v3_random_*_s*.json") + glob.glob(T + "v7_pitka_*_s*.json")):
    d = json.load(open(f)); rows = list(csv.DictReader(open(f[:-5] + ".csv")))
    fr = d["frac"]; runs.setdefault((fr, d["steps"]), []).append((d["final"]["heldout"], d["final"]["heldout_loss"], grok_step(rows), rows))
out = ["| koulutusosuus | askelia | siemeniä | pois jätetyt, tarkkuus (min–max) | häviö (keskiarvo) | yleistyksen askel (≥90 %) |", "|---|---|---|---|---|---|"]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for (fr, steps), rs in sorted(runs.items()):
    accs = [r[0] for r in rs]; gs = [r[2] for r in rs]
    out.append(f"| {fr:.3f} | {steps} | {len(rs)} | {min(accs):.3f}–{max(accs):.3f} | {st.mean(r[1] for r in rs):.2f} | {', '.join(str(g) if g else '–' for g in gs)} |")
    if steps != 100000:
        ax[0].scatter([fr] * len(accs), accs, c="C0"); 
        for g in gs:
            if g: ax[1].scatter([fr], [g], c="C1")
    for r in rs:
        if steps != 100000 and fr in (0.1, 0.12, 0.15, 0.2):
            ax[2].plot([int(x["step"]) for x in r[3]], [float(x["heldout"]) for x in r[3]], c={0.1:"C3",0.12:"C2",0.15:"C1",0.2:"C0"}[fr], alpha=.6, label=f"{fr}" if r is rs[0] else None)
ax[0].set(xlabel="koulutusosuus", ylabel="pois jätettyjen parien tarkkuus (lopussa)", title="Faasiraja"); ax[1].set(xlabel="koulutusosuus", ylabel="askel, jolla tarkkuus ≥ 90 %", title="Yleistyksen ajankohta")
ax[2].set(xlabel="askel", ylabel="pois jätettyjen tarkkuus", title="Oppimiskäyrät"); ax[2].legend(title="osuus")
for a_ in ax: a_.grid(alpha=.3)
plt.tight_layout(); plt.savefig(T + "faasikuva.png", dpi=130)
open(T + "taulukko_satunnainen.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
