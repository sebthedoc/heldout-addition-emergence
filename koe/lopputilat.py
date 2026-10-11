"""Satunnaisen pois jätön lopputilat (v2 40 k, v3/v7 100 k): koulutustarkkuus ja
pois jätettyjen häviö rajan ala- ja yläpuolella. Paperin Results- ja Discussion-luvut.
Ajo: cd koe && python lopputilat.py > ../tulokset/taulukko_lopputilat.md"""
import glob, json
T = "../tulokset/"
fs = sorted(glob.glob(T + "v2_random_f*_s*.json") + glob.glob(T + "v3_random_*_s*.json") + glob.glob(T + "v7_pitka_*_s*.json"))
rs = []
for f in fs:
    d = json.load(open(f)); x = d["final"]
    rs.append((f.split("/")[-1][:-5], d["frac"], d["steps"], x["train"], x["heldout"], x["heldout_loss"]))
print("| ajo | osuus | askelia | koulutus | pois jätetyt | häviö |\n|---|---|---|---|---|---|")
for r in rs: print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.3f} | {r[4]:.3f} | {r[5]:.2f} |")
mem = [r for r in rs if r[1] <= 0.12]; gen = [r for r in rs if r[4] >= 0.9]
rng = lambda v: f"{min(v):.2f}–{max(v):.2f}"
print(f"\nKoulutustarkkuus < 1: {[(r[0], round(r[3], 3)) for r in rs if r[3] < 1]}")
print(f"Häviö, osuus ≤ 12 %, kaikki: {rng([r[5] for r in mem])}; 100 k: {rng([r[5] for r in mem if r[2] == 100000])}")
print(f"Häviö, yleistyneet (≥ 0,9): {rng([r[5] for r in gen])} ({len(gen)} ajoa)")
