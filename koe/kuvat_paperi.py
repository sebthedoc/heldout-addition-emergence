"""Käsikirjoituksen kuvat englanniksi samoista CSV/JSON-tiedostoista kuin suomenkieliset
analyysikuvat (analysoi.py, ajoitus.py, skaala.py). Kirjoittaa ../tulokset/paperi_*.pdf.
Värit: viitepaletin kolme ensimmäistä luokkaväriä; koulutusosuus yksisävyisenä rampina."""
import csv, glob, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T = "../tulokset/"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.2,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = {0.1: "#b9d3f2", 0.12: "#7fb0ea", 0.15: "#2a78d6", 0.2: "#0d3f7d"}   # vaalea -> tumma
STYLE = {0.1: ":", 0.12: "-.", 0.15: "--", 0.2: "-"}
grid = lambda a: a.grid(alpha=.25, lw=.5)
rd = lambda f: list(csv.DictReader(open(f)))

# Kuva 1: faasiraja (lopputarkkuus osuuden funktiona) + oppimiskäyrät
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.1), gridspec_kw={"width_ratios": [1, 1.4]})
for f in sorted(glob.glob(T + "v2_random_f*_s?.json")):
    d = json.load(open(f)); acc = d["final"]["heldout"]
    ax[0].scatter(d["frac"], acc, s=22, c=BLUE if acc >= .9 else "white", edgecolors=BLUE, lw=1, zorder=3)
ax[0].plot([.07, .205], [.07, .205], c="#888888", ls="--", lw=.8, zorder=2, label="mirror share ≈ f")   # taulukko_vaihdannaisuus.md
for f in sorted(glob.glob(T + "v3_random_f0.1_pitka_s?.json") + glob.glob(T + "v7_pitka_f*_s?.json")):
    d = json.load(open(f)); acc = d["final"]["heldout"]
    ax[0].scatter(d["frac"] + .003, acc, s=22, marker="^", c=ORANGE if acc >= .9 else "white", edgecolors=ORANGE, lw=1, zorder=3)
ax[0].scatter([], [], s=22, c=BLUE, label="40k, ≥90%"); ax[0].scatter([], [], s=22, c="white", edgecolors=BLUE, label="40k, <90%")
ax[0].scatter([], [], s=22, marker="^", c="white", edgecolors=ORANGE, label="100k")
ax[0].set(xlabel="training fraction of all pairs", ylabel="held-out accuracy (final)", ylim=(-.02, 1.05)); ax[0].legend(loc="upper left", bbox_to_anchor=(0.0, .9), frameon=False); grid(ax[0])
for fr in (0.1, 0.12, 0.15, 0.2):
    for i, f in enumerate(sorted(glob.glob(T + f"v2_random_f{fr}_s?.csv"))):
        r = rd(f); ax[1].plot([int(x["step"]) / 1000 for x in r], [float(x["heldout"]) for x in r], c=RAMP[fr], ls=STYLE[fr], lw=1, label=f"{fr:.0%}" if i == 0 else None)
ax[1].set(xlabel="training step (thousands)", ylabel="held-out accuracy", ylim=(-.02, 1.05)); ax[1].legend(title="fraction", frameon=False, loc="center left", bbox_to_anchor=(1.0, .5)); grid(ax[1])
plt.tight_layout(); plt.savefig(T + "paperi_faasi.pdf")

# Kuva 2: rakennemittarit ja pois jätettyjen tarkkuus ajassa (v6_track)
fig, ax = plt.subplots(1, 3, figsize=(6.8, 1.9), sharey=True)
for j, fr in enumerate((0.1, 0.15, 0.2)):
    for i, f in enumerate(sorted(glob.glob(T + f"v6_track_f{fr}_s?.csv"))):
        r = rd(f); st = [int(x["step"]) / 1000 for x in r]
        for k, c, ls, lab in (("heldout", BLUE, "-", "held-out accuracy"), ("summainvarianssi", ORANGE, "--", "sum invariance"), ("fourier_top5", AQUA, ":", "Fourier top-5 share")):
            ax[j].plot(st, [float(x[k]) for x in r], c=c, ls=ls, lw=.9, alpha=.85, label=lab if i == 0 else None)
    ax[j].set(title=f"training fraction {fr:.0%}", xlabel="step (thousands)", ylim=(0, 1.02)); grid(ax[j])
ax[0].legend(frameon=False, loc="upper right")
plt.tight_layout(); plt.savefig(T + "paperi_ajoitus.pdf")

# Kuva 3: mallin leveys × koulutusosuus
runs = {}
for f in glob.glob(T + "v4_d*_f*_s*.json") + glob.glob(T + "v2_random_f0.15_s?.json") + glob.glob(T + "v2_random_f0.2_s?.json"):
    d = json.load(open(f)); runs.setdefault((d["d"], d["frac"]), []).append(d["final"]["heldout"])
fig, ax = plt.subplots(figsize=(3.3, 2.2))
for fr, c, m, dx in ((0.15, BLUE, "o", .92), (0.2, ORANGE, "s", 1.0), (0.3, AQUA, "^", 1.08)):
    for dd in sorted({k[0] for k in runs}):
        r = runs.get((dd, fr), []); ax.scatter([dd * dx] * len(r), r, c=c, marker=m, s=18, alpha=.85, label=f"{fr:.0%}" if dd == 32 else None)
ax.set(xscale="log", xlabel="model width d", ylabel="held-out accuracy (final)", ylim=(-.02, 1.05))
ax.set_xticks([32, 64, 128, 256, 512]); ax.set_xticklabels([32, 64, 128, 256, 512]); ax.minorticks_off()
ax.legend(title="fraction", frameon=False, loc="lower right"); grid(ax)
plt.tight_layout(); plt.savefig(T + "paperi_skaala.pdf")
print("ok")
