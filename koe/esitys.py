"""Testi 11 (mekanismi): millainen lukujen esitys syntyy yleistävään vs. muistavaan malliin?
Lukee ../tulokset/v5_malli_*.pt. Mittarit operandien upotuksista (luvut 0..N-1):
  - lineaarinen koetin: ridge-regressio 10 pääkomponentista -> luvun arvo, leave-one-out R²
  - Fourier-keskittyneisyys: osuus tehosta 5 vahvimmalla taajuudella (Nanda ym. 2023)
  - kuva: kaksi ensimmäistä pääkomponenttia luvun mukaan väritettynä"""
import glob, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T = "../tulokset/"
fs = sorted(glob.glob(T + "v5_malli_*.pt"))
fig, ax = plt.subplots(1, len(fs), figsize=(4 * len(fs), 4), squeeze=False)
print("| malli | pois jätetyt oikein | koetin R² | Fourier top-5 osuus |\n|---|---|---|---|")
for i, f in enumerate(fs):
    d = torch.load(f, weights_only=False, map_location="cpu"); a = d["args"]; N = a["n"]
    E = d["model"]["emb.weight"][:N].float().cpu(); E = E - E.mean(0)
    y = torch.arange(N).float()
    _, _, V10 = torch.linalg.svd(E, full_matrices=False); Ek = E @ V10[:10].T   # 10 pääkomponenttia: p < n
    X = torch.cat([Ek, torch.ones(N, 1)], 1); lam = 1e-2
    H = X @ torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1]), X.T)   # hat-matriisi
    res = (y - H @ y) / (1 - H.diag())                                         # LOO-residuaalit
    r2 = 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()
    P = torch.fft.rfft(E, dim=0).abs().pow(2).sum(1)[1:]; top5 = (P.topk(5).values.sum() / P.sum()).item()
    acc = __import__("json").load(open(f[:-3] + ".json"))["final"]["heldout"]
    print(f"| {f.split('/')[-1][:-3]} | {acc:.3f} | {r2:.3f} | {top5:.3f} |")
    U, S, V = torch.pca_lowrank(E, q=2, center=False); Z = E @ V
    sc = ax[0][i].scatter(Z[:, 0], Z[:, 1], c=y, cmap="viridis", s=14)
    ax[0][i].set_title(f"osuus {a['frac']}, siemen {a['seed']}\npois jätetyt {acc:.0%}, R² {r2:.2f}", fontsize=9)
for s_ in range(3):  # vertailu: satunnainen upotus samalla mittarilla
    g = torch.Generator().manual_seed(s_); E = torch.randn(N, 128, generator=g) * .02; E = E - E.mean(0)
    X = torch.cat([E, torch.ones(N, 1)], 1); H = X @ torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1]), X.T)
    res = (y - H @ y) / (1 - H.diag()); r2 = 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()
    P = torch.fft.rfft(E, dim=0).abs().pow(2).sum(1)[1:]
    print(f"| satunnainen upotus {s_} | – | {r2:.3f} | {(P.topk(5).values.sum() / P.sum()).item():.3f} |")
fig.colorbar(sc, ax=ax[0].tolist(), label="luku"); plt.savefig(T + "esitys_pca.png", dpi=130, bbox_inches="tight")
