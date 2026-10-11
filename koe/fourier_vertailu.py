"""Arvio 1 / R1a: Fourier top-5 -osuuden vertailukohdat ja vahvimmat taajuudet jaksoina.
Ei-jaksollisella akselilla 0..N-1 pelkkä lukusuora (upotus ∝ a) keskittää DFT-tehon
matalille taajuuksille, joten sen top-5-osuus on laskettava rinnalle.
Kirjoittaa ../tulokset/taulukko_fourier.md."""
import glob, json, torch
T = "../tulokset/"
def spec(E):
    E = E - E.mean(0); P = torch.fft.rfft(E, dim=0).abs().pow(2).sum(1)[1:]   # k = 1..N/2
    return P, (P.topk(5).values.sum() / P.sum()).item()
out = ["| embedding | held-out acc. | Fourier top-5 share | strongest k (period N/k) |", "|---|---|---|---|"]
for f in sorted(glob.glob(T + "v5_malli_*.pt")):
    d = torch.load(f, weights_only=False, map_location="cpu"); N = d["args"]["n"]
    P, s = spec(d["model"]["emb.weight"][:N].float()); ks = (P.topk(5).indices + 1).tolist()
    acc = json.load(open(f[:-3] + ".json"))["final"]["heldout"]
    out.append(f"| {f.split('/')[-1][:-3]} | {acc:.3f} | {s:.3f} | " + ", ".join(f"{k} ({N / k:.1f})" for k in ks) + " |")
N = 100; a = torch.arange(N).float()[:, None]
P, s = spec(a); out.append(f"| linear number line (E ∝ a) | – | {s:.3f} | " + ", ".join(f"{k} ({N / k:.1f})" for k in (P.topk(5).indices + 1).tolist()) + " |")
for sd in range(3):
    g = torch.Generator().manual_seed(sd); E = torch.randn(N, 128, generator=g)
    out.append(f"| random Gaussian, seed {sd} | – | {spec(E)[1]:.3f} | – |")
open(T + "taulukko_fourier.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
