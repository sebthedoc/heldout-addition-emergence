"""Arvio 3 / kohta 4 (R1b): energiavakioitu kontrolli kausaaliselle taajuuspoistolle.
Viidellä vahvimmalla taajuudella on 68–78 % upotuksen tehosta, joten "5 satunnaista" poistaa paljon
vähemmän. Kontrollit tallennetuille malleille (v5_malli_*.pt):
  top 1–5   : 5 vahvinta taajuutta pois (kuten rakenne_testit.py)
  top 6–10  : tehojärjestyksessä seuraavat 5 pois
  kohina    : upotuksiin lisätään keskitetty gaussinen häiriö, jonka normi = top 1–5 -komponentin normi
              (5 arvontaa, keskiarvo)
  kutistus  : upotukset kutistetaan tasaisesti niin, että jäljelle jää sama normi kuin top 1–5 -poiston jälkeen
Sarake 'osuus' = poistetun komponentin osuus keskitetyn upotuksen tehosta. Kirjoittaa
../tulokset/taulukko_poistokontrolli.md."""
import glob, torch
from rakenne_testit_malli import M, ablate
T = "../tulokset/"
out = ["| malli | pois jätetyt | top 1–5 (osuus) | top 6–10 (osuus) | kohina, sama normi | kutistus, sama normi |", "|---|---|---|---|---|---|"]
for f in sorted(glob.glob(T + "v5_malli_*.pt")):
    d = torch.load(f, weights_only=False, map_location="cpu"); a = d["args"]; N = a["n"]; te = ~d["train_mask"]
    m = M(N, a["d"], a["layers"]); m.load_state_dict(d["model"]); m.eval()
    A, B = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij"); A, B = A.flatten(), B.flatten()
    X = torch.stack([A, B, torch.full_like(A, N)], 1)[te]; y = (A + B)[te]
    E0 = m.emb.weight.data.clone(); E = E0[:N]; mu = E.mean(0); Ec = E - mu
    P = torch.fft.rfft(Ec, dim=0).abs().pow(2).sum(1); P[0] = 0; order = P.argsort(descending=True)
    def acc(En):
        m.emb.weight.data[:N] = En
        with torch.no_grad(): r = (m(X).argmax(-1) == y).float().mean().item()
        m.emb.weight.data = E0.clone(); return r
    base = acc(E)
    E1 = ablate(E, order[:5]); E2 = ablate(E, order[5:10])
    sh = lambda En: 1 - ((En - En.mean(0)) ** 2).sum().item() / (Ec ** 2).sum().item()
    nrm = (E - E1).norm()
    g = torch.Generator().manual_seed(0); kn = []
    for _ in range(5):
        Z = torch.randn(E.shape, generator=g); Z = Z - Z.mean(0); kn.append(acc(E + Z / Z.norm() * nrm))
    s = (E1 - mu).norm() / Ec.norm()
    out.append(f"| {f.split('/')[-1][:-3]} | {base:.3f} | {acc(E1):.3f} ({sh(E1):.2f}) | {acc(E2):.3f} ({sh(E2):.2f}) | {sum(kn)/5:.3f} | {acc(mu + Ec * s):.3f} |")
open(T + "taulukko_poistokontrolli.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
