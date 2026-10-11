"""Arvio 3 / kohta 3: täydennysraja samoille maskeille kuin koulutusajoissa.
Taulu T[a, b] = a + b on rank-2-matriisi (Lee ym. 2023, §5.1). Kaksi tunnistettavuuskriteeriä:
 (i) additiivinen malli T = u_a + v_b: pois jätetty (a, b) on määrätty, jos rivisolmu a ja
     sarakesolmu b ovat samassa havaintoverkon komponentissa (kaksijakoinen verkko, nähty pari = kaari);
 (ii) yleinen rank-2-täydennys (Király ym. 2015 -tyyppinen paikallinen täydennettävyys): alkio on
     määrätty, jos sen Jacobin rivi kuvauksesta (U, V) -> (U V^T)_ij on havaittujen rivien virittämässä
     aliavaruudessa satunnaisessa yleisessä pisteessä.
Kumpikin järjestetylle havaintojoukolle ja symmetrisoidulle (lisätään peiliparit, ts. malli, joka on
oppinut vaihdannaisuuden). Kirjoittaa ../tulokset/taulukko_teoriaraja.md."""
import numpy as np, torch
N = 100; T = "../tulokset/"

def maski(frac, seed):   # sama kuin lisays.py
    torch.manual_seed(seed); perm = torch.randperm(N * N)
    tr = torch.zeros(N * N, dtype=torch.bool); tr[perm[:int(frac * N * N)]] = True
    return tr.view(N, N).numpy()

def komponentit(obs):    # rivit 0..N-1, sarakkeet N..2N-1
    p = list(range(2 * N))
    def f(x):
        while p[x] != x: p[x] = p[p[x]]; x = p[x]
        return x
    for i, j in zip(*np.nonzero(obs)): p[f(i)] = f(N + j)
    return np.array([f(x) for x in range(2 * N)])

def additiivinen(obs, ho):
    c = komponentit(obs); I, J = np.nonzero(ho)
    return (c[I] == c[N + J]).mean(), len(set(c)) == 1

rng = np.random.default_rng(0); U, V = rng.standard_normal((N, 2)), rng.standard_normal((N, 2))
def jrivit(I, J):         # d(U V^T)_ij / d(U, V), U:n ja V:n parametrit peräkkäin
    R = np.zeros((len(I), 4 * N))
    for k in range(2): R[np.arange(len(I)), 2 * I + k] = V[J, k]; R[np.arange(len(I)), 2 * N + 2 * J + k] = U[I, k]
    return R

def rank2(obs, ho):
    Io, Jo = np.nonzero(obs); _, s, Vt = np.linalg.svd(jrivit(Io, Jo), full_matrices=False)
    r = int((s > s[0] * 1e-9).sum()); Q = Vt[:r]
    I, J = np.nonzero(ho); R = jrivit(I, J)
    res = np.linalg.norm(R - (R @ Q.T) @ Q, axis=1) / np.linalg.norm(R, axis=1)
    return (res < 1e-6).mean(), r

rivit = ["| osuus | siemen | additiivinen: järj. | additiivinen: symm. | rank-2: järj. | rank-2: symm. | Jacobin aste järj./symm. (täysi 396) |",
         "|---|---|---|---|---|---|---|"]
for frac in (0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.12, 0.13, 0.14, 0.15, 0.2):
    for seed in (0, 1, 2):
        tr = maski(frac, seed); ho = ~tr; sy = tr | tr.T
        a1, _ = additiivinen(tr, ho); a2, _ = additiivinen(sy, ho)
        r1, k1 = rank2(tr, ho); r2, k2 = rank2(sy, ho)
        rivit.append(f"| {frac} | {seed} | {a1:.3f} | {a2:.3f} | {r1:.3f} | {r2:.3f} | {k1}/{k2} |")
        print(rivit[-1], flush=True)
open(T + "taulukko_teoriaraja.md", "w").write(
    "Osuus pois jätetyistä pareista, jotka koulutusparit määräävät yksikäsitteisesti (`koe/teoriaraja.py`).\n"
    "järj. = vain koulutusparit; symm. = koulutusparit ja niiden peilit (vaihdannaisuus opittu).\n"
    "Osuudet 0.075–0.2 ovat samat maskit kuin koulutusajoissa; 0.02–0.05 vain teorialle.\n\n" + "\n".join(rivit) + "\n")
