"""Testi 8: numeroittainen yhteenlasku, 3-numeroiset luvut.
Koulutuksesta puuttuu täsmälleen yksi asia: esimerkit, joissa muistinumero
etenee (carry kantautuu seuraavaan numeroon, c_in=1 ja c_out=1).
Testi: yleistyykö malli niihin. Kontrolli: pois jätetty otos samasta jakaumasta."""
import argparse, csv, json, time
import torch, torch.nn as nn
p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0); p.add_argument("--steps", type=int, default=8000)
p.add_argument("--wd", type=float, default=0.1); p.add_argument("--out", default="../tulokset/numerot")
a = p.parse_args()
torch.manual_seed(a.seed); dev = "mps" if torch.backends.mps.is_available() else "cpu"
D = 3
def make(n):
    x = torch.randint(0, 10 ** D, (n, 2)); return x[:, 0], x[:, 1]
def digits(v, k): return torch.stack([(v // 10 ** i) % 10 for i in range(k - 1, -1, -1)], 1)  # MSB ensin
def propagated(A, B):  # onko jokin kohta, jossa c_in=1 ja c_out=1
    c = torch.zeros_like(A); prop = torch.zeros_like(A, dtype=torch.bool)
    for i in range(D):
        da, db = (A // 10 ** i) % 10, (B // 10 ** i) % 10
        cout = ((da + db + c) >= 10).long()
        prop |= (c == 1) & (cout == 1); c = cout
    return prop
A, B = make(400000); prop = propagated(A, B)
def enc(A, B):
    n = len(A); plus = torch.full((n, 1), 10); eq = torch.full((n, 4), 11)  # 4 kyselypaikkaa
    return torch.cat([digits(A, D), plus, digits(B, D), eq], 1), digits(A + B, D + 1)
X, Y = enc(A, B)
tr = ~prop; idx = tr.nonzero().squeeze(); g = torch.Generator().manual_seed(1)
idx = idx[torch.randperm(len(idx), generator=g)]; trn, ctl = idx[: len(idx) - 5000], idx[len(idx) - 5000:]
te = prop.nonzero().squeeze()[:5000]
Xtr, Ytr = X[trn].to(dev), Y[trn].to(dev)
sets = {"ctl_ei_etene": (X[ctl].to(dev), Y[ctl].to(dev)), "pois_etenee": (X[te].to(dev), Y[te].to(dev))}
class M(nn.Module):
    def __init__(s, d=128, L=3, h=4):
        super().__init__(); s.emb = nn.Embedding(12, d); s.pos = nn.Parameter(torch.randn(Xtr.shape[1], d) * .02)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, h, 4 * d, 0.0, batch_first=True, norm_first=True), L)
        s.out = nn.Linear(d, 10)
    def forward(s, x): return s.out(s.enc(s.emb(x) + s.pos)[:, -4:])
m = M().to(dev); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=a.wd)
def ev(X_, Y_):
    with torch.no_grad():
        o = m(X_); return (o.argmax(-1) == Y_).all(1).float().mean().item(), nn.functional.cross_entropy(o.reshape(-1, 10), Y_.reshape(-1)).item()
rows = []; t0 = time.time()
for s in range(a.steps + 1):
    m.train(); b = torch.randint(0, len(Xtr), (512,), device=dev)
    loss = nn.functional.cross_entropy(m(Xtr[b]).reshape(-1, 10), Ytr[b].reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    if s % 250 == 0:
        m.eval(); r = dict(step=s, loss=loss.item())
        for k, (x_, y_) in sets.items(): r[k], r[k + "_loss"] = ev(x_, y_)
        rows.append(r)
        if s % 1000 == 0: print(r, f"{time.time()-t0:.0f}s", flush=True)
with open(a.out + ".csv", "w", newline="") as f:
    w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
json.dump(vars(a) | dict(n_train=len(trn), final=rows[-1]), open(a.out + ".json", "w"), indent=1)
