"""Testi 9: kompositio. Mod-97, lauseke a op1 b [op2 c].
Koulutuksessa kuviot: a+b, a-b, a+b+c, a+b-c, a-b+c. Pois: a-b-c (yhdistelmä,
jonka osat kaikki on nähty). Kontrolli: nähdyn kuvion pois jätetyt kolmikot."""
import argparse, csv, json, time
import torch, torch.nn as nn
p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0); p.add_argument("--steps", type=int, default=10000)
p.add_argument("--warmup", type=int, default=0)
p.add_argument("--clip", type=float, default=0.0)
p.add_argument("--wd", type=float, default=1.0); p.add_argument("--out", default="../tulokset/kompositio")
a = p.parse_args()
torch.manual_seed(a.seed); dev = "mps" if torch.backends.mps.is_available() else "cpu"
P = 97; PLUS, MINUS, NONE = P, P + 1, P + 2   # tokenit 0..96 luvut
def make(pattern, n):
    o1, o2 = pattern; A = torch.randint(0, P, (n,)); B = torch.randint(0, P, (n,)); C = torch.randint(0, P, (n,))
    sgn = lambda o: 1 if o == "+" else -1
    if o2 is None:
        y = (A + sgn(o1) * B) % P; C = torch.full((n,), NONE); t2 = torch.full((n,), NONE)
    else:
        y = (A + sgn(o1) * B + sgn(o2) * C) % P; t2 = torch.full((n,), PLUS if o2 == "+" else MINUS)
    t1 = torch.full((n,), PLUS if o1 == "+" else MINUS)
    return torch.stack([A, t1, B, t2, C], 1), y
seen = [("+", None), ("-", None), ("+", "+"), ("+", "-"), ("-", "+")]; held = ("-", "-")
parts = [make(q, 60000 if q[1] else 8000) for q in seen]
X = torch.cat([x for x, _ in parts]); Y = torch.cat([y for _, y in parts])
perm = torch.randperm(len(X)); X, Y = X[perm].to(dev), Y[perm].to(dev)
nt = len(X) - 5000; Xtr, Ytr = X[:nt], Y[:nt]
sets = {"ctl_nahty_kuvio": (X[nt:], Y[nt:])}
xh, yh = make(held, 5000); sets["pois_a-b-c"] = (xh.to(dev), yh.to(dev))
class M(nn.Module):
    def __init__(s, d=128, L=3, h=4):
        super().__init__(); s.emb = nn.Embedding(P + 3, d); s.pos = nn.Parameter(torch.randn(5, d) * .02)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, h, 4 * d, 0.0, batch_first=True, norm_first=True), L)
        s.out = nn.Linear(d, P)
    def forward(s, x): return s.out(s.enc(s.emb(x) + s.pos).mean(1))
m = M().to(dev); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=a.wd, betas=(.9, .98))
def ev(X_, Y_):
    with torch.no_grad():
        o = m(X_); return (o.argmax(-1) == Y_).float().mean().item(), nn.functional.cross_entropy(o, Y_).item()
rows = []; t0 = time.time()
for s in range(a.steps + 1):
    m.train(); b = torch.randint(0, nt, (512,), device=dev)
    loss = nn.functional.cross_entropy(m(Xtr[b]), Ytr[b]); opt.zero_grad(); loss.backward()
    if a.clip: nn.utils.clip_grad_norm_(m.parameters(), a.clip)
    for g_ in opt.param_groups: g_['lr'] = 1e-3 * (min(1, (s + 1) / a.warmup) if a.warmup else 1)
    opt.step()
    if s % 250 == 0:
        m.eval(); r = dict(step=s, loss=loss.item())
        for k, (x_, y_) in sets.items(): r[k], r[k + "_loss"] = ev(x_, y_)
        rows.append(r)
        if s % 1000 == 0: print(r, f"{time.time()-t0:.0f}s", flush=True)
with open(a.out + ".csv", "w", newline="") as f:
    w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
json.dump(vars(a) | dict(n_train=nt, final=rows[-1]), open(a.out + ".json", "w"), indent=1)
