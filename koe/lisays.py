"""Yhteenlasku (a+b), osa pareista pois koulutuksesta.

Syöte: [a, b, '='] -> luokka a+b (0..2N-2). Pieni transformer-enkooderi.
Pois jätetyt parit mitataan erikseen (held-out) koko koulutuksen ajan.
"""
import argparse, csv, json, time
import torch, torch.nn as nn

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=100)        # operandit 0..n-1
p.add_argument("--frac", type=float, default=0.7)   # koulutusosuus pareista
p.add_argument("--holdout", default="random")       # random | diag (a==b) | row (a%10==7 & b%10==3)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--d", type=int, default=128)
p.add_argument("--layers", type=int, default=2)
p.add_argument("--warmup", type=int, default=0)
p.add_argument("--clip", type=float, default=0.0)
p.add_argument("--wd", type=float, default=1.0)
p.add_argument("--track", action="store_true")  # rakennemittarit koulutuksen aikana
p.add_argument("--save", action="store_true")
p.add_argument("--out", default="../tulokset/ajo")
a = p.parse_args()

torch.manual_seed(a.seed)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
N = a.n
A, B = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij")
A, B = A.flatten(), B.flatten()
if a.holdout == "random":
    perm = torch.randperm(N * N)
    k = int(a.frac * N * N)
    tr = torch.zeros(N * N, dtype=torch.bool); tr[perm[:k]] = True
elif a.holdout == "diag":
    tr = A != B
elif a.holdout == "row":
    tr = ~((A % 10 == 7) & (B % 10 == 3))
elif a.holdout == "commut":      # vain a<=b koulutuksessa; testi: a>b (vaihdannaisuus)
    tr = A <= B
elif a.holdout == "block":       # yhtenäinen 2D-lohko 40..49 x 40..49 pois
    tr = ~((A >= 40) & (A < 50) & (B >= 40) & (B < 50))
elif a.holdout == "newsum":      # kaikki parit, joiden summa = 99, pois: luokkaa ei nähty koskaan
    tr = (A + B) != 99
elif a.holdout == "oneshot":     # operandit 11,37,58,83 nähdään vain parina 0:n kanssa
    S = torch.tensor([11, 37, 58, 83])
    isS = lambda v: (v[:, None] == S).any(1)
    tr = ~(isS(A) | isS(B)) | ((A == 0) | (B == 0))
y = A + B
X = torch.stack([A, B, torch.full_like(A, N)], 1)  # N = '=' -tokeni
Xtr, ytr, Xte, yte = X[tr].to(dev), y[tr].to(dev), X[~tr].to(dev), y[~tr].to(dev)

class M(nn.Module):
    def __init__(s, d=128, L=2, h=4):
        super().__init__()
        s.emb = nn.Embedding(N + 1, d); s.pos = nn.Parameter(torch.randn(3, d) * .02)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, h, 4 * d, 0.0, batch_first=True, norm_first=True), L)
        s.out = nn.Linear(d, 2 * N - 1)
    def hidden(s, x):
        return s.enc(s.emb(x) + s.pos)[:, -1]
    def forward(s, x):
        return s.out(s.hidden(x))

m = M(a.d, a.layers).to(dev)
opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=a.wd, betas=(.9, .98))
acc = lambda X_, y_: (m(X_).argmax(-1) == y_).float().mean().item() if len(y_) else float("nan")
rows = []; t0 = time.time()
for s in range(a.steps + 1):
    m.train()
    loss = nn.functional.cross_entropy(m(Xtr), ytr)  # täysi erä
    opt.zero_grad(); loss.backward()
    if a.clip: nn.utils.clip_grad_norm_(m.parameters(), a.clip)
    for g_ in opt.param_groups: g_['lr'] = 1e-3 * (min(1, (s + 1) / a.warmup) if a.warmup else 1)
    opt.step()
    if s % 250 == 0:
        m.eval()
        with torch.no_grad():
            hl = nn.functional.cross_entropy(m(Xte), yte).item() if len(yte) else float("nan")
            row = dict(step=s, loss=loss.item(), train=acc(Xtr, ytr), heldout=acc(Xte, yte), heldout_loss=hl)
            if a.track:
                H = m.hidden(X.to(dev)).cpu(); yy = y
                mu = torch.zeros(2 * N - 1, H.shape[1]).index_add_(0, yy, H) / torch.bincount(yy, minlength=2 * N - 1)[:, None]
                row["summainvarianssi"] = 1 - (((H - mu[yy]) ** 2).sum() / ((H - H.mean(0)) ** 2).sum()).item()
                E = m.emb.weight[:N].detach().cpu(); E = E - E.mean(0)
                P = torch.fft.rfft(E, dim=0).abs().pow(2).sum(1)[1:]
                row["fourier_top5"] = (P.topk(5).values.sum() / P.sum()).item()
            rows.append(row)
        if s % 2500 == 0: print(rows[-1], f"{time.time()-t0:.0f}s", flush=True)
with open(a.out + ".csv", "w", newline="") as f:
    w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
json.dump(vars(a) | dict(n_train=int(tr.sum()), n_heldout=int((~tr).sum()), final=rows[-1]), open(a.out + ".json", "w"), indent=1)
if a.save:
    torch.save(dict(model=m.state_dict(), args=vars(a), train_mask=tr), a.out + ".pt")
