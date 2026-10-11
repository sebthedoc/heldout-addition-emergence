"""Arvio 3 / kohta 1: onko muistava malli vaihdannainen hakutaulu?
(i) Peiliosuus: kuinka suuri osa pois jätetyistä pareista (a, b) on sellaisia, joiden peilipari
    (b, a) on koulutuksessa. Maski toistetaan kuten lisays.py (manual_seed -> randperm).
(ii) Tallennetut 10 %:n mallit (v5_malli_f0.1_s*.pt): tarkkuus pois jätetyillä erikseen
    peilillisille ja peilittömille pareille.
Kirjoittaa ../tulokset/taulukko_vaihdannaisuus.md."""
import glob, json, torch, torch.nn as nn
T = "../tulokset/"; N = 100
A, B = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij"); A, B = A.flatten(), B.flatten()

def maski(frac, seed):
    torch.manual_seed(seed); perm = torch.randperm(N * N)
    tr = torch.zeros(N * N, dtype=torch.bool); tr[perm[:int(frac * N * N)]] = True; return tr

def peili(tr): return tr.view(N, N).T.flatten()   # peili[i] = onko (b, a) koulutuksessa

class M(nn.Module):  # sama kuin lisays.py
    def __init__(s, d=128, L=2, h=4):
        super().__init__()
        s.emb = nn.Embedding(N + 1, d); s.pos = nn.Parameter(torch.randn(3, d) * .02)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, h, 4 * d, 0.0, batch_first=True, norm_first=True), L, enable_nested_tensor=False)
        s.out = nn.Linear(d, 2 * N - 1)
    def forward(s, x): return s.out(s.enc(s.emb(x) + s.pos)[:, -1])

out = ["## Peiliosuus vs. lopullinen pois jätettyjen tarkkuus (40 k askelta, `v2_random_*`)", "",
       "| osuus | siemen | peilillisiä pois jätetyistä | pois jätetyt oikein | yleistyi |", "|---|---|---|---|---|"]
for f in sorted(glob.glob(T + "v2_random_f*_s?.json"), key=lambda f: (json.load(open(f))["frac"], f)):
    d = json.load(open(f)); tr = maski(d["frac"], d["seed"]); ho = ~tr
    assert int(tr.sum()) == d["n_train"]
    ms = peili(tr)[ho].float().mean().item(); acc = d["final"]["heldout"]
    out.append(f"| {d['frac']} | {d['seed']} | {ms:.3f} | {acc:.3f} | {'kyllä' if acc >= .9 else 'ei'} |")
out += ["", "## Muistavat mallit pareittain (`v5_malli_f0.1_s*.pt`)", "",
        "| malli | pois jätetyt | peilillisiä | oikein peilillisistä | oikein peilittömistä | oikeista peilillisiä |", "|---|---|---|---|---|---|"]
X = torch.stack([A, B, torch.full_like(A, N)], 1); y = A + B
for f in sorted(glob.glob(T + "v5_malli_f0.1_s?.pt")):
    d = torch.load(f, weights_only=False, map_location="cpu"); a = d["args"]; tr = d["train_mask"].cpu()
    assert torch.equal(tr, maski(a["frac"], a["seed"]))
    m = M(a["d"], a["layers"]); m.load_state_dict(d["model"]); m.eval()
    with torch.no_grad(): ok = m(X).argmax(-1) == y
    ho = ~tr; pm = peili(tr) & ho; pn = ~peili(tr) & ho
    out.append(f"| {f.split('/')[-1][:-3]} | {ho.sum().item()} | {pm.sum().item()} | {ok[pm].float().mean():.3f} | {ok[pn].float().mean():.3f} | {(ok & pm).sum().item()}/{(ok & ho).sum().item()} |")
open(T + "taulukko_vaihdannaisuus.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
