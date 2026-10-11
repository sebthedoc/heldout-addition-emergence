"""Onko malliin syntynyt yhteenlaskun rakenne? Kaksi testiä tallennetuille malleille.
1. Summainvarianssi: kuinka suuri osa viimeisen paikan sisäisen tilan vaihtelusta
   selittyy pelkällä summalla a+b (ryhmien välinen varianssi / kokonaisvarianssi).
   Muistitaulukossa (7,23) ja (20,10) ovat eri tiloja; algoritmissa sama tila.
2. Kausaalinen poisto: poistetaan upotuksista mallin 5 vahvinta Fourier-taajuutta
   (vs. 5 satunnaista) ja mitataan pois jätettyjen parien tarkkuus."""
import glob, json, torch, torch.nn as nn
T = "../tulokset/"
class M(nn.Module):
    def __init__(s, N, d=128, L=2, h=4):
        super().__init__()
        s.emb = nn.Embedding(N + 1, d); s.pos = nn.Parameter(torch.randn(3, d) * .02)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, h, 4 * d, 0.0, batch_first=True, norm_first=True), L, enable_nested_tensor=False)
        s.out = nn.Linear(d, 2 * N - 1)
    def hidden(s, x): return s.enc(s.emb(x) + s.pos)[:, -1]
    def forward(s, x): return s.out(s.hidden(x))
def ablate(E, ks):  # poista taajuudet ks lukujen 0..N-1 upotuksista
    N = E.shape[0]; F = torch.fft.rfft(E, dim=0); F[ks] = 0; return torch.fft.irfft(F, n=N, dim=0)
torch.manual_seed(0)
print("| malli | pois jätetyt oikein | summainvarianssi | – 5 vahvinta taajuutta | – 5 satunnaista |\n|---|---|---|---|---|")
for f in sorted(glob.glob(T + "v5_malli_*.pt")):
    d = torch.load(f, weights_only=False, map_location="cpu"); a = d["args"]; N = a["n"]; tr = d["train_mask"]
    m = M(N, a["d"], a["layers"]); m.load_state_dict(d["model"]); m.eval()
    A, B = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij"); A, B = A.flatten(), B.flatten()
    X = torch.stack([A, B, torch.full_like(A, N)], 1); y = A + B; te = ~tr
    with torch.no_grad():
        H = m.hidden(X); tot = ((H - H.mean(0)) ** 2).sum()
        means = torch.zeros(2 * N - 1, H.shape[1]).index_add_(0, y, H) / torch.bincount(y, minlength=2 * N - 1)[:, None]
        within = ((H - means[y]) ** 2).sum(); inv = 1 - (within / tot).item()
        acc = lambda: (m(X[te]).argmax(-1) == y[te]).float().mean().item()
        base = acc(); E0 = m.emb.weight.data.clone(); Ec = E0[:N] - E0[:N].mean(0)
        P = torch.fft.rfft(Ec, dim=0).abs().pow(2).sum(1); P[0] = 0; top = P.topk(5).indices
        m.emb.weight.data[:N] = ablate(E0[:N], top); a_top = acc()
        rnd = [k for k in torch.randperm(len(P)).tolist() if k != 0 and k not in top.tolist()][:5]
        m.emb.weight.data[:N] = ablate(E0[:N], torch.tensor(rnd)); a_rnd = acc(); m.emb.weight.data = E0
    print(f"| {f.split('/')[-1][:-3]} | {base:.3f} | {inv:.3f} | {a_top:.3f} | {a_rnd:.3f} |")
