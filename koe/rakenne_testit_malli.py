"""Malliluokka ja taajuuspoisto (rakenne_testit.py), tuotavaksi muihin analyyseihin."""
import torch, torch.nn as nn
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
