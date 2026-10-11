"""Rakenteelliset pois jätöt, negatiivinen kontrolli ja numero-/kompositiokokeet yhteen taulukkoon.
Lukee ../tulokset/{v2_diag,v2_row,v2_commut,v2_block,v2_oneshot,newsum,numerot,v2_kompositio}_s*.json.
Kirjoittaa ../tulokset/taulukko_rakenteet.md."""
import glob, json
T = "../tulokset/"
# (tiedostoetuliite, kuvaus, koulutustarkkuuden avain, pois jätetyn tarkkuuden avain, kontrollin avain)
kokeet = [
    ("v2_diag", "diagonal a=b", "train", "heldout", None),
    ("v2_row", "a≡7, b≡3 (mod 10)", "train", "heldout", None),
    ("v2_block", "block 40–49 × 40–49", "train", "heldout", None),
    ("v2_commut", "a>b (commutativity)", "train", "heldout", None),
    ("v2_oneshot", "4 operands seen only with 0", "train", "heldout", None),
    ("newsum", "sum 99 (class never seen)", "train", "heldout", None),
    ("numerot", "digits: carry propagation", None, "pois_etenee", "ctl_ei_etene"),
    ("v2_kompositio", "mod 97: a−b−c", None, "pois_a-b-c", "ctl_nahty_kuvio"),
]
f2 = lambda v: f"{v:.2f}"
out = ["| held out | steps | held-out n | seeds | train acc. (min–max) | in-distribution control | held-out acc. per seed | held-out loss (mean) |",
       "|---|---|---|---|---|---|---|---|"]
for pre, nimi, ktr, kte, kctl in kokeet:
    ds = [json.load(open(f)) for f in sorted(glob.glob(T + pre + "_s?.json"))]
    fin = [d["final"] for d in ds]
    tr = "–" if ktr is None else f"{f2(min(x[ktr] for x in fin))}–{f2(max(x[ktr] for x in fin))}"
    ctl = "–" if kctl is None else f"{f2(min(x[kctl] for x in fin))}–{f2(max(x[kctl] for x in fin))}"
    nte = ds[0].get("n_heldout") or "5000"
    acc = ", ".join(f2(x[kte]) for x in fin)
    loss = sum(x[kte + "_loss" if kte != "heldout" else "heldout_loss"] for x in fin) / len(fin)
    out.append(f"| {nimi} | {ds[0]['steps']} | {nte} | {len(ds)} | {tr} | {ctl} | {acc} | {loss:.2f} |")
open(T + "taulukko_rakenteet.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
