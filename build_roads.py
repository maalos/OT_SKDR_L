"""
OT_SKDR_L.parquet (BDOT10k) -> 16 plikow .rds (po jednym na wojewodztwo).

Uzycie:
  py -3.12 build_roads.py OT_SKDR_L.parquet out
  py -3.12 build_roads.py OT_SKDR_L.parquet out --step-cm 100 --tol-cm 100   # mniejsze
  py -3.12 build_roads.py OT_SKDR_L.parquet out --only 16                    # szybki test
Opcje: --pred (predyktor 2. rzedu), --impute (uzupelnij brakujace szerokosci; domyslnie 0 = nieznana),
       --step-cm (siatka), --tol-cm (uproszczenie Douglasa-Peuckera), --wstep-cm.
Wymaga: numpy pandas pyarrow shapely (geopandas tylko gdy geometria nie jest WKB).
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
import rds

COLS = ["TERYT", "KLASADROGI", "MATERIALNAWIERZCHNI", "SZEROKOSCNAWIERZCHNI"]
MIN_N = 20  # min. liczba znanych szerokosci, zeby ufac medianie grupy


def valid_width(w, maxw):
    return np.isfinite(w) & (w > 0) & (w <= maxw)


def make_fill(df, maxw):
    """Mediana szerokosci wg (klasa, nawierzchnia) -> nawierzchnia -> klasa -> globalna."""
    d = pd.DataFrame({"c": df["KLASADROGI"].fillna(""), "m": df["MATERIALNAWIERZCHNI"].fillna(""),
                      "w": df["SZEROKOSCNAWIERZCHNI"].astype(float)})
    d.loc[~valid_width(d["w"].to_numpy(), maxw), "w"] = np.nan
    glob = float(d["w"].median())
    gm = d.groupby("m")["w"].agg(["median", "count"])
    gc = d.groupby("c")["w"].agg(["median", "count"])
    g = d.groupby(["c", "m"])["w"].agg(["median", "count"]).reset_index()
    def pick(r):
        if r["count"] >= MIN_N:
            return r["median"]
        if r["m"] in gm.index and gm.loc[r["m"], "count"] >= MIN_N:
            return gm.loc[r["m"], "median"]
        if r["c"] in gc.index and gc.loc[r["c"], "count"] >= MIN_N:
            return gc.loc[r["c"], "median"]
        return glob
    g["fill"] = g.apply(pick, axis=1)
    return g.set_index(["c", "m"])["fill"], glob


def iter_chunks(path, chunk=250_000):
    pf = pq.ParquetFile(path)
    geo = json.loads(pf.schema_arrow.metadata[b"geo"])
    gcol = geo["primary_column"]
    if geo["columns"][gcol].get("encoding", "WKB") == "WKB":
        for b in pf.iter_batches(batch_size=chunk, columns=COLS + [gcol]):
            yield ({c: b.column(c) for c in COLS},
                   shapely.from_wkb(b.column(gcol).to_numpy(zero_copy_only=False)))
    else:  # np. geoarrow - czytamy calosc przez geopandas (duzo RAM)
        import geopandas as gpd
        gdf = gpd.read_parquet(path, columns=COLS + [gcol])
        for i in range(0, len(gdf), chunk):
            part = gdf.iloc[i:i + chunk]
            yield ({c: pa.array(part[c]) for c in COLS}, part.geometry.to_numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--step-cm", type=int, default=50)
    ap.add_argument("--tol-cm", type=int, default=None, help="domyslnie = step-cm, 0 = bez uproszczenia")
    ap.add_argument("--wstep-cm", type=int, default=10)
    ap.add_argument("--pred", action="store_true")
    ap.add_argument("--impute", action="store_true",
                    help="uzupelnij brakujace szerokosci mediana (klasa, nawierzchnia); domyslnie brak = nieznana")
    ap.add_argument("--only", type=int, default=None, help="tylko wojewodztwo, np. 16")
    a = ap.parse_args()
    tol = a.step_cm if a.tol_cm is None else a.tol_cm
    maxw = 255 * a.wstep_cm / 100.0
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    small = pq.read_table(a.src, columns=COLS).to_pandas()
    fill, glob = make_fill(small, maxw)
    print(f"szerokosc: znana {valid_width(small['SZEROKOSCNAWIERZCHNI'].to_numpy(float), maxw).mean():.1%}, "
          f"mediana globalna {glob:.1f} m")
    del small

    acc = {}
    dropped = 0
    for cols, geoms in iter_chunks(a.src):
        terr = cols["TERYT"].to_pandas().astype(str).str.zfill(4).str[:2].astype(int).to_numpy()
        cl = cols["KLASADROGI"].to_pandas().fillna("").to_numpy()
        mt = cols["MATERIALNAWIERZCHNI"].to_pandas().fillna("").to_numpy()
        w = cols["SZEROKOSCNAWIERZCHNI"].to_numpy(zero_copy_only=False).astype(float)
        w = np.where(valid_width(w, maxw), w, np.nan)
        if a.impute:
            fv = fill.reindex(pd.MultiIndex.from_arrays([cl, mt])).to_numpy(float)
            fv = np.where(np.isnan(fv), glob, fv)
            w = np.where(np.isnan(w), fv, w)
        wq = np.where(np.isnan(w), 0, np.clip(np.rint(w * 100 / a.wstep_cm), 1, 255)).astype(np.uint8)

        if tol > 0:
            geoms = shapely.simplify(geoms, tol / 100.0, preserve_topology=False)
        n = shapely.get_num_coordinates(geoms).astype(np.int64)
        xy = shapely.get_coordinates(geoms)
        row = np.repeat(np.arange(len(n)), n)
        q = np.rint(xy * (100.0 / a.step_cm)).astype(np.int32)
        ok = n >= 2
        dropped += int((~ok).sum())
        for v in np.unique(terr):
            sel = (terr == v) & ok
            if a.only is not None and v != a.only:
                continue
            m = sel[row]
            d = acc.setdefault(int(v), {"q": [], "n": [], "w": []})
            d["q"].append(q[m]); d["n"].append(n[sel]); d["w"].append(wq[sel])
    if dropped:
        print(f"pominieto {dropped} pustych/jednopunktowych geometrii")

    total = 0
    for v in sorted(acc):
        d = acc[v]
        blob = rds.encode(np.concatenate(d["q"]), np.concatenate(d["n"]), np.concatenate(d["w"]),
                          voiv=v, step_cm=a.step_cm, wstep_cm=a.wstep_cm, pred=a.pred)
        p = os.path.join(a.out, f"roads_{v:02d}.rds")
        with open(p, "wb") as f:
            f.write(blob)
        total += len(blob)
        nseg = sum(len(x) for x in d["n"])
        print(f"{p}: {nseg:>8} odcinkow  {len(blob)/1e6:7.2f} MB  ({len(blob)/nseg:.1f} B/odcinek)")
    print(f"RAZEM {total/1e6:.1f} MB, czas {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()