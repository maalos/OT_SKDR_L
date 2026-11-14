"""
RDS1 - kompaktowy format osi drog: polilinia (x, y) + szerokosc nawierzchni.
Jeden plik = jedno wojewodztwo. Zaleznosci: numpy (shapely tylko dla to_linestrings).

Wspolrzedne sa zapisywane w TEJ SAMEJ kolejnosci osi, w jakiej sa w pliku zrodlowym
(kolumny 0 i 1 wspolrzednych z GeoPandas), bez zamiany x/y.

Plik:
  "RDS1" | varint flags (bit0 = predyktor 2. rzedu) | xy_step_cm | w_step_cm |
  voiv | origin_x | origin_y (w krokach siatki) | n_seg |
  6 x varint (dlugosc skompresowanego strumienia) | 6 strumieni (raw LZMA2)

Strumienie (kolejnosc segmentow = krzywa Morton'a po punkcie startowym):
  0 nv      : varint (liczba wierzcholkow - 2)
  1 width   : uint8 na segment, w krokach w_step_cm, 0 = brak
  2 sdx     : svarint, delta punktu startowego wzgledem poprzedniego segmentu
  3 sdy
  4 vx      : svarint, delty kolejnych wierzcholkow (bez startu); z flaga bit0 -> reszty
  5 vy        po predykcji 2. rzedu (d[i] - d[i-1])
"""
import lzma
import numpy as np

MAGIC = b"RDS1"
_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME,
             "dict_size": 1 << 26, "lc": 3, "lp": 0, "pb": 0}]


# ---------- varint ----------
def _put(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _get(buf, pos):
    n = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        n |= (b & 0x7F) << shift
        if not b & 0x80:
            return n, pos
        shift += 7


def uvarint_encode(a):
    a = np.asarray(a, dtype=np.uint64)
    if a.size == 0:
        return b""
    nb = np.ones(a.shape, np.int64)
    t = a >> np.uint64(7)
    while t.any():
        nb += (t > 0)
        t >>= np.uint64(7)
    offs = np.cumsum(nb) - nb
    out = np.empty(int(nb.sum()), np.uint8)
    for k in range(int(nb.max())):
        sel = nb > k
        byte = (a[sel] >> np.uint64(7 * k)) & np.uint64(0x7F)
        cont = (nb[sel] > k + 1).astype(np.uint64) << np.uint64(7)
        out[offs[sel] + k] = (byte | cont).astype(np.uint8)
    return out.tobytes()


def uvarint_decode(buf):
    b = np.frombuffer(buf, np.uint8)
    if b.size == 0:
        return np.zeros(0, np.uint64)
    ends = np.flatnonzero((b & 0x80) == 0)
    starts = np.concatenate(([0], ends[:-1] + 1))
    lens = ends - starts + 1
    val = np.zeros(len(ends), np.uint64)
    for k in range(int(lens.max())):
        sel = lens > k
        val[sel] |= (b[starts[sel] + k] & 0x7F).astype(np.uint64) << np.uint64(7 * k)
    return val


def svarint_encode(a):
    a = np.asarray(a, dtype=np.int64)
    z = ((a << 1) ^ (a >> 63)).astype(np.uint64)
    return uvarint_encode(z)


def svarint_decode(buf):
    z = uvarint_decode(buf)
    return (z >> np.uint64(1)).astype(np.int64) ^ -((z & np.uint64(1)).astype(np.int64))


def _segcumsum(a, lens):
    c = np.cumsum(a)
    ends = np.cumsum(lens)
    base = np.zeros(len(lens), a.dtype)
    base[1:] = c[ends[:-1] - 1]
    return c - np.repeat(base, lens)


def _part1by1(x):
    x = x.astype(np.uint64) & np.uint64(0xFFFFFFFF)
    for sh, m in ((16, 0x0000FFFF0000FFFF), (8, 0x00FF00FF00FF00FF),
                  (4, 0x0F0F0F0F0F0F0F0F), (2, 0x3333333333333333),
                  (1, 0x5555555555555555)):
        x = (x | (x << np.uint64(sh))) & np.uint64(m)
    return x


def _comp(b):
    return lzma.compress(b, format=lzma.FORMAT_RAW, filters=_FILTERS)


def _decomp(b):
    return lzma.decompress(b, format=lzma.FORMAT_RAW, filters=_FILTERS)


# ---------- enkoder ----------
def encode(q, counts, wq, *, voiv, step_cm, wstep_cm, pred=False):
    """q: (N,2) int, wspolrzedne w krokach siatki (absolutne), counts: wierzcholki
    na segment (>=2), wq: uint8 szerokosc na segment (0 = brak). Zwraca bytes."""
    q = np.asarray(q, np.int64)
    counts = np.asarray(counts, np.int64)
    wq = np.asarray(wq, np.uint8)
    nseg = len(counts)

    # usun powtorzone kolejne wierzcholki po kwantyzacji (zostaw min. 2 wierzcholki)
    seg = np.repeat(np.arange(nseg), counts)
    keep = np.ones(len(q), bool)
    keep[1:] = ~(((q[1:] == q[:-1]).all(1)) & (seg[1:] == seg[:-1]))
    c2 = np.bincount(seg[keep], minlength=nseg).astype(np.int64)
    q = q[keep]
    single = c2 == 1
    if single.any():
        first = np.cumsum(c2) - c2
        rep = np.ones(len(q), np.int64)
        rep[first[single]] = 2
        q = np.repeat(q, rep, axis=0)
        c2[single] = 2
    counts = c2

    ox, oy = int(q[:, 0].min()), int(q[:, 1].min())
    qx, qy = q[:, 0] - ox, q[:, 1] - oy

    # kolejnosc segmentow: krzywa Morton'a po punkcie startowym
    offs = np.cumsum(counts) - counts
    key = _part1by1(qx[offs]) | (_part1by1(qy[offs]) << np.uint64(1))
    order = np.argsort(key, kind="stable")
    nc = counts[order]
    newstart = np.cumsum(nc) - nc
    gather = np.repeat(offs[order] - newstart, nc) + np.arange(int(nc.sum()))
    qx, qy, counts, wq = qx[gather], qy[gather], nc, wq[order]

    ns = np.cumsum(counts) - counts
    sdx = np.diff(qx[ns], prepend=0)
    sdy = np.diff(qy[ns], prepend=0)

    mask = np.ones(len(qx), bool)
    mask[ns] = False
    dx = np.diff(qx, prepend=qx[0])[mask]
    dy = np.diff(qy, prepend=qy[0])[mask]
    if pred:
        lens = counts - 1
        first = np.cumsum(lens) - lens
        for d in (dx, dy):
            r = d.copy()
            r[1:] -= d[:-1]
            r[first] = d[first]
            d[:] = r

    streams = [uvarint_encode(counts - 2), wq.tobytes(),
               svarint_encode(sdx), svarint_encode(sdy),
               svarint_encode(dx), svarint_encode(dy)]
    comp = [_comp(s) for s in streams]
    head = (MAGIC + _put(1 if pred else 0) + _put(step_cm) + _put(wstep_cm) +
            _put(voiv) + _put(ox) + _put(oy) + _put(nseg) +
            b"".join(_put(len(c)) for c in comp))
    return head + b"".join(comp)


# ---------- dekoder ----------
def read(path_or_bytes):
    """Zwraca dict: coords (N,2) float64 [m], offsets (nseg+1), width_m (nseg, nan = brak),
    voiv, step_cm."""
    buf = path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray)) \
        else open(path_or_bytes, "rb").read()
    assert buf[:4] == MAGIC, "to nie jest plik RDS1"
    pos = 4
    flags, pos = _get(buf, pos)
    step_cm, pos = _get(buf, pos)
    wstep_cm, pos = _get(buf, pos)
    voiv, pos = _get(buf, pos)
    ox, pos = _get(buf, pos)
    oy, pos = _get(buf, pos)
    nseg, pos = _get(buf, pos)
    lens = []
    for _ in range(6):
        n, pos = _get(buf, pos)
        lens.append(n)
    parts = []
    for n in lens:
        parts.append(_decomp(buf[pos:pos + n]))
        pos += n
    nv = uvarint_decode(parts[0]).astype(np.int64) + 2
    wq = np.frombuffer(parts[1], np.uint8)
    sx = np.cumsum(svarint_decode(parts[2]))
    sy = np.cumsum(svarint_decode(parts[3]))
    dx, dy = svarint_decode(parts[4]), svarint_decode(parts[5])
    lens_d = nv - 1
    N = int(nv.sum())
    ns = np.cumsum(nv) - nv
    mask = np.ones(N, bool)
    mask[ns] = False
    out = []
    for d, s in ((dx, sx), (dy, sy)):
        if flags & 1:
            d = _segcumsum(d, lens_d)
        x = np.empty(N, np.int64)
        x[ns] = s
        x[mask] = _segcumsum(d, lens_d) + np.repeat(s, lens_d)
        out.append(x)
    k = step_cm / 100.0
    coords = np.stack([(out[0] + ox) * k, (out[1] + oy) * k], axis=1)
    width = np.where(wq == 0, np.nan, wq * (wstep_cm / 100.0))
    offsets = np.concatenate(([0], np.cumsum(nv)))
    return {"coords": coords, "offsets": offsets, "width_m": width,
            "voiv": voiv, "step_cm": step_cm}


def to_linestrings(res):
    import shapely
    nv = np.diff(res["offsets"])
    return shapely.linestrings(res["coords"], indices=np.repeat(np.arange(len(nv)), nv))


if __name__ == "__main__":
    import sys
    r = read(sys.argv[1])
    w = r["width_m"]
    print(f"{len(w)} odcinkow, {len(r['coords'])} wierzcholkow, "
          f"szerokosc znana: {np.isfinite(w).mean():.1%}, mediana: {np.nanmedian(w):.1f} m")