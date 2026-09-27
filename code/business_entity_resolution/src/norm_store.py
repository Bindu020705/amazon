"""Build and load compact, memory-mappable normalized record stores.

For every record we precompute and cache on disk:

* ``name_b`` / ``name_o``  - normalized name, UTF-8 bytes + int64 offsets,
* ``addr_b`` / ``addr_o``  - normalized address, UTF-8 bytes + int64 offsets,
* ``atoms_flat`` / ``atoms_o`` - deduplicated core-token hashes of the record
  (name tokens first, then address-only tokens), CSR layout,
* ``atoms_flag`` - per atom: 1 = name only, 2 = address only, 3 = both,
* ``state`` - hash of the parsed state/region code, ``country`` - small int code,
* ``id_hash`` - hash of the entity id, ``ids.txt`` - the ids in row order.

The builder writes chunks straight to ``.npy`` part files as they are produced (streamed,
flat memory), then concatenates the parts on disk with bounded RAM. The ``Store`` reader
memory-maps everything, so opening a store is O(1) RAM.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import (  # noqa: E402
    addr_tokens, compact, core_tokens, h64, normalize_text, state_code, tokens,
)

COUNTRY_CODE = {"united states": 1, "us": 1, "usa": 1, "india": 2, "france": 3}
_STORE_ARRAYS = ("name_b", "name_o", "addr_b", "addr_o", "atoms_flat", "atoms_o",
                 "atoms_flag", "state", "country", "id_hash")


def country_code(c: str) -> int:
    return COUNTRY_CODE.get((c or "").strip().lower(), 0)


def encode_record(name: str, address: str, country: str) -> dict:
    """Normalize one record. Pure function (safe to run in a worker process)."""
    nn = normalize_text(name)
    an = normalize_text(address)
    ntok = tokens(nn)
    atok = tokens(an)
    ncore = core_tokens(ntok)
    acore = addr_tokens(atok)
    st = state_code(atok, country)

    order: list[str] = []
    flag: dict[str, int] = {}
    for t in ncore:
        if t not in flag:
            flag[t] = 1
            order.append(t)
        else:
            flag[t] |= 1
    for t in acore:
        if t not in flag:
            flag[t] = 2
            order.append(t)
        else:
            flag[t] |= 2

    return {
        "name": nn,
        "addr": an,
        "atoms": [h64("t", t) for t in order],
        "flags": [flag[t] for t in order],
        "cname": compact("".join(ncore)),
        "state": st,
        "country": country_code(country),
    }


def _worker_chunk(rows):
    out = {"name": [], "addr": [], "atoms": [], "flags": [], "state": [], "country": []}
    for nm, ad, co in rows:
        e = encode_record(nm or "", ad or "", co or "")
        out["name"].append(e["name"])
        out["addr"].append(e["addr"])
        out["atoms"].append(e["atoms"])
        out["flags"].append(e["flags"])
        out["state"].append(h64("st", e["state"]) if e["state"] else 0)
        out["country"].append(e["country"])
    return out


def _pack_strs(strings):
    enc = [s.encode("utf-8") for s in strings]
    off = np.zeros(len(enc) + 1, dtype=np.int64)
    np.cumsum(np.fromiter((len(b) for b in enc), dtype=np.int64, count=len(enc)), out=off[1:])
    buf = np.frombuffer(b"".join(enc), dtype=np.uint8) if enc else np.zeros(0, dtype=np.uint8)
    return np.ascontiguousarray(buf), off


def _pack_csr(lists, dtype=np.uint64):
    off = np.zeros(len(lists) + 1, dtype=np.int64)
    np.cumsum(np.fromiter((len(x) for x in lists), dtype=np.int64, count=len(lists)), out=off[1:])
    if off[-1]:
        flat = np.fromiter((v for x in lists for v in x), dtype=dtype, count=int(off[-1]))
    else:
        flat = np.zeros(0, dtype=dtype)
    return flat, off


def _absorb_chunk(res: dict) -> dict:
    """Convert one worker result into numpy arrays (frees the Python objects)."""
    name_b, name_o = _pack_strs(res["name"])
    addr_b, addr_o = _pack_strs(res["addr"])
    atoms_flat, atoms_o = _pack_csr(res["atoms"])
    flags = np.fromiter((v for x in res["flags"] for v in x), dtype=np.uint8,
                        count=int(atoms_o[-1]))
    return {
        "name_b": name_b, "name_o": name_o, "addr_b": addr_b, "addr_o": addr_o,
        "atoms_flat": atoms_flat, "atoms_o": atoms_o, "atoms_flag": flags,
        "state": np.asarray(res["state"], dtype=np.uint64),
        "country": np.asarray(res["country"], dtype=np.uint8),
    }


class _PartWriter:
    """Append numpy chunks to per-array part files; merge on close."""

    def __init__(self, tmp_dir: str, arrays):
        os.makedirs(tmp_dir, exist_ok=True)
        self.dir = tmp_dir
        self.fhs = {a: open(os.path.join(tmp_dir, f"{a}.parts"), "wb") for a in arrays}
        self.lens = {a: [] for a in arrays}   # entries per part, per array

    def write(self, arrays: dict) -> None:
        for a, fh in self.fhs.items():
            x = arrays[a]
            fh.write(np.ascontiguousarray(x).tobytes())
            self.lens[a].append(x.shape[0])

    def merge(self, out_dir: str) -> None:
        """Concatenate the part files into the final .npy arrays.

        Plain arrays are copied through unchanged. Offset arrays (``*_o``) are special:
        each appended part repeats its leading base value, so when merging we drop the
        first entry of every part after the first and shift it by the running byte
        base, producing a single ``n_records + 1`` offsets array. Parts are read with
        their exact recorded entry counts, so buffer boundaries never split a part.
        """
        import shutil
        for a, fh in self.fhs.items():
            fh.close()
        for a in self.fhs:
            src = os.path.join(self.dir, f"{a}.parts")
            dtype = _PART_DTYPE[a]
            itemsize = np.dtype(dtype).itemsize
            lens = self.lens[a]
            is_off = a.endswith("_o")
            total = sum(lens) if not is_off else sum(lens) - len(lens) + 1
            out = np.lib.format.open_memmap(
                os.path.join(out_dir, a + ".npy"), mode="w+",
                dtype=(np.int64 if is_off else dtype), shape=(total,))
            base, w = 0, 0
            with open(src, "rb") as f:
                for pi, ln in enumerate(lens):
                    chunk = np.frombuffer(f.read(ln * itemsize),
                                          dtype=(np.int64 if is_off else dtype))
                    if is_off:
                        take = chunk if pi == 0 else chunk[1:] + base
                        base += int(chunk[-1])
                    else:
                        take = chunk
                    out[w:w + take.size] = take
                    w += take.size
            del out
            os.remove(src)
        shutil.rmtree(self.dir, ignore_errors=True)


_PART_DTYPE = {
    "name_b": np.uint8, "name_o": np.int64, "addr_b": np.uint8, "addr_o": np.int64,
    "atoms_flat": np.uint64, "atoms_o": np.int64, "atoms_flag": np.uint8,
    "state": np.uint64, "country": np.uint8, "id_hash": np.uint64,
}


def build_store(chunk_iter, out_dir: str, workers: int = 8, verbose: bool = True) -> int:
    """Encode a stream of ``(ids, rows)`` chunks into ``out_dir``.

    Chunks are normalized in worker processes, converted to arrays and appended to part
    files immediately; peak RAM is a few chunks of flat arrays.
    """
    os.makedirs(out_dir, exist_ok=True)
    from collections import deque
    from concurrent.futures import ProcessPoolExecutor

    tmp = os.path.join(out_dir, "_parts_tmp")
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    w = _PartWriter(tmp, _STORE_ARRAYS)
    n_rec = 0
    window = max(workers, 2)
    with open(os.path.join(out_dir, "ids.txt"), "w", encoding="utf-8") as id_out, \
            ProcessPoolExecutor(max_workers=workers) as ex:
        pending: deque = deque()

        def absorb(fut, ids):
            nonlocal n_rec
            arrays = _absorb_chunk(fut.result())
            arrays["id_hash"] = np.fromiter((h64("id", i) for i in ids), dtype=np.uint64,
                                            count=len(ids))
            w.write(arrays)
            id_out.write("".join(i + "\n" for i in ids))
            n_rec += len(ids)

        for ids, rows in chunk_iter:
            pending.append((ex.submit(_worker_chunk, rows), ids))
            while len(pending) > window:
                absorb(*pending.popleft())
            if verbose and n_rec and n_rec % 1_000_000 < 25_000:
                print(f"    ... {n_rec:,} records", flush=True)
        while pending:
            absorb(*pending.popleft())

    w.merge(out_dir)
    if verbose:
        print(f"    store written: {n_rec:,} records", flush=True)
    return n_rec


def _concat_npy(paths: list[str], out_file: str, dtype) -> None:
    """Concatenate several .npy files into one, streaming through a small buffer."""
    size = 0
    infos = []
    for p in paths:
        a = np.load(p, mmap_mode="r")
        infos.append((p, a.dtype, a.shape))
        size += a.shape[0]
        del a
    arr = np.lib.format.open_memmap(out_file, mode="w+", dtype=dtype, shape=(size,))
    off = 0
    for p, _, _ in infos:
        a = np.load(p, mmap_mode="r")
        arr[off:off + a.shape[0]] = a[:]
        off += a.shape[0]
        del a
    del arr
    for p, _, _ in infos:
        os.remove(p)


def _merge_offset_npy(paths: list[str], out_file: str) -> None:
    """Merge several offset (.npy) files, shifting each by the running base."""
    infos, bases = [], []
    base = 0
    for p in paths:
        a = np.load(p, mmap_mode="r")
        infos.append((p, a.shape[0]))
        bases.append(base)
        base += int(a[-1])
        del a
    total = base + 1
    arr = np.lib.format.open_memmap(out_file, mode="w+", dtype=np.int64, shape=(total,))
    w = 0
    for (p, ln), b in zip(infos, bases):
        a = np.load(p, mmap_mode="r")
        if w == 0:
            arr[0:ln] = a[:]          # first chunk keeps its leading 0
            w = ln
        else:
            arr[w:w + ln - 1] = a[1:] + b
            w += ln - 1
        del a
        os.remove(p)
    del arr


class Store:
    """Memory-mapped view over a store directory."""

    def __init__(self, path: str, with_ids: bool = True):
        self.path = path
        load = lambda n: np.load(os.path.join(path, n), mmap_mode="r")  # noqa: E731
        self.name_b, self.name_o = load("name_b.npy"), load("name_o.npy")
        self.addr_b, self.addr_o = load("addr_b.npy"), load("addr_o.npy")
        self.atoms_flat, self.atoms_o = load("atoms_flat.npy"), load("atoms_o.npy")
        self.atoms_flag = load("atoms_flag.npy")
        self.state = load("state.npy")
        self.country = load("country.npy")
        self.id_hash = load("id_hash.npy")
        self.n = int(self.atoms_o.size) - 1
        if with_ids:
            with open(os.path.join(path, "ids.txt"), encoding="utf-8") as f:
                self.ids = [ln.rstrip("\n") for ln in f]
        else:
            self.ids = []

    def name_str(self, i: int) -> str:
        return bytes(self.name_b[self.name_o[i]:self.name_o[i + 1]]).decode("utf-8")

    def addr_str(self, i: int) -> str:
        return bytes(self.addr_b[self.addr_o[i]:self.addr_o[i + 1]]).decode("utf-8")

    def id(self, i: int) -> str:
        return self.ids[i]


def read_source_iter(path: str, limit: int | None = None, chunk: int = 25_000):
    """Stream a source TSV as ``(ids, rows)`` chunks, honouring an optional limit."""
    seen = 0
    for ids, rows in iter_source(path, chunk):
        if limit is not None:
            room = limit - seen
            if room <= 0:
                return
            ids, rows = ids[:room], rows[:room]
        seen += len(ids)
        yield ids, rows


def iter_source(path: str, chunk: int = 25_000):
    """Yield ``(ids, rows)`` chunks from a source TSV, streaming line by line."""
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        ci = {c: i for i, c in enumerate(header)}
        i_id, i_nm = ci["entity_id"], ci["business_name"]
        i_ad, i_co = ci["business_address"], ci["country"]
        ncol = len(header)
        ids: list[str] = []
        rows: list[tuple] = []
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < ncol:
                parts += [""] * (ncol - len(parts))
            ids.append(parts[i_id])
            rows.append((parts[i_nm], parts[i_ad], parts[i_co]))
            if len(ids) >= chunk:
                yield ids, rows
                ids, rows = [], []
        if ids:
            yield ids, rows
