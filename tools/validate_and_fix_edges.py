# tools/validate_and_fix_edges.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import pathlib
import sys
import pandas as pd
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
VIZ_PREP = ROOT / "visualizations" / "prepared"
VIZ_GEO = ROOT / "visualizations" / "geo"

EDGE_CSV = DATA_INTERIM / "edge_list.csv"
EDGE_PT = DATA_PROCESSED / "edge_index.pt"         # chỉ fallback nếu cần
CENTROIDS = VIZ_GEO / "muni_centroids.parquet"     # chứa geocode, lon, lat
OUT_PARQUET = VIZ_PREP / "edges.parquet"

GEOCODE_MIN_THRESHOLD = 100_000   # IBGE geocode 7 digits, chắc chắn > 100k

def human(n: int) -> str:
    return f"{n:,}".replace(",", "_")

def load_edges_csv(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Không thấy {path}")
    # auto delimiter + no header tolerant
    try:
        df = pd.read_csv(path, sep=None, engine="python", header=None, names=["a", "b"])
    except Exception:
        df = pd.read_csv(path, header=None, names=["a", "b"])
    # nếu file có header hợp lệ
    known = {"src", "dst", "source_geocode", "target_geocode", "u", "v", "a", "b"}
    if set(df.columns) - known:
        # có header string → đoán
        cols = {c.lower(): c for c in df.columns}
        if {"src","dst"} <= set(cols):
            df = df.rename(columns={cols["src"]:"a", cols["dst"]:"b"})
        elif {"source_geocode","target_geocode"} <= set(cols):
            df = df.rename(columns={cols["source_geocode"]:"a", cols["target_geocode"]:"b"})
        elif {"u","v"} <= set(cols):
            df = df.rename(columns={cols["u"]:"a", cols["v"]:"b"})
        else:
            # giữ nguyên a,b
            pass
    # chuẩn kiểu
    df = df[["a","b"]].dropna()
    for c in ["a","b"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().astype({"a":"int64","b":"int64"})
    return df

def load_geocodes() -> np.ndarray:
    if not CENTROIDS.exists():
        raise FileNotFoundError(
            f"Không thấy {CENTROIDS}. Hãy chạy rebuild_centroids.py trước."
        )
    c = pd.read_parquet(CENTROIDS)
    # chấp nhận cột 'id' hoặc 'geocode'
    if "geocode" not in c.columns and "id" in c.columns:
        c = c.rename(columns={"id":"geocode"})
    if "geocode" not in c.columns:
        raise ValueError("muni_centroids.parquet thiếu cột 'geocode' hoặc 'id'.")
    gcodes = pd.to_numeric(c["geocode"], errors="coerce").dropna().astype(np.int64).unique()
    gcodes.sort()
    return gcodes

def detect_mode(edges: pd.DataFrame, gcodes: np.ndarray) -> str:
    """Trả về 'geocode' hoặc 'index'."""
    a_min, b_min = edges["a"].min(), edges["b"].min()
    a_max, b_max = edges["a"].max(), edges["b"].max()
    n_nodes = len(gcodes)
    # điều kiện mạnh: tất cả < n_nodes và không có số âm -> index
    if a_min >= 0 and b_min >= 0 and a_max < n_nodes and b_max < n_nodes:
        return "index"
    # điều kiện dựa trên ngưỡng địa chỉ: có số lớn kiểu 7-digit -> geocode
    if (a_max >= GEOCODE_MIN_THRESHOLD) or (b_max >= GEOCODE_MIN_THRESHOLD):
        return "geocode"
    # fallback: nếu phần trăm số nhỏ <100k chiếm đa số → index
    small = ((edges["a"] < GEOCODE_MIN_THRESHOLD) & (edges["b"] < GEOCODE_MIN_THRESHOLD)).mean()
    return "index" if small > 0.8 else "geocode"

def map_index_to_geocode(edges: pd.DataFrame, gcodes: np.ndarray) -> pd.DataFrame:
    """Ánh xạ 0..N-1 → geocode theo thứ tự gcodes tăng dần."""
    n_nodes = len(gcodes)
    bad = edges[(edges["a"] < 0) | (edges["b"] < 0) | (edges["a"] >= n_nodes) | (edges["b"] >= n_nodes)]
    if len(bad) > 0:
        print(f"[EDGE] WARNING: Có {human(len(bad))} dòng nằm ngoài [0..{n_nodes-1}], sẽ bị bỏ.")
        edges = edges.drop(bad.index)
    mapped = pd.DataFrame({
        "src": gcodes[edges["a"].to_numpy()],
        "dst": gcodes[edges["b"].to_numpy()],
    })
    return mapped

def clean_edges(df: pd.DataFrame, gcodes: np.ndarray) -> pd.DataFrame:
    df = df.dropna()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().astype({"src":"int64","dst":"int64"})
    # bỏ self-loop
    before = len(df)
    df = df[df["src"] != df["dst"]]
    removed_loops = before - len(df)
    # bỏ cạnh trùng
    df = df.drop_duplicates()
    # lọc theo tập geocode hợp lệ
    valid = set(gcodes.tolist())
    mask = df["src"].isin(valid) & df["dst"].isin(valid)
    dropped = len(df) - mask.sum()
    df = df[mask].reset_index(drop=True)
    if removed_loops:
        print(f"[CLEAN] Bỏ {human(removed_loops)} self-loop")
    if dropped:
        print(f"[CLEAN] Bỏ {human(dropped)} cạnh nằm ngoài tập geocode hợp lệ")
    return df

def main():
    print("=== Validate & Fix Edges ===")
    # 1) geocode universe
    gcodes = load_geocodes()
    print(f"[INFO] Có {human(len(gcodes))} geocode hợp lệ từ centroid")

    # 2) load edges
    edges_raw = load_edges_csv(EDGE_CSV)
    print(f"[READ] edge_list.csv: {human(len(edges_raw))} dòng, min/max = "
          f"a[{edges_raw['a'].min()}..{edges_raw['a'].max()}], "
          f"b[{edges_raw['b'].min()}..{edges_raw['b'].max()}]")

    # 3) detect mode
    mode = detect_mode(edges_raw, gcodes)
    print(f"[DETECT] Diễn giải cạnh dạng: {mode.upper()}")

    # 4) map/normalize to geocode
    if mode == "index":
        edges_geo = map_index_to_geocode(edges_raw, gcodes)
    else:
        # đã là geocode
        edges_geo = edges_raw.rename(columns={"a":"src","b":"dst"})[["src","dst"]]

    # 5) clean
    edges_geo = clean_edges(edges_geo, gcodes)
    print(f"[OK] Sau clean còn {human(len(edges_geo))} cạnh")

    # 6) thống kê nhanh
    deg = pd.concat([
        edges_geo["src"].value_counts(),
        edges_geo["dst"].value_counts()
    ], axis=1).fillna(0).sum(axis=1).astype(int)
    print(f"[STATS] degree min/median/mean/max = "
          f"{deg.min()}/{int(deg.median())}/{deg.mean():.2f}/{deg.max()} "
          f"(trên {human(deg.shape[0])} nút có bậc > 0)")

    # 7) save parquet
    VIZ_PREP.mkdir(parents=True, exist_ok=True)
    edges_geo.to_parquet(OUT_PARQUET, index=False)
    print(f"[WRITE] → {OUT_PARQUET} (src,dst=int64)")

    # 8) gợi ý tiếp
    print("\nTiếp theo hãy chạy:")
    print("  python tools/export_dashboard.py --max-edges 12000")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
