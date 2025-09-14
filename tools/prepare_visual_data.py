# -*- coding: utf-8 -*-
"""
Chuẩn bị dữ liệu cho trực quan hóa nâng cao (nhẹ RAM, PyArrow 21+).
Đầu vào sau khi đã chạy convert_node_predictions.py:
  - visualizations/parquet/node_predictions_ds/  (partition year/epiweek, hive)
  - visualizations/parquet/weekly_report.parquet
  - visualizations/parquet/summary.parquet, summary.jsonl

Bổ sung:
  - GeoJSON: data/raw/geojs-100-mun.json (id == geocode) -> copy sang visualizations/geo/
  - Edge:    data/interim/edge_list.csv  hoặc  data/processed/edge_index.pt
             -> chuẩn hóa cột src,dst (int64) -> visualizations/prepared/edges.parquet

Sản phẩm:
  - visualizations/prepared/panel_node_ds/           (node panel tối giản, partition hive year/epiweek)
  - visualizations/prepared/panel_node_agg.parquet   (aggregate theo year,epiweek)
  - visualizations/prepared/choropleth/*.jsonl       (mỗi tuần 1 file, offline)
  - visualizations/prepared/geocode_meta.parquet     (meta từ panel: min/max/mean dự báo, số tuần hiện diện)
  - visualizations/prepared/edges.parquet            (nếu có edge)
  - visualizations/geo/geojs-100-mun.json            (copy)
  - visualizations/geo/muni_centroids.parquet        (nếu có geopandas+shapely)
"""

from __future__ import annotations
import os
import sys
import json
import time
import shutil
import pathlib
from typing import List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

# ======================
# ĐƯỜNG DẪN
# ======================
ROOT = pathlib.Path(__file__).resolve().parents[1]
VIZ_DIR = ROOT / "visualizations"
PARQUET_DIR = VIZ_DIR / "parquet"
PREPARED_DIR = VIZ_DIR / "prepared"
GEO_DIR = VIZ_DIR / "geo"

# Đầu vào đã có
NODE_DS_DIR = PARQUET_DIR / "node_predictions_ds"
WEEKLY_PARQUET = PARQUET_DIR / "weekly_report.parquet"
SUMMARY_PARQUET = PARQUET_DIR / "summary.parquet"

# Đầu vào bổ sung theo bạn mô tả
RAW_GEOJSON_1 = ROOT / "data" / "raw" / "geojs-100-mun.geojson"
RAW_GEOJSON_2 = ROOT / "data" / "raw" / "geojs-100-mun.json"     # bạn có file này
EDGE_CSV = ROOT / "data" / "interim" / "edge_list.csv"
EDGE_PT = ROOT / "data" / "processed" / "edge_index.pt"

# Đầu ra
PANEL_NODE_DS = PREPARED_DIR / "panel_node_ds"
PANEL_NODE_AGG = PREPARED_DIR / "panel_node_agg.parquet"
CHORO_DIR = PREPARED_DIR / "choropleth"
GEOCODE_META = PREPARED_DIR / "geocode_meta.parquet"
EDGES_PARQUET = PREPARED_DIR / "edges.parquet"
GEOJSON_OUT = GEO_DIR / "geojs-100-mun.json"
CENTROIDS_PARQUET = GEO_DIR / "muni_centroids.parquet"

# ======================
# CẤU HÌNH NHẸ RAM
# ======================
ROW_GROUP = 128_000
BATCH_SIZE = 256_000

def _ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)

def _make_write_options() -> ds.FileWriteOptions:
    fmt = ds.ParquetFileFormat()
    return fmt.make_write_options(compression="zstd")

def _tprint(msg: str):
    print(msg, flush=True)

# ======================
# CHUẨN HÓA CỘT NODE
# ======================
def _normalize_node_columns(tbl: pa.Table) -> pa.Table:
    # map tên thường -> tên gốc
    cols = {c.lower(): c for c in tbl.schema.names}
    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_split  = pick("split")
    c_geoc   = pick("geocode", "region_code", "nodeid", "node_id", "node")
    c_year   = pick("year", "Year")
    c_epi    = pick("epiweek", "Epiweek")
    c_ytrue  = pick("y_true", "y_true_real")
    c_ypred  = pick("y_pred", "y_pred_real")

    if c_geoc is None:
        raise ValueError("Thiếu geocode trong node panel!")
    if c_year is None or c_epi is None:
        raise ValueError("Thiếu year/epiweek trong node panel!")

    arrays, names = [], []
    nrows = tbl.num_rows

    # split
    if c_split:
        split_arr = pa.compute.cast(tbl[c_split], pa.string())
    else:
        split_arr = pa.array(["unknown"] * nrows, type=pa.string())
    arrays.append(split_arr); names.append("split")

    # geocode
    geoc_arr = pa.compute.cast(tbl[c_geoc], pa.int64())
    arrays.append(geoc_arr); names.append("geocode")

    # year, epiweek
    year_arr = pa.compute.cast(tbl[c_year], pa.int32())
    epi_arr  = pa.compute.cast(tbl[c_epi],  pa.int32())
    arrays.append(year_arr); names.append("year")
    arrays.append(epi_arr);  names.append("epiweek")

    # y_true
    if c_ytrue:
        ytrue_arr = pa.compute.cast(tbl[c_ytrue], pa.float32())
    else:
        ytrue_arr = pa.array([None] * nrows, type=pa.float32())
    arrays.append(ytrue_arr); names.append("y_true")

    # y_pred (>=0)
    if c_ypred:
        ypred_arr = pa.compute.cast(tbl[c_ypred], pa.float32())
        zero = pa.scalar(0.0, type=pa.float32())
        ypred_arr = pa.compute.max_element_wise(ypred_arr, zero)
    else:
        ypred_arr = pa.array([None] * nrows, type=pa.float32())
    arrays.append(ypred_arr); names.append("y_pred")

    # residual = y_true - y_pred (nếu đủ)
    try:
        residual_arr = pa.compute.subtract(ytrue_arr, ypred_arr)
        residual_arr = pa.compute.cast(residual_arr, pa.float32())
    except Exception:
        residual_arr = pa.array([None] * nrows, type=pa.float32())
    arrays.append(residual_arr); names.append("residual")

    return pa.Table.from_arrays(arrays, names=names)


# ======================
# 1) BUILD PANEL DATASET GỌN TỪ node_predictions_ds
# ======================
def build_panel_node_dataset():
    if not NODE_DS_DIR.exists():
        raise FileNotFoundError(f"Không thấy dataset node: {NODE_DS_DIR}")

    if PANEL_NODE_DS.exists():
        shutil.rmtree(PANEL_NODE_DS, ignore_errors=True)
    _ensure_dir(PANEL_NODE_DS)

    partitioning = ds.partitioning(
        pa.schema([pa.field("year", pa.int32()), pa.field("epiweek", pa.int32())]),
        flavor="hive"
    )

    src = ds.dataset(NODE_DS_DIR, format="parquet", partitioning="hive")

    wanted = [
        "Split","split","geocode","region_code","NodeID","nodeid",
        "Year","year","Epiweek","epiweek",
        "y_true","y_true_real","y_pred","y_pred_real"
    ]
    src_cols = set(src.schema.names)
    proj = [c for c in wanted if c in src_cols] or None

    scanner = src.scanner(columns=proj, batch_size=BATCH_SIZE, use_threads=True)

    total = 0
    t0 = time.time()
    _tprint("[PANEL] Chuẩn hóa & ghi dataset gọn -> prepared/panel_node_ds/")

    for i, batch in enumerate(scanner.to_batches()):
        tbl = pa.Table.from_batches([batch])
        tbl = _normalize_node_columns(tbl)

        ds.write_dataset(
            data=tbl,
            base_dir=PANEL_NODE_DS.as_posix(),
            format="parquet",
            partitioning=partitioning,
            existing_data_behavior="overwrite_or_ignore",
            file_options=_make_write_options(),
            max_rows_per_group=ROW_GROUP,
            min_rows_per_group=ROW_GROUP,
        )
        total += tbl.num_rows
        if (i+1) % 50 == 0:
            _tprint(f"  ... đã ghi {total:,} dòng")

    _tprint(f"[PANEL] xong {total:,} dòng | {time.time()-t0:.1f}s")

# ======================
# 2) AGG THEO WEEK
# ======================
def build_week_agg():
    _tprint("[AGG] Tổng hợp theo (year, epiweek) -> panel_node_agg.parquet")
    src = ds.dataset(PANEL_NODE_DS, format="parquet", partitioning="hive")

    # chỉ lấy y_pred, y_true
    cols = ["geocode", "year", "epiweek", "y_pred", "y_true"]
    scanner = src.scanner(columns=[c for c in cols if c in src.schema.names],
                          batch_size=BATCH_SIZE, use_threads=True)

    # gom theo pandas (streaming batches)
    df_acc = []
    for batch in scanner.to_batches():
        df = batch.to_pandas(types_mapper=pd.ArrowDtype).astype({
            "geocode": "int64", "year": "int32", "epiweek": "int32"
        }, errors="ignore")
        df_acc.append(df)

    if not df_acc:
        raise RuntimeError("Không đọc được dữ liệu panel_node_ds!")

    df = pd.concat(df_acc, ignore_index=True)
    agg = df.groupby(["year","epiweek"]).agg(
        n=("geocode","size"),
        mean_pred=("y_pred","mean"),
        sum_pred=("y_pred","sum"),
        mean_true=("y_true","mean"),
        sum_true=("y_true","sum"),
    ).reset_index()

    _ensure_dir(PREPARED_DIR)
    agg.to_parquet(PANEL_NODE_AGG, index=False)
    _tprint(f"[AGG] Ghi: {PANEL_NODE_AGG} ({agg.shape[0]} tuần)")

# ======================
# 3) CHOROPLETH JSONL (mỗi tuần 1 file)
# ======================
def build_choropleth_jsonl():
    _tprint("[CHORO] Sinh choropleth JSONL per-week (offline)")
    _ensure_dir(CHORO_DIR)
    # đọc panel_node_ds gọn, chỉ cần (geocode,year,epiweek,y_pred,y_true)
    src = ds.dataset(PANEL_NODE_DS, format="parquet", partitioning="hive")
    cols = ["geocode","year","epiweek","y_pred","y_true"]
    scanner = src.scanner(columns=[c for c in cols if c in src.schema.names],
                          batch_size=BATCH_SIZE, use_threads=True)

    # gom bằng dict {(y,e): list}
    buckets = {}
    for batch in scanner.to_batches():
        df = batch.to_pandas()
        for (y,e), g in df.groupby(["year","epiweek"], sort=False):
            buckets.setdefault((int(y),int(e)), []).append(
                g[["geocode","y_pred","y_true"]].astype({"geocode":"int64"})
            )

    # ghi từng tuần
    for (y,e), parts in buckets.items():
        dfe = pd.concat(parts, ignore_index=True)
        out = CHORO_DIR / f"week_{y}_{e}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for _, r in dfe.iterrows():
                obj = {
                    "geocode": int(r["geocode"]),
                    "y_pred": None if pd.isna(r["y_pred"]) else float(r["y_pred"]),
                    "y_true": None if pd.isna(r["y_true"]) else float(r["y_true"]),
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    _tprint(f"[CHORO] Done -> {CHORO_DIR}")

# ======================
# 4) EDGE: CSV hoặc .pt
# ======================
def _load_edges_from_csv(path: pathlib.Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # chuẩn hóa cột
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None
    c_src = pick("src","source","u","i","from")
    c_dst = pick("dst","target","v","j","to")
    if c_src is None or c_dst is None:
        raise ValueError(f"edge_list.csv không tìm thấy cột src/dst (hoặc tương đương) trong {path}")
    out = df[[c_src, c_dst]].rename(columns={c_src:"src", c_dst:"dst"}).astype({"src":"int64","dst":"int64"})
    return out

def _load_edges_from_pt(path: pathlib.Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        import torch
    except Exception:
        _tprint("[EDGE] WARNING: Không có torch -> bỏ qua .pt")
        return None
    obj = torch.load(path, map_location="cpu")
    # thường edge_index dạng Tensor [2, E]
    if isinstance(obj, dict) and "edge_index" in obj:
        ei = obj["edge_index"]
    else:
        ei = obj
    if hasattr(ei, "shape") and len(ei.shape) == 2 and ei.shape[0] == 2:
        src = ei[0].to("cpu").numpy().astype("int64")
        dst = ei[1].to("cpu").numpy().astype("int64")
        return pd.DataFrame({"src": src, "dst": dst})
    else:
        raise ValueError(f"edge_index.pt không ở dạng [2, E]: {path}")

def build_edges():
    df = None
    if EDGE_CSV.exists():
        _tprint(f"[EDGE] Đọc CSV: {EDGE_CSV}")
        df = _load_edges_from_csv(EDGE_CSV)
    if df is None and EDGE_PT.exists():
        _tprint(f"[EDGE] Đọc PT: {EDGE_PT}")
        df = _load_edges_from_pt(EDGE_PT)

    if df is None:
        _tprint("[EDGE] WARNING: Không tìm thấy edge_list.csv hoặc edge_index.pt, bỏ qua.")
        return

    _ensure_dir(PREPARED_DIR)
    # loại self-loop, trùng
    df = df[df["src"] != df["dst"]].drop_duplicates().reset_index(drop=True)
    df.to_parquet(EDGES_PARQUET, index=False)
    _tprint(f"[EDGE] Ghi: {EDGES_PARQUET} (E={len(df):,})")

# ======================
# 5) GEOJSON: copy & (tuỳ chọn) centroids
# ======================
def copy_geojson_and_centroids():
    src = None
    if RAW_GEOJSON_1.exists():
        src = RAW_GEOJSON_1
    elif RAW_GEOJSON_2.exists():
        src = RAW_GEOJSON_2

    if src is None:
        _tprint("[GEO] WARNING: Không tìm thấy geojs-100-mun.(geo)json trong data/raw/, bỏ qua copy.")
        return

    _ensure_dir(GEO_DIR)
    shutil.copyfile(src, GEOJSON_OUT)
    _tprint(f"[GEO] Copied -> {GEOJSON_OUT}")

    # Tạo centroids nếu có geopandas + shapely
    try:
        import geopandas as gpd
        gdf = gpd.read_file(GEOJSON_OUT)
        # cột id có thể nằm ở gdf["id"] hoặc gdf.properties.id; dùng logic đơn giản
        if "id" in gdf.columns:
            gdf["geocode"] = gdf["id"].astype("int64")
        elif "geocode" in gdf.columns:
            gdf["geocode"] = gdf["geocode"].astype("int64")
        else:
            # thử từ properties
            if "properties" in gdf.columns and "id" in gdf["properties"].iloc[0]:
                gdf["geocode"] = gdf["properties"].apply(lambda p: int(p["id"]))
            else:
                raise ValueError("GeoJSON không có id/geocode.")

        cent = gdf.geometry.centroid
        out = pd.DataFrame({
            "geocode": gdf["geocode"].astype("int64"),
            "lon": cent.x.astype("float64"),
            "lat": cent.y.astype("float64"),
        })
        out.to_parquet(CENTROIDS_PARQUET, index=False)
        _tprint(f"[CENTROID] Ghi: {CENTROIDS_PARQUET} ({len(out):,} muni)")
    except Exception as e:
        _tprint("[CENTROID] WARNING: Không tạo được centroids (thiếu geopandas/shapely hoặc GeoJSON không hỗ trợ).")

# ======================
# 6) META GEOCODE (tĩnh) — thống kê nhanh cho UI
# ======================
def build_geocode_meta():
    src = ds.dataset(PANEL_NODE_DS, format="parquet", partitioning="hive")
    cols = ["geocode","y_pred","y_true"]
    scanner = src.scanner(columns=[c for c in cols if c in src.schema.names],
                          batch_size=BATCH_SIZE, use_threads=True)
    s_min, s_max, s_mean, n_weeks = {}, {}, {}, {}
    for batch in scanner.to_batches():
        df = batch.to_pandas()
        g = df.groupby("geocode", sort=False)
        # thống kê nhanh
        s_min.update(g["y_pred"].min().to_dict())
        s_max.update(g["y_pred"].max().to_dict())
        s_mean.update(g["y_pred"].mean().to_dict())
        n_weeks.update(g.size().to_dict())

    keys = sorted(n_weeks.keys())
    meta = pd.DataFrame({
        "geocode": keys,
        "pred_min": [s_min.get(k, None) for k in keys],
        "pred_max": [s_max.get(k, None) for k in keys],
        "pred_mean": [s_mean.get(k, None) for k in keys],
        "weeks": [n_weeks.get(k, 0) for k in keys],
    })
    _ensure_dir(PREPARED_DIR)
    meta.to_parquet(GEOCODE_META, index=False)
    _tprint(f"[ENRICH] Ghi meta tĩnh: {GEOCODE_META} ({len(meta):,} geocode)")

# ======================
# MAIN
# ======================
def main():
    print("=== Prepare visual data ===")
    build_panel_node_dataset()     # OK
    build_week_agg()               # OK
    build_choropleth_jsonl()       # OK
    build_edges()                  # NEW: CSV or PT
    copy_geojson_and_centroids()   # NEW: copy + (optional) centroids
    build_geocode_meta()           # OK
    print("\n✅ DONE: Prepared all visual data in 'visualizations/prepared' and 'visualizations/geo'.")

if __name__ == "__main__":
    main()
