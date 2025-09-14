# tools/convert_node_predictions.py
# -*- coding: utf-8 -*-
"""
Chuyển đổi các file .csv/.json sang Parquet/JSONL (tương thích PyArrow 21+).
- Không dùng WriterProperties/append (API cũ).
- Dùng ds.write_dataset với partition Hive (cần schema).
- Thân thiện máy i7-6700 / RAM 16GB, tránh OOM bằng chunksize.

Pipeline:
  1) weekly_report.csv  -> weekly_report.parquet (zstd)
  2) gat_global_summary.json -> summary.jsonl + summary.parquet
  3) node_predictions.csv (lớn) -> Parquet Dataset partition Year/Epiweek (Hive),
     ghi theo shard=xxxxx để tránh đụng tên file.
"""

from __future__ import annotations
import os
import sys
import json
import time
import shutil
import pathlib
from typing import Dict, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

# ======================
# CẤU HÌNH ĐƯỜNG DẪN
# ======================
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_INTERIM = ROOT / "data" / "interim"
VIZ_DIR = ROOT / "visualizations"
PARQUET_DIR = VIZ_DIR / "parquet"
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

# Input
WEEKLY_CSV = DATA_INTERIM / "gat_global_weekly_report.csv"
SUMMARY_JSON = DATA_INTERIM / "gat_global_summary.json"
NODE_CSV = VIZ_DIR / "data" / "node_predictions.csv"

# Output
WEEKLY_PARQUET = PARQUET_DIR / "weekly_report.parquet"
SUMMARY_JSONL = PARQUET_DIR / "summary.jsonl"
SUMMARY_PARQUET = PARQUET_DIR / "summary.parquet"
NODE_DS_DIR = PARQUET_DIR / "node_predictions_ds"

# ======================
# THÔNG SỐ KỸ THUẬT
# ======================
NODE_CHUNK_SIZE = 200_000     # phù hợp RAM 16GB
ROW_GROUP = 128_000
MAX_ROWS_PER_FILE = 2_000_000
PARQUET_COMPRESSION = "zstd"
PARTITION_KEYS = ["Year", "Epiweek"]  # sẽ dùng schema cho Hive

# ======================
# TIỆN ÍCH
# ======================
def _print_env():
    print("=== Convert to Parquet / JSONL (PyArrow 21+ ready) ===")
    print(f"Python {sys.version.split()[0]} | pandas {pd.__version__}")
    try:
        import pyarrow
        print(f"pyarrow {pyarrow.__version__}\n")
    except Exception:
        print()

def _ensure_dir(path: pathlib.Path):
    path.mkdir(parents=True, exist_ok=True)

def _normalize_year_epi(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Year", "Epiweek"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64").astype("float").astype("int64")
    if "Split" in df.columns:
        df["Split"] = df["Split"].astype("string")
    return df

def _parquet_write_table(tbl: pa.Table, out_path: pathlib.Path, *, row_group_size: int = ROW_GROUP):
    pq.write_table(
        tbl,
        out_path.as_posix(),
        compression=PARQUET_COMPRESSION,
        row_group_size=row_group_size,
    )

def _make_parquet_write_options() -> ds.FileWriteOptions:
    # PyArrow 21+: tạo write_options qua FileFormat
    fmt = ds.ParquetFileFormat()
    return fmt.make_write_options(compression=PARQUET_COMPRESSION)

def _dataset_write_chunk(
    tbl: pa.Table,
    base_dir: pathlib.Path,
    partitioning: ds.Partitioning,
    shard_idx: int,
    *,
    max_rows_per_file: int = MAX_ROWS_PER_FILE,
    row_group_size: int = ROW_GROUP,
    mode: str = "overwrite_or_ignore",
):
    shard_dir = base_dir / f"shard={shard_idx:05d}"
    _ensure_dir(shard_dir)

    ds.write_dataset(
        data=tbl,
        base_dir=shard_dir.as_posix(),
        format="parquet",
        partitioning=partitioning,
        existing_data_behavior=mode,
        file_options=_make_parquet_write_options(),
        max_rows_per_file=max_rows_per_file,
        max_rows_per_group=row_group_size,
        min_rows_per_group=row_group_size,
    )

def _estimate_csv_rows(csv_path: pathlib.Path, sample_lines: int = 200_000) -> Optional[int]:
    try:
        file_size = csv_path.stat().st_size
        n = 0
        total_bytes = 0
        with open(csv_path, "rb") as f:
            header = f.readline()
            while n < sample_lines:
                line = f.readline()
                if not line:
                    break
                total_bytes += len(line)
                n += 1
        if n == 0 or total_bytes == 0:
            return None
        avg = total_bytes / n
        return int(file_size / avg)
    except Exception:
        return None

def _node_dtype_pandas() -> Dict[str, str]:
    return {
        "NodeID": "Int64",
        "Year": "Int64",
        "Epiweek": "Int64",
        "Split": "string",
        "y_true_log": "float32",
        "y_pred_log": "float32",
        "y_true_real": "float32",
        "y_pred_real": "float32",
        "cls_prob": "float32",
        "cls_label": "Int8",
        "region_code": "string",
        "lat": "float32",
        "lon": "float32",
    }

def _coerce_node_df(df: pd.DataFrame) -> pd.DataFrame:
    dtypes = _node_dtype_pandas()
    for col, dt in dtypes.items():
        if col in df.columns:
            try:
                if dt.startswith("Int"):
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64").astype("float").astype("int64")
                elif dt == "Int8":
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int8")
                elif dt == "float32":
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
                elif dt == "string":
                    df[col] = df[col].astype("string")
            except Exception:
                pass
    # bắt buộc 2 cột partition
    for col in PARTITION_KEYS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64").astype("float").astype("int64")
    return df

# ======================
# B1) WEEKLY CSV -> PARQUET
# ======================
def convert_weekly_csv():
    if not WEEKLY_CSV.exists():
        print(f"[WEEKLY] SKIP (not found): {WEEKLY_CSV}")
        return
    print(f"[WEEKLY] Đọc: {WEEKLY_CSV}")
    df = pd.read_csv(WEEKLY_CSV)
    df = _normalize_year_epi(df)
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    print(f"[WEEKLY] Ghi Parquet: {WEEKLY_PARQUET}")
    _parquet_write_table(tbl, WEEKLY_PARQUET, row_group_size=ROW_GROUP)

# ======================
# B2) SUMMARY JSON -> JSONL + PARQUET
# ======================
def convert_summary():
    if not SUMMARY_JSON.exists():
        print(f"[SUMMARY] SKIP (not found): {SUMMARY_JSON}")
        return
    print("[SUMMARY] Đọc JSON summary...")
    with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[SUMMARY] -> {SUMMARY_JSONL} & {SUMMARY_PARQUET}")
    with open(SUMMARY_JSONL, "w", encoding="utf-8") as fo:
        fo.write(json.dumps(data, ensure_ascii=False) + "\n")
    df = pd.DataFrame([data])
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    _parquet_write_table(tbl, SUMMARY_PARQUET, row_group_size=ROW_GROUP)

# ======================
# B3) NODE_PREDICTIONS CSV -> PARQUET DATASET (HIVE)
# ======================
def convert_node_predictions():
    if not NODE_CSV.exists():
        print(f"[NODE] SKIP (not found): {NODE_CSV}")
        return

    print(f"[NODE] Bắt đầu chuyển -> dataset: {NODE_DS_DIR}")
    if NODE_DS_DIR.exists():
        print(f"[NODE] Xóa dataset cũ: {NODE_DS_DIR}")
        shutil.rmtree(NODE_DS_DIR, ignore_errors=True)
    _ensure_dir(NODE_DS_DIR)

    # *** SỬA LỖI TẠI ĐÂY ***
    # PyArrow 21: với flavor="hive" KHÔNG được dùng field_names, phải dùng schema:
    part_schema = pa.schema([
        pa.field("Year", pa.int64()),
        pa.field("Epiweek", pa.int64()),
    ])
    partitioning = ds.partitioning(part_schema, flavor="hive")

    est_rows = _estimate_csv_rows(NODE_CSV, sample_lines=200_000)
    if est_rows:
        print(f"[NODE] Ước lượng tổng số dòng: {est_rows:,}")

    dtype_map = _node_dtype_pandas()
    shard_idx = 0
    processed_rows = 0
    t0 = time.time()

    for chunk_idx, df in enumerate(pd.read_csv(NODE_CSV, chunksize=NODE_CHUNK_SIZE, dtype=dtype_map)):
        df = _coerce_node_df(df)

        missing = [c for c in PARTITION_KEYS if c not in df.columns]
        if missing:
            raise ValueError(f"Thiếu cột partition {missing} trong node_predictions.csv!")

        tbl = pa.Table.from_pandas(df, preserve_index=False)

        _dataset_write_chunk(
            tbl,
            base_dir=NODE_DS_DIR,
            partitioning=partitioning,
            shard_idx=shard_idx,
            max_rows_per_file=MAX_ROWS_PER_FILE,
            row_group_size=ROW_GROUP,
            mode="overwrite_or_ignore",
        )

        shard_idx += 1
        processed_rows += len(df)

        if est_rows:
            pct = 100.0 * processed_rows / max(1, est_rows)
            print(f"[NODE] Chunk {chunk_idx:05d} | +{len(df):,} rows | {processed_rows:,}/{est_rows:,} (~{pct:.1f}%)")
        else:
            print(f"[NODE] Chunk {chunk_idx:05d} | +{len(df):,} rows | total {processed_rows:,}")

    dt = time.time() - t0
    print(f"[NODE] Hoàn tất. Tổng {processed_rows:,} dòng → {NODE_DS_DIR} | {dt:.1f}s")

# ======================
# MAIN
# ======================
def main():
    _print_env()
    convert_weekly_csv()
    convert_summary()
    convert_node_predictions()
    print("\n✅ DONE: Tất cả file đã được chuyển sang Parquet/JSONL cho trực quan hóa.")

if __name__ == "__main__":
    main()
