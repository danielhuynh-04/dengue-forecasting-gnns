# -*- coding: utf-8 -*-
"""
Rebuild centroid (geo-only) phù hợp pipeline hiện tại.

Đầu vào:
  - data/raw/geojs-100-mun.json (hoặc .geojson)
    * có cột 'geocode' hoặc 'id' (trùng với geocode ở các file khác)

Đầu ra:
  - visualizations/geo/geojs-100-mun.json (copy, nếu tồn tại nguồn)
  - visualizations/geo/muni_centroids.parquet  (các cột: geocode, lon, lat, [name*])

Yêu cầu:
  - geopandas, pyarrow
"""

from __future__ import annotations
import pathlib
import shutil
import sys

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ========= ĐƯỜNG DẪN =========
ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_GEOJSON = None
# ưu tiên .json rồi .geojson
for cand in ["geojs-100-mun.json", "geojs-100-mun.geojson"]:
    p = RAW_DIR / cand
    if p.exists():
        RAW_GEOJSON = p
        break

VIZ_DIR = ROOT / "visualizations"
GEO_DIR = VIZ_DIR / "geo"
GEO_DIR.mkdir(parents=True, exist_ok=True)

OUT_GEOJSON_COPY = GEO_DIR / "geojs-100-mun.json"
OUT_CENTROIDS = GEO_DIR / "muni_centroids.parquet"

PARQUET_COMPRESSION = "zstd"

# ========= HÀM PHỤ =========
def _detect_geocode_column(gdf: gpd.GeoDataFrame) -> str:
    cols = [c.lower() for c in gdf.columns]
    lc2orig = {c.lower(): c for c in gdf.columns}

    # ưu tiên 'geocode', sau đó 'id'
    if "geocode" in cols:
        return lc2orig["geocode"]
    if "id" in cols:
        return lc2orig["id"]
    raise ValueError("Không tìm thấy cột khóa 'geocode' hoặc 'id' trong GeoJSON!")

def _detect_name_column(gdf: gpd.GeoDataFrame) -> str | None:
    # Thử tìm các tên hay gặp
    candidates = ["name_muni", "name", "nm_mun", "NM_MUN", "NM_MUNI", "NM_MUNIC", "NM_MUNICIP"]
    for cand in candidates:
        if cand in gdf.columns:
            return cand
        if cand.lower() in [c.lower() for c in gdf.columns]:
            # trả về đúng tên gốc (phân biệt hoa/thường)
            for c in gdf.columns:
                if c.lower() == cand.lower():
                    return c
    return None

def _write_centroids_parquet(df: pd.DataFrame, out_path: pathlib.Path):
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(tbl, out_path.as_posix(), compression=PARQUET_COMPRESSION)

# ========= MAIN =========
def main():
    print("=== Rebuild GEO/CENTROID (EPSG:5880 -> EPSG:4326) ===")

    if RAW_GEOJSON is None:
        print(f"[GEO] Không tìm thấy geojson trong {RAW_DIR}/ (geojs-100-mun.json|geojs-100-mun.geojson). Thoát.")
        sys.exit(1)

    print(f"[GEO] Đọc: {RAW_GEOJSON}")
    gdf = gpd.read_file(RAW_GEOJSON)

    # Đặt CRS mặc định nếu thiếu (đa số file geojson là EPSG:4326)
    if gdf.crs is None:
        print("[GEO] CRS trống -> gán tạm EPSG:4326 (lat/lon).")
        gdf.set_crs(epsg=4326, inplace=True)

    # Xác định cột geocode
    key_col = _detect_geocode_column(gdf)
    name_col = _detect_name_column(gdf)
    print(f"[GEO] Khóa: {key_col}" + (f" | Tên: {name_col}" if name_col else ""))

    # Ép kiểu geocode về int64
    gdf[key_col] = pd.to_numeric(gdf[key_col], errors="coerce").astype("Int64").astype("float").astype("int64")
    gdf = gdf.dropna(subset=[key_col])

    # Reproject sang CRS phẳng để tính centroid chuẩn hơn
    # Brazil Polyconic (SIRGAS 2000) — phù hợp lãnh thổ Brazil
    PROJ_CRS = 5880
    gdf_proj = gdf.to_crs(PROJ_CRS)

    # Tính centroid trong CRS phẳng
    print(f"[GEO] Tính centroid trong EPSG:{PROJ_CRS} ...")
    cent_proj = gdf_proj.geometry.centroid  # GeoSeries (CRS=5880)

    # Chuyển centroid về WGS84 (EPSG:4326) để lấy lon/lat
    cent_ll = cent_proj.to_crs(4326)
    lon = cent_ll.x.astype("float64")
    lat = cent_ll.y.astype("float64")

    out_cols = {
        "geocode": gdf[key_col].astype("int64"),
        "lon": lon,
        "lat": lat,
    }
    if name_col:
        out_cols["name"] = gdf[name_col].astype("string")

    out_df = pd.DataFrame(out_cols).dropna(subset=["geocode"])
    out_df = out_df.drop_duplicates(subset=["geocode"]).reset_index(drop=True)

    # Ghi parquet
    print(f"[CENTROID] Ghi: {OUT_CENTROIDS} ({len(out_df):,} muni)")
    _write_centroids_parquet(out_df, OUT_CENTROIDS)

    # Copy geojson source cho app offline
    try:
        if not OUT_GEOJSON_COPY.exists() or RAW_GEOJSON.read_bytes() != OUT_GEOJSON_COPY.read_bytes():
            shutil.copy2(RAW_GEOJSON, OUT_GEOJSON_COPY)
            print(f"[GEO] Copied -> {OUT_GEOJSON_COPY}")
        else:
            print(f"[GEO] GeoJSON đã có sẵn -> {OUT_GEOJSON_COPY}")
    except Exception as e:
        print(f"[GEO] WARNING: Không copy được GeoJSON: {e}")

    print("✅ DONE: muni_centroids.parquet & geojson đã sẵn sàng trong visualizations/geo/")

if __name__ == "__main__":
    main()
