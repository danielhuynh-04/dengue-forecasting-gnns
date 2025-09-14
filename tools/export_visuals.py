# -*- coding: utf-8 -*-
"""
Xuất các trực quan hóa nâng cao (offline HTML, tự chứa):
- Choropleth timelapse theo tuần
- Scatter centroid (true/pred) có lọc/hover
- Network animation (lan truyền theo tuần)
- (Optional) Biome layer overlay nếu có

Phụ thuộc:
  pandas, pyarrow, geopandas (chỉ cần nếu tính lại gì đó), plotly
"""

from __future__ import annotations
import argparse
import json
import pathlib
import sys
from typing import List, Tuple

import pandas as pd
import pyarrow.parquet as pq
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# ================= PATHS =================
ROOT = pathlib.Path(__file__).resolve().parents[1]
VIZ_DIR = ROOT / "visualizations"
GEO_DIR = VIZ_DIR / "geo"
PREP_DIR = VIZ_DIR / "prepared"
EXPORT_DIR = VIZ_DIR / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

GEOJSON_PATH = GEO_DIR / "geojs-100-mun.json"
CENTROIDS_PQ = GEO_DIR / "muni_centroids.parquet"

PANEL_DS_DIR = PREP_DIR / "panel_node_ds"     # dataset đã chuẩn hoá (geocode, year, epiweek, y_true, y_pred, ...)
CHORO_DIR = PREP_DIR / "choropleth"           # *.jsonl (per-week)
EDGES_PQ = PREP_DIR / "edges.parquet"         # (src, dst) / (geocode_src, geocode_dst) + weight(optional)

# =============== TIỆN ÍCH ===============
def _load_geojson():
    if not GEOJSON_PATH.exists():
        print(f"[WARN] Không thấy GeoJSON: {GEOJSON_PATH}")
        return None
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        gj = json.load(f)
    # cố định featureidkey = 'properties.id' hoặc 'properties.geocode'
    # dò key hợp lệ:
    sample_props = None
    for feat in gj.get("features", [])[:1]:
        sample_props = feat.get("properties", {})
    if not sample_props:
        print("[WARN] GeoJSON trống hoặc không có properties.")
    return gj

def _guess_featureidkey(geojson) -> str:
    # Ưu tiên id, sau đó geocode
    keys = []
    if not geojson or "features" not in geojson or not geojson["features"]:
        return "properties.id"
    props = geojson["features"][0].get("properties", {})
    for k in ["id", "geocode", "GEOCODE", "CD_MUN", "CD_GEOCMU"]:
        if k in props:
            keys.append(k)
            break
    return f"properties.{keys[0] if keys else 'id'}"

def _load_centroids() -> pd.DataFrame:
    if not CENTROIDS_PQ.exists():
        print(f"[WARN] Không thấy centroids: {CENTROIDS_PQ}")
        return pd.DataFrame(columns=["geocode", "lon", "lat"])
    return pd.read_parquet(CENTROIDS_PQ)

def _load_panel_sample(max_rows: int = 3_000_000) -> pd.DataFrame:
    """Đọc dataset panel_node_ds theo lô để tránh OOM, gộp dần tới max_rows."""
    if not PANEL_DS_DIR.exists():
        print(f"[WARN] Không thấy dataset: {PANEL_DS_DIR}")
        return pd.DataFrame()
    ds = pq.ParquetDataset(PANEL_DS_DIR.as_posix())
    rows = []
    count = 0
    for piece in ds.pieces:
        df = piece.read().to_pandas()
        # giữ cột chính (nhẹ nhất có thể)
        keep = [c for c in df.columns if c.lower() in {
            "geocode", "year", "epiweek", "y_true", "y_pred",
            "y_true_log", "y_pred_log", "split"
        }]
        df = df[keep]
        rows.append(df)
        count += len(df)
        if count >= max_rows:
            break
    if rows:
        out = pd.concat(rows, ignore_index=True)
        # ép kiểu để hover/text mượt:
        for c in ["year", "epiweek", "geocode"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
        for c in ["y_true", "y_pred"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        return out.dropna(subset=["geocode"]).reset_index(drop=True)
    return pd.DataFrame()

def _list_choro_jsonl() -> List[pathlib.Path]:
    if not CHORO_DIR.exists():
        return []
    return sorted(CHORO_DIR.glob("*.jsonl"))

def _load_edges(max_edges: int = 20000) -> pd.DataFrame:
    if not EDGES_PQ.exists():
        print(f"[WARN] Không thấy edges: {EDGES_PQ}")
        return pd.DataFrame(columns=["src", "dst"])
    df = pd.read_parquet(EDGES_PQ)
    # Chuẩn hóa tên cột
    cols = [c.lower() for c in df.columns]
    mapper = dict(zip(df.columns, cols))
    df = df.rename(columns=mapper)
    if "src" not in df.columns or "dst" not in df.columns:
        # thử geocode_src/dst
        if {"geocode_src", "geocode_dst"}.issubset(df.columns):
            df = df.rename(columns={"geocode_src": "src", "geocode_dst": "dst"})
        else:
            # thử edge_list.csv layout khác
            guess = [c for c in df.columns if "src" in c or "origin" in c]
            guess2 = [c for c in df.columns if "dst" in c or "target" in c]
            if guess and guess2:
                df = df.rename(columns={guess[0]: "src", guess2[0]: "dst"})
    # cắt nhỏ theo giới hạn
    if len(df) > max_edges:
        df = df.sample(n=max_edges, random_state=42)
    # ép kiểu
    for c in ["src", "dst"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df.dropna(subset=["src", "dst"]).astype({"src": "int64", "dst": "int64"})

def _epi_label(y: int, w: int) -> str:
    return f"{y}-W{int(w)%100:02d}" if pd.notna(y) and pd.notna(w) else "NA"

# ========== 1) Choropleth timelapse ==========
def export_choropleth_timelapse(geojson, featureidkey: str):
    jsonls = _list_choro_jsonl()
    if not jsonls:
        print("[CHORO] Bỏ qua (không có prepared/choropleth/*.jsonl).")
        return

    # Đọc gộp các tuần -> wide hoặc long
    frames = []
    weeks = []
    for p in jsonls:
        # file name có thể dạng: 2021-202101.jsonl, 2019-201952.jsonl ...
        week_key = p.stem  # dùng làm nhãn slider
        weeks.append(week_key)
        # mỗi dòng là {"geocode":..., "y_true":..., "y_pred":...}
        df = pd.read_json(p, lines=True)
        # lấy một metric chính để tô màu (ưu tiên y_pred nếu có)
        value_col = "y_pred" if "y_pred" in df.columns else ("y_true" if "y_true" in df.columns else None)
        if value_col is None:
            continue
        df = df[["geocode", value_col]].rename(columns={value_col: "value"})
        df["week_key"] = week_key
        frames.append(df)

    if not frames:
        print("[CHORO] Không đọc được dữ liệu từ jsonl.")
        return
    data = pd.concat(frames, ignore_index=True).dropna(subset=["geocode"])
    data["geocode"] = pd.to_numeric(data["geocode"], errors="coerce").astype("Int64")

    # Dùng Choropleth (projection) -> không cần Mapbox token
    # Slider frames
    fig = go.Figure()

    # màu thống nhất
    zmax = data["value"].quantile(0.99)
    zmin = 0.0

    # frame đầu
    w0 = sorted(data["week_key"].unique())[0]
    d0 = data[data["week_key"] == w0]
    fig.add_trace(
        go.Choropleth(
            geojson=geojson,
            locations=d0["geocode"],
            z=d0["value"],
            featureidkey=featureidkey,
            colorscale="YlOrRd",
            zmin=zmin,
            zmax=zmax,
            marker_line_width=0.1,
            colorbar_title="Incidence (pred/true)",
            hovertemplate="<b>%{location}</b><br>Value: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_geos(fitbounds="locations", visible=False)

    # Tạo frames
    frames_plotly = []
    for wk in sorted(data["week_key"].unique()):
        dd = data[data["week_key"] == wk]
        frames_plotly.append(go.Frame(
            data=[go.Choropleth(
                geojson=geojson,
                locations=dd["geocode"],
                z=dd["value"],
                featureidkey=featureidkey,
                colorscale="YlOrRd",
                zmin=zmin,
                zmax=zmax,
                marker_line_width=0.1,
                hovertemplate="<b>%{location}</b><br>Value: %{z:.3f}<extra></extra>",
            )],
            name=wk
        ))
    fig.frames = frames_plotly

    # Slider & buttons
    steps = []
    for fr in fig.frames:
        steps.append(dict(method="animate", args=[[fr.name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}], label=fr.name))
    sliders = [dict(active=0, pad={"t": 30}, steps=steps)]
    fig.update_layout(
        title="Choropleth Timelapse (weekly)",
        sliders=sliders,
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=1.05, y=1.15,
            buttons=[
                dict(label="▶ Play", method="animate", args=[None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True}]),
                dict(label="⏸ Pause", method="animate", args=[[None], {"mode": "immediate"}]),
            ],
        )],
        margin=dict(l=0, r=0, t=60, b=0),
    )

    out = EXPORT_DIR / "choropleth_timelapse.html"
    pio.write_html(fig, file=out.as_posix(), full_html=True, include_plotlyjs="inline")
    print(f"[CHORO] ✅ {out}")

# ========== 2) Scatter centroid ==========
def export_centroid_scatter(centroids: pd.DataFrame, panel_sample: pd.DataFrame):
    if centroids.empty or panel_sample.empty:
        print("[SCATTER] Bỏ qua (thiếu centroid hoặc panel).")
        return

    df = panel_sample.merge(centroids, how="left", on="geocode")
    df = df.dropna(subset=["lon", "lat"])

    # Chọn metric hiển thị màu
    color_col = "y_pred" if "y_pred" in df.columns else ("y_true" if "y_true" in df.columns else None)
    if color_col is None:
        print("[SCATTER] Bỏ qua (không có y_true/y_pred).")
        return

    df["epi_label"] = df.apply(lambda r: f"{r['year']}-W{int(r['epiweek'])%100:02d}" if pd.notna(r["year"]) and pd.notna(r["epiweek"]) else "NA", axis=1)

    # vẽ frame đầu
    week0 = sorted(df["epi_label"].dropna().unique())[0]
    d0 = df[df["epi_label"] == week0]

    fig = go.Figure(data=[
        go.Scattergeo(
            lon=d0["lon"], lat=d0["lat"],
            mode="markers",
            marker=dict(size=4, opacity=0.7, color=d0[color_col], colorscale="YlGnBu", colorbar=dict(title=color_col)),
            text=[f"geocode: {g}<br>{color_col}: {v:.3f}" for g, v in zip(d0["geocode"], d0[color_col])],
            hoverinfo="text",
        )
    ])
    fig.update_geos(fitbounds="locations", visible=False)

    # frames
    frames = []
    for wk in sorted(df["epi_label"].unique()):
        dd = df[df["epi_label"] == wk]
        frames.append(go.Frame(
            data=[go.Scattergeo(
                lon=dd["lon"], lat=dd["lat"],
                mode="markers",
                marker=dict(size=4, opacity=0.7, color=dd[color_col], colorscale="YlGnBu"),
                text=[f"geocode: {g}<br>{color_col}: {v:.3f}" for g, v in zip(dd["geocode"], dd[color_col])],
                hoverinfo="text",
            )],
            name=wk
        ))
    fig.frames = frames

    steps = []
    for fr in fig.frames:
        steps.append(dict(method="animate", args=[[fr.name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                          label=fr.name))
    sliders = [dict(active=0, pad={"t": 30}, steps=steps)]

    fig.update_layout(
        title="Centroid Scatter (weekly, predicted/true)",
        sliders=sliders,
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=1.05, y=1.15,
            buttons=[
                dict(label="▶ Play", method="animate", args=[None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True}]),
                dict(label="⏸ Pause", method="animate", args=[[None], {"mode": "immediate"}]),
            ],
        )],
        margin=dict(l=0, r=0, t=60, b=10),
    )
    out = EXPORT_DIR / "centroid_scatter.html"
    pio.write_html(fig, file=out.as_posix(), full_html=True, include_plotlyjs="inline")
    print(f"[SCATTER] ✅ {out}")

# ========== 3) Network animation ==========
def export_network_animation(centroids: pd.DataFrame, edges: pd.DataFrame, panel_sample: pd.DataFrame, max_nodes: int = 3000):
    if centroids.empty or edges.empty or panel_sample.empty:
        print("[NET] Bỏ qua (thiếu centroid/edges/panel).")
        return

    # Lọc tập node xuất hiện trong panel_sample (đã sample) & centroid
    nodes = pd.DataFrame({"geocode": pd.unique(panel_sample["geocode"].dropna().astype("int64"))})
    nodes = nodes.merge(centroids, on="geocode", how="left").dropna(subset=["lon", "lat"])
    if len(nodes) > max_nodes:
        nodes = nodes.sample(n=max_nodes, random_state=42)

    # Lọc edges theo node set
    keep = set(nodes["geocode"].tolist())
    e = edges[edges["src"].isin(keep) & edges["dst"].isin(keep)]
    # nối coord
    e = e.merge(nodes[["geocode", "lon", "lat"]].rename(columns={"geocode": "src", "lon": "lon_s", "lat": "lat_s"}), on="src", how="left")
    e = e.merge(nodes[["geocode", "lon", "lat"]].rename(columns={"geocode": "dst", "lon": "lon_t", "lat": "lat_t"}), on="dst", how="left")
    e = e.dropna(subset=["lon_s", "lat_s", "lon_t", "lat_t"])

    # Metric màu node theo tuần
    color_col = "y_pred" if "y_pred" in panel_sample.columns else ("y_true" if "y_true" in panel_sample.columns else None)
    if color_col is None:
        print("[NET] Bỏ qua (không có y_true/y_pred).")
        return

    # chỉ giữ record của node trong tuần có trong sample + nodes set
    pp = panel_sample[panel_sample["geocode"].isin(keep)].copy()
    pp["epi_label"] = pp.apply(lambda r: f"{r['year']}-W{int(r['epiweek'])%100:02d}" if pd.notna(r["year"]) and pd.notna(r["epiweek"]) else "NA", axis=1)

    # vẽ frame đầu
    w0 = sorted(pp["epi_label"].dropna().unique())[0]
    n0 = pp[pp["epi_label"] == w0].merge(nodes, on="geocode", how="left").dropna(subset=["lon", "lat"])

    fig = go.Figure()

    # edges (dạng nhiều segment)
    edge_x, edge_y = [], []
    for _, row in e.iterrows():
        edge_x += [row["lon_s"], row["lon_t"], None]
        edge_y += [row["lat_s"], row["lat_t"], None]
    fig.add_trace(go.Scattergeo(
        lon=edge_x, lat=edge_y,
        mode="lines",
        line=dict(width=0.3),
        opacity=0.3,
        hoverinfo="none",
        name="edges",
    ))
    # nodes
    fig.add_trace(go.Scattergeo(
        lon=n0["lon"], lat=n0["lat"],
        mode="markers",
        marker=dict(size=4, color=n0[color_col], colorscale="Turbo", colorbar=dict(title=color_col), opacity=0.85),
        text=[f"{g}: {v:.3f}" for g, v in zip(n0["geocode"], n0[color_col])],
        hoverinfo="text",
        name="nodes",
    ))
    fig.update_geos(fitbounds="locations", visible=False)

    # frames
    frames = []
    for wk in sorted(pp["epi_label"].unique()):
        nn = pp[pp["epi_label"] == wk].merge(nodes, on="geocode", how="left").dropna(subset=["lon", "lat"])
        frames.append(go.Frame(
            data=[
                fig.data[0],  # giữ edges như cũ
                go.Scattergeo(
                    lon=nn["lon"], lat=nn["lat"],
                    mode="markers",
                    marker=dict(size=4, color=nn[color_col], colorscale="Turbo", opacity=0.85),
                    text=[f"{g}: {v:.3f}" for g, v in zip(nn["geocode"], nn[color_col])],
                    hoverinfo="text",
                    name="nodes",
                )
            ],
            name=wk
        ))
    fig.frames = frames

    steps = []
    for fr in fig.frames:
        steps.append(dict(method="animate", args=[[fr.name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}], label=fr.name))
    sliders = [dict(active=0, pad={"t": 30}, steps=steps)]

    fig.update_layout(
        title="Network Animation (lan truyền theo tuần)",
        sliders=sliders,
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=1.05, y=1.15,
            buttons=[
                dict(label="▶ Play", method="animate", args=[None, {"frame": {"duration": 250, "redraw": True}, "fromcurrent": True}]),
                dict(label="⏸ Pause", method="animate", args=[[None], {"mode": "immediate"}]),
            ],
        )],
        margin=dict(l=0, r=0, t=60, b=10),
    )

    out = EXPORT_DIR / "network_animation.html"
    pio.write_html(fig, file=out.as_posix(), full_html=True, include_plotlyjs="inline")
    print(f"[NET] ✅ {out}")

# ========== 4) Biome layer (optional) ==========
def export_biome_layer(geojson, featureidkey: str):
    # kiểm tra có thuộc tính biome/koppen không
    if not geojson or "features" not in geojson:
        print("[BIOME] Bỏ qua (GeoJSON không hợp lệ).")
        return
    props = geojson["features"][0].get("properties", {})
    biome_key = None
    for k in ["biome", "BIOME", "koppen", "KOPPEN"]:
        if k in props:
            biome_key = k
            break
    if not biome_key:
        print("[BIOME] Bỏ qua (không thấy thuộc tính biome/koppen trong properties).")
        return

    # tạo dataframe mã hoá biome -> id
    lst = []
    for feat in geojson["features"]:
        p = feat.get("properties", {})
        if "id" in p:
            code = p["id"]
        elif "geocode" in p:
            code = p["geocode"]
        else:
            continue
        lst.append({"geocode": code, "biome": p.get(biome_key)})
    df = pd.DataFrame(lst)
    df["geocode"] = pd.to_numeric(df["geocode"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["geocode"])
    # map biome về số thứ tự để tô màu
    cats = {b: i for i, b in enumerate(sorted(df["biome"].dropna().unique()))}
    df["biome_id"] = df["biome"].map(cats)

    fig = go.Figure(go.Choropleth(
        geojson=geojson,
        locations=df["geocode"],
        z=df["biome_id"],
        featureidkey=featureidkey,
        colorscale="Viridis",
        colorbar=dict(title="Biome/Köppen (id)"),
        hovertemplate="<b>%{location}</b><br>Biome: %{z}<extra></extra>",
        marker_line_width=0.1,
    ))
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(title="Biome / Köppen Layer", margin=dict(l=0, r=0, t=60, b=0))

    out = EXPORT_DIR / "biome_layer.html"
    pio.write_html(fig, file=out.as_posix(), full_html=True, include_plotlyjs="inline")
    print(f"[BIOME] ✅ {out}")

# =============== MAIN =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-nodes", type=int, default=3000, help="giới hạn node cho animation mạng")
    ap.add_argument("--max-edges", type=int, default=15000, help="giới hạn edge cho animation mạng")
    ap.add_argument("--panel-rows", type=int, default=3_000_000, help="số hàng đọc tối đa từ panel_node_ds")
    args = ap.parse_args()

    print("=== Export advanced visuals (offline HTML) ===")

    geojson = _load_geojson()
    featureidkey = _guess_featureidkey(geojson) if geojson else "properties.id"
    centroids = _load_centroids()
    panel = _load_panel_sample(max_rows=args.panel_rows)
    edges = _load_edges(max_edges=args.max_edges)

    # 1) Choropleth timelapse
    if geojson:
        export_choropleth_timelapse(geojson, featureidkey)

    # 2) Scatter centroid
    export_centroid_scatter(centroids, panel)

    # 3) Network animation
    export_network_animation(centroids, edges, panel, max_nodes=args.max_nodes)

    # 4) Biome layer (nếu có thuộc tính)
    if geojson:
        export_biome_layer(geojson, featureidkey)

    print(f"\n✅ DONE. HTML đã ở: {EXPORT_DIR}")

if __name__ == "__main__":
    main()
