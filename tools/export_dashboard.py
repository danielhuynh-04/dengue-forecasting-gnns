# tools/export_dashboard.py

# -*- coding: utf-8 -*-

"""

Dashboard offline mượt (không frames), UI gọn, trực quan hóa cao:

- Streaming bằng Plotly.restyle + requestAnimationFrame (không queue, không giật).

- Play/Pause một nút, Restart riêng, TỐC ĐỘ kiểu “N tuần / S giây”.

- Timeline rõ ràng (YYYY-Www), kéo là dừng và nhảy chính xác.

- Truth/Pred (Turbo), Residual (RdBu, zmid=0) với colorbar tách rời, cố định toàn kỳ.

- Colorbar tự dàn bố cục theo số lớp đang bật; margin phải tự điều chỉnh → không bị đè.

- Biome từ data/raw/environ_vars.csv (RGBA) luôn dưới cùng (không che lớp khác).

- Density, Centroids, Edges bật/tắt; Edges/Points luôn trên cùng.

"""


from __future__ import annotations

import json, argparse, pathlib

from typing import List, Tuple, Optional, Dict


import numpy as np

import pandas as pd

import pyarrow.dataset as ds

import plotly.graph_objects as go

import plotly.io as pio


# ======================

# DEFAULTS

# ======================

DEFAULT_MAX_EDGES = 12_000

DEFAULT_MAX_NODES = 3_000

DEFAULT_OUT = "visualizations/dashboard_chinhthuc.html"

DEFAULT_NO_HOVER = False

DEFAULT_START = None   # "2021_1"

DEFAULT_END = None     # "2024_24"

DEFAULT_PALETTE_NUM = "Turbo"  # Truth/Pred

DEFAULT_PALETTE_DIV = "RdBu"   # Residual (diverging, zmid=0)


# ======================

# PATHS

# ======================

ROOT = pathlib.Path(__file__).resolve().parents[1]

PREP_DIR = ROOT / "visualizations" / "prepared"

GEO_DIR = ROOT / "visualizations" / "geo"


PANEL_DS_DIR = PREP_DIR / "panel_node_ds"          # parquet partition (year=YYYY/epiweek=YYYYWW)

EDGES_PARQUET = PREP_DIR / "edges.parquet"


GEOJSON_PATH = GEO_DIR / "geojs-100-mun.json"

CENTROIDS_PARQUET = GEO_DIR / "muni_centroids.parquet"


ENVIRON_CSV = ROOT / "data" / "raw" / "environ_vars.csv"  # geocode, name_muni, biome, koppen, ...


# ======================

# HELPERS

# ======================

def _read_with_encodings(path: pathlib.Path, reader=pd.read_csv, **kw):

    encs = ["utf-8", "utf-8-sig", "latin1", "cp1252"]

    last_err = None

    for enc in encs:

        try:

            return reader(path, encoding=enc, **kw)

        except Exception as e:

            last_err = e

    # fallback latin1 replace

    with open(path, "rb") as f:

        txt = f.read().decode("latin1", errors="replace")

    from io import StringIO

    return reader(StringIO(txt), **kw)


def list_weeks(panel_ds_dir: pathlib.Path) -> List[Tuple[int,int,str]]:

    ds0 = ds.dataset(panel_ds_dir.as_posix(), format="parquet", partitioning="hive")

    weeks = set()

    for p in ds0.files:

        if "/year=" in p and "/epiweek=" in p:

            try:

                y = int(p.split("/year=")[1].split("/")[0])

                w = int(p.split("/epiweek=")[1].split("/")[0])

                weeks.add((y, w))

            except Exception:

                pass

    weeks = sorted(weeks)

    labels = [f"{y}-W{str(w % 1000).zfill(2)}" for y, w in weeks]

    return [(y, w, lbl) for (y, w), lbl in zip(weeks, labels)]


# ======================

# LOADERS

# ======================

def read_panel_week(year: int, epiweek: int, max_nodes: Optional[int]=None) -> pd.DataFrame:

    dataset = ds.dataset(PANEL_DS_DIR.as_posix(), format="parquet", partitioning="hive")

    cols = ["split","geocode","y_true","y_pred","residual","year","epiweek"]

    tbl = dataset.to_table(columns=[c for c in cols if c in dataset.schema.names],

                           filter=(ds.field("year")==year) & (ds.field("epiweek")==epiweek))

    df = tbl.to_pandas().copy()

    for c in ["geocode","y_true","y_pred","residual"]:

        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["geocode"]).copy()

    df["geocode"] = df["geocode"].astype(np.int64)

    if max_nodes and len(df) > max_nodes:

        df = df.nlargest(max_nodes, "y_true")

    return df


def load_centroids() -> pd.DataFrame:

    g = pd.read_parquet(CENTROIDS_PARQUET)

    ren = {}

    for a, b in [("id","geocode"), ("lng","lon"), ("longitude","lon"), ("latitude","lat")]:

        if a in g.columns: ren[a] = b

    g = g.rename(columns=ren)

    need = ["geocode","lon","lat"]

    for c in need:

        if c not in g.columns: raise ValueError(f"Centroids thiếu cột {c}")

    g["geocode"] = pd.to_numeric(g["geocode"], errors="coerce")

    g = g.dropna(subset=["geocode","lon","lat"]).copy()

    g["geocode"] = g["geocode"].astype(np.int64)

    if "name" not in g.columns: g["name"] = ""

    return g[["geocode","lon","lat","name"]]


def load_meta_from_environ() -> pd.DataFrame:

    if not ENVIRON_CSV.exists():

        return pd.DataFrame()

    df = _read_with_encodings(ENVIRON_CSV, keep_default_na=False, dtype={"geocode":"Int64"})

    lower = {c.lower(): c for c in df.columns}

    def pick(*opts):

        for o in opts:

            if o in lower: return lower[o]

        return None

    c_geo = pick("geocode","id","code")

    c_name= pick("name_muni","name","municipality","muni")

    c_bio = pick("biome")

    c_kop = pick("koppen","köppen")

    cols = [c_geo] + [x for x in [c_name,c_bio,c_kop] if x]

    out = df[cols].copy()

    out = out.rename(columns={

        c_geo:"geocode",

        c_name:"name_muni" if c_name else "name_muni",

        c_bio:"biome" if c_bio else "biome",

        c_kop:"koppen" if c_kop else "koppen",

    })

    out["geocode"] = pd.to_numeric(out["geocode"], errors="coerce").astype("Int64")

    out = out.dropna(subset=["geocode"]).copy()

    out["geocode"] = out["geocode"].astype(np.int64)

    for c in ["name_muni","biome","koppen"]:

        if c not in out.columns: out[c] = ""

        out[c] = out[c].astype(str)

    return out.drop_duplicates("geocode")


def load_edges(max_edges: int, cent: pd.DataFrame) -> Optional[pd.DataFrame]:

    if not EDGES_PARQUET.exists() or max_edges <= 0: return None

    e = pd.read_parquet(EDGES_PARQUET)

    ren = {}

    if "src" not in e.columns and "a" in e.columns: ren["a"] = "src"

    if "dst" not in e.columns and "b" in e.columns: ren["b"] = "dst"

    e = e.rename(columns=ren)

    if not {"src","dst"}.issubset(e.columns): return None

    e = e.head(max_edges).copy()

    c1 = cent[["geocode","lon","lat"]].rename(columns={"geocode":"src","lon":"lon_src","lat":"lat_src"})

    c2 = cent[["geocode","lon","lat"]].rename(columns={"geocode":"dst","lon":"lon_dst","lat":"lat_dst"})

    e = e.merge(c1, on="src", how="left").merge(c2, on="dst", how="left")

    e = e.dropna(subset=["lon_src","lat_src","lon_dst","lat_dst"])

    return e


# ======================

# HOVER

# ======================

def build_hover(df: pd.DataFrame, col: str, meta: pd.DataFrame, no_hover: bool) -> List[str]:

    if no_hover: return [""] * len(df)

    show = df[["geocode", col]].copy()

    if meta is not None and len(meta) > 0:

        keep = [c for c in ["geocode","biome","koppen","name_muni"] if c in meta.columns]

        show = show.merge(meta[keep], on="geocode", how="left")

    out = []

    for _, r in show.iterrows():

        try: sval = f"{float(r[col]):.3f}"

        except: sval = str(r[col])

        s = f"<b>{int(r.geocode)}</b><br>{col}: {sval}"

        if "name_muni" in show.columns and isinstance(r.get("name_muni"), str) and r.get("name_muni"):

            s += f"<br>{r['name_muni']}"

        if "biome" in show.columns and isinstance(r.get("biome"), str) and r.get("biome"):

            s += f"<br>Biome: {r['biome']}"

        if "koppen" in show.columns and isinstance(r.get("koppen"), str) and r.get("koppen"):

            s += f"<br>Köppen: {r['koppen']}"

        out.append(s)

    return out


# ======================

# TRACES (order controls z-order)

# Biome (0) -> Truth (1) -> Pred (2) -> Residual (3) -> Density (4) -> Centroids (5) -> Edges (6)

# ======================

def trace_biome(geojson_path: pathlib.Path, geocodes: np.ndarray, biome: List[str]) -> go.Choroplethmapbox:

    cats = pd.Categorical(pd.Series(biome, dtype="string").fillna("unknown").astype(str))

    z = cats.codes.astype(float)

    base = [

        "rgba(27,158,119,0.30)","rgba(217,95,2,0.30)","rgba(117,112,179,0.30)",

        "rgba(231,41,138,0.30)","rgba(102,166,30,0.30)","rgba(230,171,2,0.30)",

        "rgba(166,118,29,0.30)","rgba(102,102,102,0.30)","rgba(31,120,180,0.30)","rgba(178,223,138,0.30)"

    ]

    uniq = list(pd.unique(cats))

    n = max(1, min(10, len(uniq)))

    colorscale = []

    for i in range(n):

        t = i/max(1,n-1); c = base[i % len(base)]

        colorscale += [[t, c], [min(1.0, t+1e-6), c]]

    hov = [f"<b>{int(g)}</b><br>Biome: {b} / Quần xã: {b}"

           for g,b in zip(geocodes, cats.astype(str).tolist())]

    return go.Choroplethmapbox(

        name="Biome",

        geojson=json.load(open(geojson_path, "r", encoding="utf-8")),

        featureidkey="properties.id",

        locations=[str(int(x)) for x in geocodes],

        z=z, colorscale=colorscale,

        zmin=float(np.nanmin(z)), zmax=float(np.nanmax(z) if len(z) else 1.0),

        marker_line_width=0,

        text=hov, hovertemplate="%{text}<extra></extra>",

        visible=False, showscale=False

    )


def trace_choro(geojson_path: pathlib.Path, name: str, colorscale: str, colorbar_x: float, zmid=None) -> go.Choroplethmapbox:

    return go.Choroplethmapbox(

        name=name,

        geojson=json.load(open(geojson_path, "r", encoding="utf-8")),

        featureidkey="properties.id",

        locations=[], z=[],

        colorscale=colorscale,

        zmin=0, zmax=1,

        zmid=zmid,

        text=[],

        hovertemplate="%{text}<extra></extra>",

        visible=(name=="Truth"),

        marker_line_width=0.1,

        colorbar=dict(

            title=name + " | " + {"Truth":"Thực tế","Prediction":"Dự báo","Residual":"Sai số"}.get(name,name),

            thickness=16, ticklen=3, xpad=6, x=colorbar_x, y=0.5, len=0.92, yanchor="middle"

        ),

        showscale=True

    )


def trace_density() -> go.Densitymapbox:

    return go.Densitymapbox(

        name="Density",

        lon=[], lat=[], z=[],

        radius=22, opacity=0.45,

        visible=False,

        coloraxis="coloraxis2"

    )


def trace_centroids() -> go.Scattermapbox:

    return go.Scattermapbox(

        name="Centroids",

        lon=[], lat=[], mode="markers",

        marker=dict(size=8, opacity=0.65),

        text=[],

        hovertemplate="%{text}<extra></extra>",

        visible=False

    )


def trace_edges(edges: Optional[pd.DataFrame]) -> go.Scattermapbox:

    if edges is None or len(edges)==0:

        return go.Scattermapbox(name="Edges", lon=[], lat=[], visible=False)

    lon, lat = [], []

    for _, r in edges.iterrows():

        lon += [r["lon_src"], r["lon_dst"], None]

        lat += [r["lat_src"], r["lat_dst"], None]

    return go.Scattermapbox(

        name="Edges", lon=lon, lat=lat,

        mode="lines", line=dict(width=1), opacity=0.35,

        hoverinfo="skip", visible=False

    )


# ======================

# DATA PACK (per-week for JS restyle)

# ======================

def build_datapack(weeks: List[Tuple[int,int,str]],

                   meta: pd.DataFrame,

                   cent: pd.DataFrame,

                   max_nodes: int,

                   no_hover: bool) -> Tuple[Dict, Dict[str,float], List[Dict]]:

    pack, gmax, stats = {"labels": [], "weeks": [], "biome": {}}, {"true":0.0,"pred":0.0,"res":0.0}, []

    for (y,w,lbl) in weeks:

        df = read_panel_week(y, w, max_nodes=max_nodes)

        locs = df["geocode"].to_numpy()

        z_true = pd.to_numeric(df["y_true"], errors="coerce").fillna(0.0).to_numpy(float)

        z_pred = pd.to_numeric(df["y_pred"], errors="coerce").fillna(0.0).to_numpy(float)

        z_res  = pd.to_numeric(df["residual"], errors="coerce").fillna(0.0).to_numpy(float)


        if len(z_true): gmax["true"] = max(gmax["true"], float(np.nanmax(z_true)))

        if len(z_pred): gmax["pred"] = max(gmax["pred"], float(np.nanmax(z_pred)))

        if len(z_res):  gmax["res"]  = max(gmax["res"],  float(np.nanmax(np.abs(z_res))))


        hov_true = build_hover(df, "y_true", meta, no_hover)

        hov_pred = build_hover(df, "y_pred", meta, no_hover)

        hov_res  = build_hover(df, "residual", meta, no_hover)


        cx = df[["geocode","y_true"]].merge(cent, on="geocode", how="left").dropna(subset=["lon","lat"])

        dens_lon = cx["lon"].tolist(); dens_lat = cx["lat"].tolist()

        dens_z   = pd.to_numeric(cx["y_true"], errors="coerce").fillna(0.0).astype(float).tolist()


        cen_lon = cx["lon"].tolist(); cen_lat = cx["lat"].tolist()

        cen_v   = pd.to_numeric(cx["y_true"], errors="coerce").fillna(0.0)

        vmax = max(1.0, float(cen_v.max() if len(cen_v) else 1.0))

        cen_size = (4 + 10 * (cen_v / vmax)).astype(float).tolist()

        cen_text = [f"{int(g)} · {float(v):.1f}" for g,v in zip(cx["geocode"], cen_v)]


        pack["labels"].append(lbl)

        pack["weeks"].append(dict(

            locs=[int(x) for x in locs],

            z_true=z_true.tolist(), z_pred=z_pred.tolist(), z_res=z_res.tolist(),

            hov_true=hov_true, hov_pred=hov_pred, hov_res=hov_res,

            dens_lon=dens_lon, dens_lat=dens_lat, dens_z=dens_z,

            cen_lon=cen_lon, cen_lat=cen_lat, cen_size=cen_size, cen_text=cen_text

        ))


        s_true = float(np.nansum(z_true)) if len(z_true) else 0.0

        mean_pred = float(np.nanmean(z_pred)) if len(z_pred) else 0.0

        rmse = float(np.sqrt(np.nanmean((z_pred - z_true)**2))) if len(z_true) else 0.0

        tmp = pd.DataFrame({"geocode": locs, "y_true": z_true})

        top5 = tmp.nlargest(5, "y_true") if len(tmp) else pd.DataFrame(columns=["geocode","y_true"])

        top5_list = [{"geocode": int(g), "y": float(v)} for g, v in zip(top5["geocode"], top5["y_true"])]

        stats.append({"label": lbl, "sum_true": s_true, "mean_pred": mean_pred, "rmse": rmse, "top5": top5_list})


    for k in gmax: gmax[k] = max(1.0, gmax[k])

    if len(meta)>0 and "biome" in meta.columns:

        pack["biome"] = dict(geocode=meta["geocode"].astype(int).tolist(),

                             biome=meta["biome"].astype(str).tolist())

    return pack, gmax, stats


# ======================

# FIGURE + HTML (biome dưới cùng)

# ======================

SETTINGS_HTML = """

<div id="settings-panel" class="drag" style="position:absolute; top:12px; left:12px; background:rgba(255,255,255,0.98); padding:12px 14px; border-radius:12px; box-shadow:0 8px 24px rgba(0,0,0,0.15); font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif; z-index:10; min-width:240px; resize:both; overflow:auto;">

  <div id="settings-header" style="cursor:move; font-weight:800; margin-bottom:8px;">⚙️ Settings / Cài đặt</div>

  <div style="margin-bottom:6px; font-size:13px; opacity:0.8;">Layers / Bật tắt lớp</div>

  <label style="display:block; margin:4px 0;"><input type="checkbox" class="layer-toggle" data-index="0"> Biome (Sinh cảnh)</label>

  <label style="display:block; margin:4px 0;"><input type="checkbox" class="layer-toggle" data-index="1" checked> Truth (Thực tế)</label>

  <label style="display:block; margin:4px 0;"><input type="checkbox" class="layer-toggle" data-index="2"> Prediction (Dự báo)</label>

  <label style="display:block; margin:4px 0;"><input type="checkbox" class="layer-toggle" data-index="3"> Residual (Sai số)</label>

  <label style="display:block; margin:4px 0;"><input type="checkbox" class="layer-toggle" data-index="4"> Density (Mật độ)</label>

  <label style="display:block; margin:4px 0;"><input type="checkbox" class="layer-toggle" data-index="5"> Centroids (Điểm)</label>

  <label style="display:block; margin:4px 0;"><input type="checkbox" class="layer-toggle" data-index="6"> Edges (Cạnh)</label>

  <hr style="margin:10px 0;">

  <div style="font-weight:700; margin:6px 0 4px;">Palette / Bảng màu</div>

  <div>

    <label style="display:block; font-size:12px; margin:3px 0;">Truth/Pred:

      <select id="palette-num" style="width:100%; padding:6px; border-radius:8px; border:1px solid #ddd;">

        <option value="Turbo" selected>Turbo (rõ cường độ)</option>

        <option value="Viridis">Viridis (không gây hiểu lầm)</option>

        <option value="Inferno">Inferno (nhấn điểm nóng)</option>

        <option value="Reds">Reds (đơn sắc tăng dần)</option>

        <option value="Blues">Blues (đơn sắc tăng dần)</option>

        <option value="Cividis">Cividis (mù màu thân thiện)</option>

      </select>

    </label>

    <label style="display:block; font-size:12px; margin:3px 0;">Residual:

      <select id="palette-div" style="width:100%; padding:6px; border-radius:8px; border:1px solid #ddd;">

        <option value="RdBu" selected>RdBu (tâm 0)</option>

        <option value="Picnic">Picnic</option>

        <option value="Portland">Portland</option>

        <option value="Balance">Balance</option>

      </select>

    </label>

  </div>

</div>

"""


UI_HTML = """

<div id="bottom-bar" style="position:absolute; left:0; right:0; bottom:0; background:linear-gradient(180deg, rgba(255,255,255,0.88), rgba(255,255,255,0.98)); padding:8px 14px; display:grid; grid-template-columns: 420px 1fr 520px; align-items:center; gap:14px; z-index:9; border-top:1px solid #eee; font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;">

  <div id="controls" style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">

    <button id="btn-toggle"  style="padding:6px 10px; border-radius:8px; border:1px solid #1c7; background:#1c7; color:#fff; cursor:pointer;">▶ Play</button>

    <button id="btn-restart" style="padding:6px 10px; border-radius:8px; border:1px solid #888; background:#fff; cursor:pointer;">⟲ Restart</button>

    <div style="display:flex; gap:8px; align-items:center; margin-left:6px; font-size:12px;">

      <span>Tốc độ:</span>

      <label>N tuần <input id="speed-w" type="number" value="1" min="1" step="1" style="width:64px; padding:3px 6px; margin-left:4px;"></label>

      <span>/</span>

      <label>S giây <input id="speed-s" type="number" value="0.8" min="0.1" step="0.1" style="width:72px; padding:3px 6px; margin-left:4px;"></label>

      <span style="opacity:0.7;">(ví dụ: 5 tuần / 13 giây)</span>

    </div>

  </div>

  <div style="display:flex; align-items:center; gap:10px;">

    <span id="label-start" style="font-size:12px; opacity:0.75; width:78px; text-align:right;">—</span>

    <input id="range" type="range" min="0" value="0" step="1" style="flex:1;">

    <span id="label-end" style="font-size:12px; opacity:0.75; width:78px;">—</span>

    <div id="label-current" style="min-width:110px; font-weight:800; text-align:center;">—</div>

  </div>

  <div id="stats" style="font-size:12px; line-height:1.5; display:grid; grid-template-columns: repeat(2, 1fr); gap:4px 12px;">

    <div><b>Tuần/Week:</b> <span id="st-week">—</span></div>

    <div><b>RMSE:</b> <span id="st-rmse">—</span></div>

    <div><b>Tổng ca:</b> <span id="st-sum">—</span></div>

    <div><b>Mean dự báo:</b> <span id="st-mean">—</span></div>

    <div style="grid-column:1/3; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;"><b>Top-5:</b> <span id="st-top">—</span></div>

  </div>

</div>

"""


def build_figure(edges_df: Optional[pd.DataFrame], biome_meta: Optional[Dict]) -> go.Figure:

    # Biome dưới cùng

    traces = []

    if biome_meta and biome_meta.get("geocode"):

        traces.append(trace_biome(GEOJSON_PATH,

                                  np.array(biome_meta["geocode"], dtype=np.int64),

                                  biome_meta["biome"]))

    else:

        traces.append(go.Choroplethmapbox(name="Biome", locations=[], z=[], visible=False, showscale=False))

    # 3 choropleth, colorbar đặt lệch phải (sẽ tái bố trí bằng JS theo số lớp bật)

    traces.append(trace_choro(GEOJSON_PATH, "Truth",      DEFAULT_PALETTE_NUM, 1.02, None))

    traces.append(trace_choro(GEOJSON_PATH, "Prediction", DEFAULT_PALETTE_NUM, 1.10, None))

    traces.append(trace_choro(GEOJSON_PATH, "Residual",   DEFAULT_PALETTE_DIV, 1.18, 0.0))

    # Density, Points & Edges

    traces.append(trace_density())

    traces.append(trace_centroids())

    traces.append(trace_edges(edges_df))


    fig = go.Figure(data=traces)

    fig.update_layout(

        title="Dengue Spatiotemporal Dashboard (Offline)",

        margin=dict(l=0, r=210, t=54, b=130),

        mapbox=dict(style="carto-positron", center=dict(lon=-51.9253, lat=-14.2350), zoom=3.2),

        legend=dict(orientation="h", y=1.02, x=0.01),

        coloraxis2=dict(colorscale=DEFAULT_PALETTE_NUM)  # density palette

    )

    fig.update_layout(updatemenus=[])  # không dùng Plotly slider

    return fig


# JS: requestAnimationFrame + restyle, Play/Pause toggle + tốc độ N tuần / S giây

# và tái bố trí colorbar theo số lớp đang bật

JS = """

<script>

(function(){

  const plotId = 'plot';

  const pack  = JSON.parse(document.getElementById('week-pack').textContent); // {labels,weeks,biome}

  const stats = JSON.parse(document.getElementById('week-stats').textContent);

  const n = pack.labels.length;

  let iNow = 0, playing = false, lastTs = 0, accum = 0, busy = false;


  const rangeEl = document.getElementById('range');

  const btnTog  = document.getElementById('btn-toggle');

  const btnRes  = document.getElementById('btn-restart');

  const spdWEl  = document.getElementById('speed-w'); // N tuần

  const spdSEl  = document.getElementById('speed-s'); // S giây


  rangeEl.max = Math.max(0, n-1);

  document.getElementById('label-start').textContent = pack.labels[0] || '—';

  document.getElementById('label-end').textContent   = pack.labels[n-1] || '—';


  let zmax_true = parseFloat(document.getElementById('gmax-true').textContent);

  let zmax_pred = parseFloat(document.getElementById('gmax-pred').textContent);

  let zmax_res  = parseFloat(document.getElementById('gmax-res').textContent);


  function msPerWeek(){

    let w = parseFloat(spdWEl.value||'1'); if(!isFinite(w) || w<=0) w=1;

    let s = parseFloat(spdSEl.value||'1'); if(!isFinite(s) || s<=0) s=1;

    return (s*1000.0)/w; // S giây cho N tuần  => mỗi tuần = S/N giây

  }

  function setToggleUI(){

    btnTog.textContent = playing ? "⏸ Pause" : "▶ Play";

    btnTog.style.background = playing ? "#d33" : "#1c7";

    btnTog.style.borderColor = playing ? "#d33" : "#1c7";

  }


  async function applyWeek(i){

    const wk = pack.weeks[i]; if(!wk) return;

    if(busy) return; busy = true;


    const locs = wk.locs.map(x=>String(x));

    // Truth

    await Plotly.restyle(plotId, {locations:[locs], z:[wk.z_true], text:[wk.hov_true], zmin:[0], zmax:[zmax_true], zmid:[null]}, [1]);

    // Pred

    await Plotly.restyle(plotId, {locations:[locs], z:[wk.z_pred], text:[wk.hov_pred], zmin:[0], zmax:[zmax_pred], zmid:[null]}, [2]);

    // Residual (đối xứng)

    await Plotly.restyle(plotId, {locations:[locs], z:[wk.z_res],  text:[wk.hov_res],  zmin:[-zmax_res], zmax:[zmax_res], zmid:[0]}, [3]);

    // Density

    await Plotly.restyle(plotId, {lon:[wk.dens_lon], lat:[wk.dens_lat], z:[wk.dens_z]}, [4]);

    // Centroids

    await Plotly.restyle(plotId, {lon:[wk.cen_lon], lat:[wk.cen_lat], "marker.size":[wk.cen_size], text:[wk.cen_text]}, [5]);


    // UI labels

    document.getElementById('label-current').textContent = pack.labels[i];

    rangeEl.value = String(i);


    // Stats

    const s = stats[i] || null;

    if(s){

      document.getElementById('st-week').textContent = s.label;

      document.getElementById('st-sum').textContent  = s.sum_true.toFixed(1);

      document.getElementById('st-mean').textContent = s.mean_pred.toFixed(3);

      document.getElementById('st-rmse').textContent = s.rmse.toFixed(3);

      const top = (s.top5||[]).map(o=>o.geocode + ':' + o.y.toFixed(1)).join('  ·  ');

      document.getElementById('st-top').textContent = top || '—';

    }

    busy = false;

  }


  function rafLoop(ts){

    if(!playing){ lastTs = ts; return; }

    if(!lastTs) lastTs = ts;

    accum += (ts - lastTs);

    lastTs = ts;


    const stepMs = msPerWeek(); // mốc thời gian cho mỗi tuần

    if(accum >= stepMs){

      accum = 0;

      applyWeek(iNow);

      if(iNow < n-1) iNow++; else { playing=false; setToggleUI(); }

    }

    window.requestAnimationFrame(rafLoop);

  }


  btnTog.addEventListener('click', ()=>{

    playing = !playing;

    setToggleUI();

    if(playing){ lastTs = 0; accum = 0; window.requestAnimationFrame(rafLoop); }

  });

  btnRes.addEventListener('click', ()=>{ playing=false; setToggleUI(); iNow=0; applyWeek(0); });


  spdWEl.addEventListener('change', ()=>{ /* áp dụng ngay vòng kế */ });

  spdSEl.addEventListener('change', ()=>{ /* áp dụng ngay vòng kế */ });


  rangeEl.addEventListener('input', (e)=>{

    playing = false; setToggleUI();

    const v = Math.max(0, Math.min(n-1, parseInt(e.target.value||'0',10)));

    iNow = v; applyWeek(v);

  });


  // Panel kéo-thả

  (function(){

    const panel = document.getElementById('settings-panel');

    const header= document.getElementById('settings-header');

    let sx=0, sy=0, dx=0, dy=0, dragging=false;

    header.addEventListener('mousedown', (e)=>{ dragging=true; sx=e.clientX; sy=e.clientY; const r=panel.getBoundingClientRect(); dx=r.left; dy=r.top; e.preventDefault(); });

    window.addEventListener('mousemove', (e)=>{ if(!dragging) return; panel.style.left=(dx+e.clientX-sx)+'px'; panel.style.top=(dy+e.clientY-sy)+'px'; panel.style.right='auto'; });

    window.addEventListener('mouseup', ()=> dragging=false);

  })();


  // Tối ưu Colorbar: ẩn/hiện theo lớp, dàn đều theo số lớp bật (1..3), tự chỉnh margin phải

  async function repositionColorbars(){

    const plot = document.getElementById(plotId);

    if(!plot || !plot.data) return;

    // các trace index của choropleths có colorbar

    const idxs = [1,2,3];

    // lớp nào đang visible === true (không phải 'legendonly')

    const vis = idxs.filter(i => (plot.data[i] && plot.data[i].visible!==false && plot.data[i].visible!=='legendonly'));

    const m = vis.length;


    // x-positions hợp lý theo m

    let xs = [];

    if(m === 1) xs = [1.08];

    else if(m === 2) xs = [1.04, 1.14];

    else if(m >= 3) xs = [1.02, 1.10, 1.18];


    // set showscale true/false theo visible, và đặt x theo thứ tự

    let k = 0;

    for (const i of idxs){

      const isOn = vis.includes(i);

      await Plotly.restyle(plotId, {'showscale': isOn}, [i]);

      if(isOn){

        await Plotly.restyle(plotId, {'colorbar.x': xs[k]}, [i]);

        k += 1;

      }

    }

    // margin phải theo số colorbar

    const r = 40 + m*80; // 1:120, 2:200, 3:280

    await Plotly.relayout(plotId, {'margin.r': r});

  }


  // Toggle lớp: set visible + showscale (đối với trace 1..3), sau đó bố trí lại colorbar

  function toggleLayer(idx, checked){

    const plot = document.getElementById(plotId);

    const traces = (plot && plot.data) ? plot.data.length : 0;

    if(idx >= traces) return;

    const vis = checked ? true : 'legendonly';

    Plotly.restyle(plotId, {'visible': vis}, [idx]).then(()=>{

      if([1,2,3].includes(idx)) repositionColorbars();

    });

  }

  document.querySelectorAll('#settings-panel .layer-toggle').forEach(cb=>{

    cb.addEventListener('change', (e)=> toggleLayer(parseInt(e.target.dataset.index), e.target.checked));

  });


  // Palettes live

  document.getElementById('palette-num').addEventListener('change', (e)=>{

    const p = e.target.value || 'Turbo';

    Plotly.restyle(plotId, {'colorscale': p}, [1,2]);   // Truth + Pred

    Plotly.relayout(plotId, {'coloraxis2.colorscale': p}); // Density

  });

  document.getElementById('palette-div').addEventListener('change', (e)=>{

    const p = e.target.value || 'RdBu';

    Plotly.restyle(plotId, {'colorscale': p}, [3]);     // Residual

  });


  // Init frame 0 + bố trí colorbar lần đầu

  applyWeek(0).then(()=> repositionColorbars());

  setToggleUI();

})();

</script>

"""


def write_html(fig: go.Figure,

               out_path: pathlib.Path,

               datapack: Dict,

               gmax: Dict[str,float],

               stats: List[Dict]):

    body = pio.to_html(fig, full_html=False, include_plotlyjs=True, config={"displaylogo": False}, div_id="plot")

    pack_json = json.dumps(datapack, ensure_ascii=False)

    stats_json= json.dumps(stats, ensure_ascii=False)


    html = f"""<!doctype html>

<html lang="vi">

<head>

<meta charset="utf-8" />

<meta name="viewport" content="width=device-width, initial-scale=1"/>

<title>Dengue Dashboard (Offline)</title>

<style>

  html, body {{ height:100%; width:100%; margin:0; background:#fafafa; }}

  #wrap {{ position:relative; height:100%; width:100%; }}

  #plot {{ position:absolute; inset:0 0 0 0; }}

</style>

</head>

<body>

<div id="wrap">

  {SETTINGS_HTML}

  {body}

  {UI_HTML}

</div>

<script id="week-pack"  type="application/json">{pack_json}</script>

<script id="week-stats" type="application/json">{stats_json}</script>

<script id="gmax-true"  type="application/json">{gmax['true']}</script>

<script id="gmax-pred"  type="application/json">{gmax['pred']}</script>

<script id="gmax-res"   type="application/json">{gmax['res']}</script>

{JS}

</body>

</html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(html, encoding="utf-8")


# ======================

# MAIN

# ======================

def parse_args():

    p = argparse.ArgumentParser(description="Export offline dashboard (streaming restyle + RAF + speed Nweeks/Sseconds)")

    p.add_argument("--max-edges", type=int, default=DEFAULT_MAX_EDGES)

    p.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_NODES)

    p.add_argument("--out", type=str, default=str(DEFAULT_OUT))

    p.add_argument("--start-week", type=str, default=DEFAULT_START, help="YYYY_WW (vd: 2021_1)")

    p.add_argument("--end-week", type=str, default=DEFAULT_END, help="YYYY_WW")

    p.add_argument("--no-hover", action="store_true", default=DEFAULT_NO_HOVER)

    return p.parse_args()


def main():

    args = parse_args()

    print("=== Export advanced visuals (offline, streaming + RAF + Nweeks/Sseconds) ===")


    meta = load_meta_from_environ()          # biome, name_muni, koppen

    centroids = load_centroids()

    edges_df = load_edges(args.max_edges, centroids)


    weeks_all = list_weeks(PANEL_DS_DIR)

    if not weeks_all:

        raise RuntimeError("Không tìm thấy tuần trong panel_node_ds.")


    def parse_w(s):

        if not s: return None

        y, w = s.split("_")

        return (int(y), int(w))


    sw, ew = parse_w(args.start_week), parse_w(args.end_week)

    weeks: List[Tuple[int,int,str]] = []

    for y, w, lbl in weeks_all:

        ok = True

        if sw and (y < sw[0] or (y==sw[0] and (w%1000) < (sw[1]%1000))): ok = False

        if ew and (y > ew[0] or (y==ew[0] and (w%1000) > (ew[1]%1000))): ok = False

        if ok: weeks.append((y, w, lbl))

    if not weeks:

        weeks = weeks_all


    print(f"[WEEKS] {len(weeks)} (từ {weeks[0][2]} → {weeks[-1][2]})")

    print(f"[EDGE] {0 if edges_df is None else len(edges_df):,} edges")


    datapack, gmax, stats = build_datapack(weeks, meta, centroids, args.max_nodes, args.no_hover)


    fig = build_figure(edges_df, datapack.get("biome", {}))


    out_path = pathlib.Path(args.out)

    write_html(fig, out_path, datapack, gmax, stats)

    print(f"✅ DONE: {out_path}")


if __name__ == "__main__":

    main() 