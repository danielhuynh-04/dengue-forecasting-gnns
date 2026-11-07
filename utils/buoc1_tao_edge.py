# utils/buoc1_tao_edge.py
import os
import json
import pandas as pd
import torch
import geopandas as gpd
import numpy as np
from sklearn.neighbors import NearestNeighbors

RAW_DIR = "data/raw"
OUT_INTERIM = "data/interim"
OUT_PROCESSED = "data/processed"

os.makedirs(OUT_INTERIM, exist_ok=True)
os.makedirs(OUT_PROCESSED, exist_ok=True)

# 1) Đọc GeoJSON (geojs-100-mun.json)
geo_path = os.path.join(RAW_DIR, "geojs-100-mun.json")
gdf = gpd.read_file(geo_path)

# Cột mã là 'id' (7 chữ số)
if "id" not in gdf.columns:
    raise KeyError("GeoJSON không có cột 'id'. Hãy kiểm tra lại tên cột mã đô thị.")

# Chuẩn hoá id -> chuỗi 7 kí tự
gdf["id"] = gdf["id"].astype(str).str.zfill(7)

# 2) Tạo node list & mapping
all_nodes = sorted(gdf["id"].unique())
node2idx = {gid: i for i, gid in enumerate(all_nodes)}
with open(os.path.join(OUT_PROCESSED, "node2idx.json"), "w", encoding="utf-8") as f:
    json.dump(node2idx, f, ensure_ascii=False)

# 3) Adjacency theo contiguity (Queen)
gdf = gdf[["id", "geometry"]].reset_index(drop=True)
gdf = gdf.set_geometry("geometry")
sindex = gdf.sindex

edges = set()
for i, row in gdf.iterrows():
    geom = row.geometry
    if geom is None or geom.is_empty:
        continue
    cand_idx = list(sindex.intersection(geom.bounds))
    for j in cand_idx:
        if i == j:
            continue
        geom2 = gdf.at[j, "geometry"]
        if geom2 is None or geom2.is_empty:
            continue
        # Queen contiguity: dùng intersects để bao quát tiếp xúc biên/điểm
        if geom.intersects(geom2):
            u = node2idx[gdf.at[i, "id"]]
            v = node2idx[gdf.at[j, "id"]]
            if u != v:
                a, b = (u, v) if u < v else (v, u)
                edges.add((a, b))

# 4) Bổ sung KNN cho node cô lập (đảm bảo liên thông tốt hơn)
centroids = gdf.geometry.centroid
coords = np.array([[pt.x, pt.y] for pt in centroids])
K = 6
nbrs = NearestNeighbors(n_neighbors=min(K+1, len(coords))).fit(coords)
dist, idxs = nbrs.kneighbors(coords)

deg = np.zeros(len(all_nodes), dtype=int)
for u, v in edges:
    deg[u] += 1
    deg[v] += 1

for i in range(len(all_nodes)):
    if deg[i] == 0:
        for j in idxs[i][1:]:  # bỏ chính nó
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in edges:
                edges.add((a, b))
                deg[a] += 1
                deg[b] += 1

# 5) Lưu kết quả
edges_df = pd.DataFrame(list(edges), columns=["src", "dst"]).sort_values(["src", "dst"]).reset_index(drop=True)
edges_df.to_csv(os.path.join(OUT_INTERIM, "edge_list.csv"), index=False)

edge_index = torch.tensor(edges_df[["src", "dst"]].values.T, dtype=torch.long)
torch.save(edge_index, os.path.join(OUT_PROCESSED, "edge_index.pt"))

print(f"✅ Done. Nodes: {len(all_nodes)} | Edges: {edge_index.shape[1]}")
print(f"- node2idx.json @ {os.path.join(OUT_PROCESSED, 'node2idx.json')}")
print(f"- edge_list.csv @ {os.path.join(OUT_INTERIM, 'edge_list.csv')}")
print(f"- edge_index.pt @ {os.path.join(OUT_PROCESSED, 'edge_index.pt')}")
