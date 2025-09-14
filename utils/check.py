# utils/check_artifacts.py
import os
import json
import glob
import numpy as np
import pandas as pd
import torch

RAW_DIR = "data/raw"
INTERIM_DIR = "data/interim"
PROCESSED_DIR = "data/processed"
PT_DIR = os.path.join(PROCESSED_DIR, "weekly_pt")

def green(msg): print(f"\033[92m{msg}\033[0m")
def yellow(msg): print(f"\033[93m{msg}\033[0m")
def red(msg): print(f"\033[91m{msg}\033[0m")

def main():
    ok = True

    # 1) Kiểm tra node2idx, edge_index, và list snapshots
    node2idx_path = os.path.join(PROCESSED_DIR, "node2idx.json")
    edge_index_path = os.path.join(PROCESSED_DIR, "edge_index.pt")

    if not os.path.exists(node2idx_path):
        red("❌ Missing processed/node2idx.json")
        return
    if not os.path.exists(edge_index_path):
        red("❌ Missing processed/edge_index.pt")
        return
    if not os.path.isdir(PT_DIR):
        red("❌ Missing processed/weekly_pt/")
        return

    with open(node2idx_path, "r", encoding="utf-8") as f:
        node2idx = json.load(f)
    idx2node = [k for k, v in sorted(node2idx.items(), key=lambda kv: kv[1])]
    N = len(idx2node)
    green(f"✅ node2idx loaded: N={N}")

    # Edge index
    try:
        edge_index = torch.load(edge_index_path, map_location="cpu", weights_only=True)
    except TypeError:
        # weights_only not available in older torch -> fallback
        edge_index = torch.load(edge_index_path, map_location="cpu")
        yellow("⚠️ Using torch.load without weights_only (older torch version).")
    ei_ok = isinstance(edge_index, torch.Tensor) and edge_index.ndim == 2 and edge_index.shape[0] == 2
    print(f"edge_index shape: {tuple(edge_index.shape)}")
    if not ei_ok:
        red("❌ edge_index is not a 2xE tensor")
        ok = False
    else:
        green("✅ edge_index tensor OK")

    # Snapshots
    pts = sorted(glob.glob(os.path.join(PT_DIR, "*.pt")))
    print(f"Found {len(pts)} snapshots in {PT_DIR}")
    if len(pts) == 0:
        red("❌ No snapshot .pt files found.")
        return

    # 2) Kiểm tra CSV full & yearly
    full_csv = os.path.join(INTERIM_DIR, "weekly_features_labels.csv")
    if os.path.exists(full_csv):
        df = pd.read_csv(full_csv)
        green(f"✅ CSV full exists: {full_csv} with shape {df.shape}")
        required_cols = ["geocode","year","epiweek","casos","temp_med","precip_tot","rainy_days","rel_humid_med"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            yellow(f"⚠️ CSV missing columns (not critical if intentionally dropped): {missing}")
        # No NaN check on core columns
        core = [c for c in required_cols if c in df.columns]
        if core:
            null_counts = df[core].isna().sum()
            if null_counts.sum() > 0:
                yellow(f"⚠️ NaNs in core columns:\n{null_counts[null_counts>0]}")
            else:
                green("✅ No NaNs in core columns")
        # Week range sanity
        year_min, year_max = df["year"].min(), df["year"].max()
        week_min, week_max = df["epiweek"].min(), df["epiweek"].max()
        print(f"Year range: {year_min}..{year_max}, epiweek range: {week_min}..{week_max}")
    else:
        red("❌ Missing CSV full: data/interim/weekly_features_labels.csv")
        ok = False

    yearly_dir = os.path.join(INTERIM_DIR, "yearly")
    if os.path.isdir(yearly_dir):
        ycsvs = sorted(glob.glob(os.path.join(yearly_dir, "weekly_*.csv")))
        print(f"Yearly CSV files: {len(ycsvs)}")
        if len(ycsvs) == 0:
            yellow("⚠️ No per-year CSVs found (maybe no data years?)")
        else:
            green("✅ Per-year CSVs present.")
    else:
        yellow("⚠️ data/interim/yearly/ folder not found")

    # 3) Mở ngẫu nhiên 2 snapshot để kiểm tra cấu trúc
    sample = pts[::max(1, len(pts)//2)][:2]  # 2 samples spread across
    for p in sample:
        d = torch.load(p, map_location="cpu")
        print(f"\nInspect: {os.path.basename(p)}")
        # Required keys
        for k in ["x","y","edge_index","year","epiweek","feature_cols","label_col","geocodes","train_mask","val_mask","test_mask"]:
            if k not in d:
                red(f"❌ Missing key in snapshot: {k}")
                ok = False

        x, y = d["x"], d["y"]
        fcols = d.get("feature_cols", [])
        geocodes = d.get("geocodes", [])
        tm, vm, sm = d["train_mask"], d["val_mask"], d["test_mask"]

        # Shapes
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[0] != N:
            red(f"❌ x shape mismatch. Expect ({N}, F), got {tuple(x.shape)}")
            ok = False
        else:
            green(f"✅ x shape OK: {tuple(x.shape)} with {len(fcols)} features")

        if not isinstance(y, torch.Tensor) or y.ndim != 1 or y.shape[0] != N:
            red(f"❌ y shape mismatch. Expect ({N},), got {tuple(y.shape)}")
            ok = False
        else:
            green("✅ y shape OK")

        # Geocodes alignment
        if len(geocodes) != N or geocodes != [k for k, v in sorted(node2idx.items(), key=lambda kv: kv[1])]:
            yellow("⚠️ geocodes list != idx2node (order/length mismatch). Check Bước 1 output.")
        else:
            green("✅ geocodes aligned with node2idx")

        # Masks
        for name, m in [("train_mask", tm), ("val_mask", vm), ("test_mask", sm)]:
            if not isinstance(m, torch.Tensor) or m.dtype != torch.bool or m.numel() != N:
                red(f"❌ {name} invalid: dtype/bool/size mismatch")
                ok = False
        if tm.any() or vm.any() or sm.any():
            if not ((tm & vm).any() or (tm & sm).any() or (vm & sm).any()):
                green("✅ masks look mutually exclusive")
            else:
                yellow("⚠️ some masks overlap")

        # NaN check in x/y
        if torch.isnan(x).any() or torch.isinf(x).any():
            red("❌ x contains NaN/Inf")
            ok = False
        else:
            green("✅ x has no NaN/Inf")
        if torch.isnan(y).any() or torch.isinf(y).any():
            red("❌ y contains NaN/Inf")
            ok = False
        else:
            green("✅ y has no NaN/Inf")

        # Feature names sanity
        expected_subset = set([
            'temp_med','precip_tot','rainy_days','rel_humid_med',
            'POPULACAO','altitude','area_km2','population_density',
            'incidence_per_1k','humid_heat_index','precip_std_roll3',
            'dry_spell_len','neighbor_cases_prev1','incidence_lag1'
        ])
        missing = [c for c in expected_subset if c not in fcols]
        if missing:
            yellow(f"⚠️ Some expected features not in snapshot (maybe absent in data): {missing}")
        else:
            green("✅ feature_cols include the expected set (14 cols)")

    # 4) Thống kê nhanh để nhìn sanity
    try:
        df = pd.read_csv(full_csv)
        print("\nSanity stats (from full CSV):")
        for col in ["casos","incidence_per_1k","population_density","humid_heat_index","neighbor_cases_prev1"]:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                print(f"  {col:22s}: min={s.min():.3f}  p50={s.median():.3f}  max={s.max():.3f}  NaNs={s.isna().sum()}")
    except Exception as e:
        yellow(f"⚠️ Could not compute sanity stats: {e}")

    if ok:
        green("\n🎉 All checks passed (or acceptable warnings only).")
    else:
        red("\n❗ Some checks failed. See messages above.")
    # ví dụ trong main(), ngay sau đọc pts
    sample_path = pts[0]
    d = torch.load(sample_path, map_location="cpu")
    print("Sample feature_cols:", d["feature_cols"])
    print("x shape:", d["x"].shape, "y shape:", d["y"].shape, "year/week:", d["year"], d["epiweek"])

    
    
if __name__ == "__main__":
    main()
