# utils/buoc4_check_scaled.py
# ---------------------------
# BƯỚC 4: Kiểm tra snapshot đã scale (weekly_pt_scaled)
# - Đọc node2idx.json, scaler_weekly.json
# - Duyệt toàn bộ *.pt trong data/processed/weekly_pt_scaled
# - Kiểm tra: khóa bắt buộc, shape x/y, NaN/Inf, masks, số feature, geocodes khớp
# - Xuất báo cáo CSV để rà soát nhanh

import os
import json
import glob
import math
import torch
import numpy as np
import pandas as pd

PROCESSED_DIR = "data/processed"
INTERIM_DIR = "data/interim"
PT_DIR = os.path.join(PROCESSED_DIR, "weekly_pt_scaled")
NODE2IDX_PATH = os.path.join(PROCESSED_DIR, "node2idx.json")
SCALER_PATH = os.path.join(PROCESSED_DIR, "scaler_weekly.json")

os.makedirs(INTERIM_DIR, exist_ok=True)

def _safe_load_pt(path):
    # PyTorch khuyến nghị dùng weights_only=True để tránh cảnh báo pickle
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # fallback cho phiên bản torch cũ không có tham số weights_only
        return torch.load(path, map_location="cpu")

def _check_masks(train_mask, val_mask, test_mask):
    tm = train_mask.numpy().astype(bool)
    vm = val_mask.numpy().astype(bool)
    sm = test_mask.numpy().astype(bool)
    # mutually exclusive
    overlap = (tm & vm) | (tm & sm) | (vm & sm)
    return not overlap.any()

def main():
    # 1) Load node2idx
    if not os.path.exists(NODE2IDX_PATH):
        raise FileNotFoundError(f"Missing {NODE2IDX_PATH}")
    with open(NODE2IDX_PATH, "r", encoding="utf-8") as f:
        node2idx = json.load(f)
    idx2node = [k for k, v in sorted(node2idx.items(), key=lambda kv: kv[1])]
    N = len(idx2node)
    print(f"✅ node2idx loaded: N={N}")

    # 2) Load scaler meta
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Missing {SCALER_PATH}")
    with open(SCALER_PATH, "r", encoding="utf-8") as f:
        scaler_meta = json.load(f)
    scaler_feats = scaler_meta.get("feature_cols", [])
    print(f"✅ scaler_weekly.json loaded with {len(scaler_feats)} features")

    # 3) Scan scaled snapshots
    paths = sorted(glob.glob(os.path.join(PT_DIR, "*.pt")))
    if not paths:
        raise FileNotFoundError(f"No snapshots found in {PT_DIR}")
    print(f"Found {len(paths)} snapshots in {PT_DIR}")

    # 4) Iterate & validate
    rows = []
    required_keys = {
        "x","y","edge_index","year","epiweek","feature_cols",
        "label_col","geocodes","train_mask","val_mask","test_mask"
    }

    issues = 0
    for p in paths:
        d = _safe_load_pt(p)

        missing = required_keys - set(d.keys())
        if missing:
            print(f"❌ {os.path.basename(p)} missing keys: {missing}")
            issues += 1
            continue

        x = d["x"]; y = d["y"]
        geocodes = d["geocodes"]
        feat_cols = d["feature_cols"]
        tm, vm, sm = d["train_mask"], d["val_mask"], d["test_mask"]
        year = d["year"]; epiweek = d["epiweek"]

        # shapes
        ok_shape = True
        if x.ndim != 2 or x.shape[0] != N:
            ok_shape = False
            print(f"❌ {os.path.basename(p)}: bad x shape {tuple(x.shape)}")
        if y.ndim != 1 or y.shape[0] != N:
            ok_shape = False
            print(f"❌ {os.path.basename(p)}: bad y shape {tuple(y.shape)}")

        # feature count
        F = x.shape[1] if x.ndim == 2 else -1
        if len(feat_cols) != F:
            ok_shape = False
            print(f"❌ {os.path.basename(p)}: feature_cols({len(feat_cols)}) != x.F({F})")

        # scaler match
        ok_scaler = (feat_cols == scaler_feats)
        if not ok_scaler:
            print(f"⚠️  {os.path.basename(p)}: feature_cols not equal to scaler feature list")

        # geocode alignment
        ok_geo = (len(geocodes) == N and list(geocodes) == idx2node)
        if not ok_geo:
            print(f"❌ {os.path.basename(p)}: geocodes misaligned")

        # masks mutual exclusivity
        ok_masks = _check_masks(tm, vm, sm)
        if not ok_masks:
            print(f"❌ {os.path.basename(p)}: masks overlap")

        # NaN/Inf checks
        x_np = x.numpy()
        y_np = y.numpy()
        nan_x = int(np.isnan(x_np).sum())
        inf_x = int(np.isinf(x_np).sum())
        nan_y = int(np.isnan(y_np).sum())
        inf_y = int(np.isinf(y_np).sum())

        if nan_x or inf_x or nan_y or inf_y:
            print(f"❌ {os.path.basename(p)}: NaN/Inf x({nan_x}/{inf_x}) y({nan_y}/{inf_y})")
            issues += 1

        rows.append({
            "file": os.path.basename(p),
            "year": int(year),
            "epiweek": int(epiweek),
            "N": N,
            "F": F,
            "mask_train": int(tm.sum().item()),
            "mask_val": int(vm.sum().item()),
            "mask_test": int(sm.sum().item()),
            "nan_x": nan_x,
            "inf_x": inf_x,
            "nan_y": nan_y,
            "inf_y": inf_y,
            "ok_shape": ok_shape,
            "ok_scaler": ok_scaler,
            "ok_geo": ok_geo,
            "ok_masks": ok_masks,
        })

    # 5) Report
    rep = pd.DataFrame(rows).sort_values(["year","epiweek"])
    out_csv = os.path.join(INTERIM_DIR, "check_weekly_pt_scaled_report.csv")
    rep.to_csv(out_csv, index=False)
    print(f"✅ Wrote report: {out_csv}")

    # Summary
    bad = rep[~(rep["ok_shape"] & rep["ok_scaler"] & rep["ok_geo"] & rep["ok_masks"])]
    if len(bad) == 0 and rep[["nan_x","inf_x","nan_y","inf_y"]].to_numpy().sum() == 0:
        print("🎉 All scaled snapshots look good.")
    else:
        print(f"⚠️  Found potential issues in {len(bad)} files. See report for details.")

    # Quick glance
    print("\nSanity (first & last rows):")
    print(rep.head(3).to_string(index=False))
    print("...")
    print(rep.tail(3).to_string(index=False))

if __name__ == "__main__":
    main()
