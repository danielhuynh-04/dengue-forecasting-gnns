# run_global_gat.py
# ------------------------------------------------------------
# Global GAT theo tuần (CPU), HuberLoss, early stopping,
# metric đầy đủ (log/real, R2 trim), AUC/PR từ hồi quy.
# Bias-correction: Duan smearing (ưu tiên) + fallback sigma^2.
# Thêm clamp theo phân vị cao của TRAIN để giảm outlier ở miền thực.
# ------------------------------------------------------------
from __future__ import annotations
import os, glob, time, json, random
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.trainer_weekly import build_temporal_seq, make_model, _find_lag_columns
from evaluation.metrics import evaluate_regression, trimmed_r2, classification_metrics_from_regression


# ---------------------- cấu hình ----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SNAP_DIR = "data/processed/weekly_pt_scaled"
EDGE_PATH = "data/processed/edge_index.pt"

REPORT_DIR = "data/interim"
CKPT_DIR = "checkpoints"
VIS_DIR = "visualizations"
VIS_DATA_DIR = os.path.join(VIS_DIR, "data")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(VIS_DATA_DIR, exist_ok=True)

WRITE_PREDICTIONS = True
PREDICTIONS_CSV = os.path.join(VIS_DATA_DIR, "node_predictions.csv")


# ---------------------- utils ----------------------
def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _safe_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_edge(path: str) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def year_from_file(path: str) -> int:
    base = os.path.basename(path)
    return int(base.split("_")[0])


def list_snapshots() -> List[str]:
    paths = sorted(glob.glob(os.path.join(SNAP_DIR, "*.pt")))
    if not paths:
        raise FileNotFoundError(f"Không tìm thấy *.pt trong {SNAP_DIR}")
    return paths


def split_weeks(paths: List[str]) -> Tuple[List[str], List[str], List[str]]:
    train, val, test = [], [], []
    for p in paths:
        y = year_from_file(p)
        if 2010 <= y <= 2020:   train.append(p)
        elif y in (2021, 2022): val.append(p)
        elif y in (2023, 2024): test.append(p)
    return train, val, test


# ---------------------- backtransform helpers ----------------------
@torch.no_grad()
def backtransform_smear(y_log: torch.Tensor, smear: float | None) -> torch.Tensor:
    """
    Duan's smearing: Y ≈ exp(mu) * S - 1, với S = E[exp(eps)] ≥ 1.
    Nếu smear None -> rơi về expm1(mu).
    """
    if smear is None or smear <= 0:
        return torch.expm1(y_log).clamp_min(0.0)
    return (torch.exp(y_log) * float(smear) - 1.0).clamp_min(0.0)


@torch.no_grad()
def backtransform_sigma(y_log: torch.Tensor, sigma2: float = 0.0) -> torch.Tensor:
    """
    Fallback log-normal: Y ≈ exp(mu + 0.5*sigma^2) - 1
    """
    shift = 0.5 * float(max(0.0, sigma2))
    return torch.expm1(y_log + shift).clamp_min(0.0)


@torch.no_grad()
def apply_headroom_clamp(y_real: np.ndarray, cap: float | None) -> np.ndarray:
    """
    Chặn đỉnh dự báo để giảm tác động outlier cực trị (cap học từ TRAIN).
    """
    if cap is None or cap <= 0:
        return y_real
    return np.clip(y_real, 0.0, float(cap))


# ---------------------- metrics block ----------------------
def _metrics_block(y_true_log_t: torch.Tensor,
                   y_pred_log_t: torch.Tensor,
                   smear: float | None,
                   sigma2: float | None,
                   cap: float | None) -> Dict[str, float]:
    # log
    mlog = evaluate_regression(y_pred_log_t, y_true_log_t)
    r2t_log = trimmed_r2(y_true_log_t.numpy(), y_pred_log_t.numpy(), trim=0.01)

    # real (smear ưu tiên, sigma fallback) + clamp
    yt_r = torch.expm1(y_true_log_t).clamp_min(0).numpy()
    if smear is not None and smear > 0:
        yp_r = backtransform_smear(y_pred_log_t, smear).cpu().numpy()
    else:
        yp_r = backtransform_sigma(y_pred_log_t, sigma2 or 0.0).cpu().numpy()

    yp_r = apply_headroom_clamp(yp_r, cap)

    mreal = evaluate_regression(yp_r, yt_r)
    r2t_real = trimmed_r2(yt_r, yp_r, trim=0.01)

    return {
        "MAE_log": mlog["MAE"],
        "RMSE_log": mlog["RMSE"],
        "R2_log": mlog["R2"],
        "R2trim_log": r2t_log,
        "MAE_real": mreal["MAE"],
        "RMSE_real": mreal["RMSE"],
        "SMAPE_real": mreal["SMAPE"],
        "R2_real": mreal["R2"],
        "R2trim_real": r2t_real,
    }


# ---------------------- epoch loop ----------------------
def run_epoch(model,
              edge_index: torch.Tensor,
              paths: List[str],
              phase: str = "train",
              opt: torch.optim.Optimizer | None = None,
              loss_fn: nn.Module | None = None,
              smear: float | None = None,
              sigma2: float | None = None,
              cap: float | None = None) -> tuple[float, list[dict], dict | None]:
    is_train = (phase == "train")
    model.train(is_train)

    total_loss, n_steps = 0.0, 0
    weekly_rows: list[dict] = []
    mic_true_log, mic_pred_log = [], []

    for p in paths:
        d = _safe_load(p)
        x = d["x"].to(DEVICE); y = d["y"].to(DEVICE)

        mask = d.get(f"{phase}_mask", d.get(f"mask_{phase}", None))
        if mask is None: raise KeyError(f"Thiếu {phase}_mask trong {p}")
        mask = mask.to(DEVICE)
        if mask.sum().item() == 0:
            continue

        feature_cols = d.get("feature_cols", [])
        temporal_seq, _ = build_temporal_seq(x, feature_cols)

        with torch.set_grad_enabled(is_train):
            y_hat = model(x, edge_index, temporal_seq=temporal_seq)
            loss_t = (loss_fn(y_hat[mask], y[mask]) if loss_fn is not None
                      else F.mse_loss(y_hat[mask], y[mask]))
            if is_train:
                opt.zero_grad()
                loss_t.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        total_loss += float(loss_t.item()); n_steps += 1

        if phase in ("val", "test"):
            base = os.path.basename(p).replace(".pt", "")
            yr_s, ew_s = base.split("_")
            wk_m = _metrics_block(y[mask].detach().cpu(),
                                  y_hat[mask].detach().cpu(),
                                  smear=smear, sigma2=sigma2, cap=cap)
            weekly_rows.append({"Year": int(yr_s), "Epiweek": int(ew_s), "Split": phase, **wk_m})
            mic_true_log.append(y[mask].detach().cpu())
            mic_pred_log.append(y_hat[mask].detach().cpu())

        # Ghi dự báo node-level cho trực quan hoá
        if WRITE_PREDICTIONS and (phase in ("val", "test")) and ("geocodes" in d):
            with torch.no_grad():
                base = os.path.basename(p).replace(".pt", "")
                yr_s, ew_s = base.split("_")
                yt_l = y.detach().cpu(); yp_l = model(x, edge_index, temporal_seq=temporal_seq).detach().cpu()
                yt_r = torch.expm1(yt_l).clamp_min(0).numpy()
                if smear is not None and smear > 0:
                    yp_r = backtransform_smear(yp_l, smear).numpy()
                else:
                    yp_r = backtransform_sigma(yp_l, sigma2 or 0.0).numpy()
                yp_r = apply_headroom_clamp(yp_r, cap)

                idx = mask.cpu().numpy().astype(bool)
                geos = d["geocodes"]
                out_df = pd.DataFrame({
                    "Year": int(yr_s),
                    "Epiweek": int(ew_s),
                    "Split": phase,
                    "geocode": np.asarray(geos)[idx],
                    "y_true_log": yt_l.numpy()[idx],
                    "y_pred_log": yp_l.numpy()[idx],
                    "y_true": yt_r[idx],
                    "y_pred": yp_r[idx],
                })
                header = not os.path.exists(PREDICTIONS_CSV)
                out_df.to_csv(PREDICTIONS_CSV, mode="a", header=header, index=False, encoding="utf-8")

    avg_loss = total_loss / max(1, n_steps)

    micro = None
    if mic_true_log:
        yt = torch.cat(mic_true_log, 0); yp = torch.cat(mic_pred_log, 0)
        mic_log = evaluate_regression(yp, yt)
        mic_r2t_log = trimmed_r2(yt.numpy(), yp.numpy(), trim=0.01)
        yt_r = torch.expm1(yt).clamp_min(0).numpy()
        if smear is not None and smear > 0:
            yp_r = backtransform_smear(yp, smear).numpy()
        else:
            yp_r = backtransform_sigma(yp, sigma2 or 0.0).numpy()
        yp_r = apply_headroom_clamp(yp_r, cap)
        mic_real = evaluate_regression(yp_r, yt_r)
        mic_r2t_real = trimmed_r2(yt_r, yp_r, trim=0.01)

        micro = {
            "micro_MAE_log": mic_log["MAE"],
            "micro_RMSE_log": mic_log["RMSE"],
            "micro_R2_log": mic_log["R2"],
            "micro_R2trim_log": mic_r2t_log,
            "micro_MAE_real": mic_real["MAE"],
            "micro_RMSE_real": mic_real["RMSE"],
            "micro_SMAPE_real": mic_real["SMAPE"],
            "micro_R2_real": mic_real["R2"],
            "micro_R2trim_real": mic_r2t_real,
        }

    return avg_loss, weekly_rows, micro


# ---------------------- ước lượng thống kê TRAIN ----------------------
@torch.no_grad()
def estimate_backtransform_stats_on_train(model, edge_index, train_paths) -> dict:
    """
    Trả về:
      - smear: Duan smearing factor S = mean(exp(residual))
      - sigma2: var(residual) trên miền log (fallback)
      - cap: phân vị cao (q=0.999) của Y_train thực để chặn đỉnh dự báo
    """
    resids = []
    y_train_real = []

    model.eval()
    for p in train_paths:
        d = _safe_load(p)
        x = d["x"].to(DEVICE); y = d["y"].to(DEVICE)
        mask = d.get("train_mask", d.get("mask_train", None))
        if mask is None or mask.sum().item() == 0:
            continue
        feature_cols = d.get("feature_cols", [])
        temporal_seq, _ = build_temporal_seq(x, feature_cols)
        y_hat = model(x, edge_index, temporal_seq=temporal_seq)
        r = (y[mask] - y_hat[mask]).detach().cpu().numpy()
        if r.size > 0:
            resids.append(r)
        # gom Y thực để ước lượng cap
        y_train_real.append(torch.expm1(y[mask]).clamp_min(0).cpu().numpy())

    if not resids:
        return {"smear": None, "sigma2": 0.0, "cap": None}

    r = np.concatenate(resids, axis=0)
    sigma2 = float(max(0.0, np.var(r, ddof=1)))
    smear = float(np.mean(np.exp(r)))  # >= 1 theo lý thuyết
    cap = None
    if y_train_real:
        ytr = np.concatenate(y_train_real, axis=0)
        cap = float(np.quantile(ytr, 0.999))  # chặn 99.9% giá trị train

    return {"smear": smear, "sigma2": sigma2, "cap": cap}


# ---------------------- main ----------------------
def main():
    seed_all(42)
    paths = list_snapshots()
    tr_paths, va_paths, te_paths = split_weeks(paths)

    sample = _safe_load(tr_paths[0])
    in_dim = int(sample["x"].shape[1])
    has_temporal = bool(_find_lag_columns(sample.get("feature_cols", [])))
    edge_index = _load_edge(EDGE_PATH).to(DEVICE)

    model = make_model(in_dim=in_dim, has_temporal=has_temporal,
                       hidden=128, heads=(4, 4), gat_dropout=0.2).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    # Huber hơi “mềm” hơn một chút để đỡ bóp đỉnh thật: delta=1.2
    loss_fn = nn.HuberLoss(delta=1.2)

    patience, bad = 30, 0
    EPOCHS = 200
    best_val = float("inf")
    best_ckpt = os.path.join(CKPT_DIR, "gat_global_best.pt")

    print(f"Global training with {len(tr_paths)} train weeks, {len(va_paths)} val weeks, {len(te_paths)} test weeks")
    print(f"Device: {DEVICE} | Model: in={in_dim}, hidden=128, heads=(4,4), temporal={'ON' if has_temporal else 'OFF'}")

    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        tr_loss, _, _ = run_epoch(model, edge_index, tr_paths, phase="train", opt=opt, loss_fn=loss_fn)
        va_loss, _, _ = run_epoch(model, edge_index, va_paths, phase="val", opt=None, loss_fn=loss_fn)

        if (ep <= 10) or (ep % 10 == 0) or (ep % 10 == 1):
            print(f"Epoch {ep:03d} | Train {tr_loss:.4f} | Val {va_loss:.4f}")

        if va_loss + 1e-9 < best_val:
            best_val = va_loss; bad = 0
            torch.save({"state_dict": model.state_dict(), "in_dim": in_dim, "has_temporal": has_temporal}, best_ckpt)
        else:
            bad += 1
            if bad >= patience:
                print(f"⏹️ Early stopping @ epoch {ep}")
                break

    # load best & ước lượng thống kê TRAIN cho backtransform
    ckpt = _safe_load(best_ckpt)
    model.load_state_dict(ckpt["state_dict"]); model.to(DEVICE); model.eval()
    stats = estimate_backtransform_stats_on_train(model, edge_index, tr_paths)
    smear, sigma2, cap = stats["smear"], stats["sigma2"], stats["cap"]

    # đánh giá chi tiết
    _, val_rows, val_micro = run_epoch(model, edge_index, va_paths, phase="val",
                                       opt=None, loss_fn=loss_fn, smear=smear, sigma2=sigma2, cap=cap)
    _, test_rows, test_micro = run_epoch(model, edge_index, te_paths, phase="test",
                                         opt=None, loss_fn=loss_fn, smear=smear, sigma2=sigma2, cap=cap)

    weekly_rows = val_rows + test_rows
    columns = ["Year","Epiweek","Split","MAE_log","RMSE_log","R2_log","R2trim_log",
               "MAE_real","RMSE_real","SMAPE_real","R2_real","R2trim_real"]
    weekly_df = pd.DataFrame(weekly_rows, columns=columns)
    weekly_csv = os.path.join(REPORT_DIR, "gat_global_weekly_report.csv")
    weekly_df.to_csv(weekly_csv, index=False)

    # macro theo tuần
    def macro_avg(df: pd.DataFrame, split: str) -> dict:
        sub = df[df["Split"] == split]
        if sub.empty: return {}
        keys = ["MAE_log","RMSE_log","R2_log","R2trim_log","MAE_real","RMSE_real","SMAPE_real","R2_real","R2trim_real"]
        return {f"{split}_macro_{k}": float(sub[k].mean()) for k in keys}

    macro_val = macro_avg(weekly_df, "val")
    macro_test = macro_avg(weekly_df, "test")

    # AUC/PR từ hồi quy (q=0.90)
    def cls_metrics(paths, phase: str) -> dict:
        yt_all, yp_all = [], []
        for p in paths:
            d = _safe_load(p)
            x = d["x"].to(DEVICE); y = d["y"].to(DEVICE)
            mask = d.get(f"{phase}_mask", d.get(f"mask_{phase}", None))
            if mask is None or mask.sum().item() == 0: continue
            feature_cols = d.get("feature_cols", [])
            temporal_seq, _ = build_temporal_seq(x, feature_cols)
            yp = model(x, edge_index, temporal_seq=temporal_seq)
            yt_all.append(y[mask].detach().cpu()); yp_all.append(yp[mask].detach().cpu())
        if not yt_all: return {}
        yt = torch.cat(yt_all, 0); yp = torch.cat(yp_all, 0)
        yt_r = torch.expm1(yt).clamp_min(0).numpy()
        if smear is not None and smear > 0:
            yp_r = backtransform_smear(yp, smear).numpy()
        else:
            yp_r = backtransform_sigma(yp, sigma2 or 0.0).numpy()
        yp_r = apply_headroom_clamp(yp_r, cap)
        cls = classification_metrics_from_regression(yt_r, yp_r, pos_threshold=None, q=0.90)
        return {f"{phase}_ROC_AUC": cls["ROC_AUC"], f"{phase}_PR_AUC": cls["PR_AUC"], f"{phase}_pos_rate": cls["pos_rate"]}

    cls_val = cls_metrics(va_paths, "val")
    cls_test = cls_metrics(te_paths, "test")

    summary = {
        "best_val_loss": float(best_val),
        "smear_train": None if smear is None else float(smear),
        "sigma2_train_log": float(sigma2),
        "cap_train_q999": None if cap is None else float(cap),
        **macro_val, **(val_micro or {}), **cls_val,
        **macro_test, **(test_micro or {}), **cls_test,
    }

    with open(os.path.join(REPORT_DIR, "gat_global_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved best checkpoint: {best_ckpt}")
    print(f"✅ Wrote weekly report:  {weekly_csv}")
    if WRITE_PREDICTIONS:
        print(f"✅ Wrote node-level predictions: {PREDICTIONS_CSV}")
    print(f"✅ Summary: {summary}")
    print(f"⏱️ Done in {(time.time()-t0)/60:.2f} min.")


if __name__ == "__main__":
    main()
