#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse training logs like:
[YYYY-MM-DD HH:MM:SS,ms] [INFO] Epoch: [E][i/I]  Time x (x)  Loss x (x) ...
[YYYY-MM-DD HH:MM:SS,ms] [INFO] giou: ..., ciou: ..., dice: ...

Usage:
  python parse_log.py /path/to/train.log --out_csv parsed_iters.csv --out_summary_csv summary.csv

It outputs:
- per-iteration table (one row per "Epoch:" line)
- per-epoch summary table (last iter row per epoch + (optional) validation metrics)
"""
import os
import argparse, re, pandas as pd, pathlib

ITER_PAT = re.compile(
    r"\[([0-9:\- ,]+)\]\s+\[INFO\]\s+Epoch:\s+\[(\d+)\]\[\s*(\d+)\s*/\s*(\d+)\]\s+"
    r"Time\s+([0-9.]+)\s+\(\s*([0-9.]+)\)\s+"
    r"Loss\s+([0-9.]+)\s+\(\s*([0-9.]+)\)\s+"
    r"CeLoss\s+([0-9.]+)\s+\(\s*([0-9.]+)\)\s+"
    r"MaskLoss\s+([0-9.]+)\s+\(\s*([0-9.]+)\)\s+"
    r"MaskBCELoss\s+([0-9.]+)\s+\(\s*([0-9.]+)\)\s+"
    r"MaskDICELoss\s+([0-9.]+)\s+\(\s*([0-9.]+)\)"
    r"(?:\s+.*)?$"   # 允许后面还有其它字段
)

VAL_PAT = re.compile(
    r"\[([0-9:\- ,]+)\]\s+\[INFO\]\s+giou:\s*([0-9.]+),\s*ciou:\s*([0-9.]+),\s*dice:\s*([0-9.]+)\s*$"
)

def parse_lines(lines):
    rows = []
    val_rows = []
    for line in lines:
        line_norm = line.replace("\t", " ")
        m = ITER_PAT.search(line_norm)
        if m:
            ts, epoch, it, itmax = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
            rows.append({
                "timestamp": ts,
                "epoch": epoch,
                "iter": it,
                "iter_max": itmax,
                "time": float(m.group(5)),
                "time_avg": float(m.group(6)),
                "loss": float(m.group(7)),
                "loss_avg": float(m.group(8)),
                "celoss": float(m.group(9)),
                "celoss_avg": float(m.group(10)),
                "maskloss": float(m.group(11)),
                "maskloss_avg": float(m.group(12)),
                "mask_bce": float(m.group(13)),
                "mask_bce_avg": float(m.group(14)),
                "mask_dice": float(m.group(15)),
                "mask_dice_avg": float(m.group(16)),
            })
            continue
        mv = VAL_PAT.search(line_norm)
        if mv:
            val_rows.append({
                "timestamp": mv.group(1),
                "giou": float(mv.group(2)),
                "ciou": float(mv.group(3)),
                "dice": float(mv.group(4)),
            })
    return pd.DataFrame(rows), pd.DataFrame(val_rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log_path",
        type=str,
        default="./runs/sira/train.log",
        help="path to log file",
    )
    ap.add_argument("--out_csv", type=str, default="iters.csv")
    ap.add_argument("--out_summary_csv", type=str, default="summary.csv")
    args = ap.parse_args()

    log_path = pathlib.Path(args.log_path)
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    df_iters, df_val = parse_lines(lines)
    if df_iters.empty:
        raise SystemExit("No iteration lines matched. Check the regex or log format.")

    # Per-epoch summary: take the last iter row per epoch
    summary = df_iters.sort_values(["epoch", "iter"]).groupby("epoch").tail(1).copy()
    summary = summary[["epoch","iter","iter_max","loss_avg","celoss_avg","maskloss_avg","mask_bce_avg","mask_dice_avg","time_avg"]]

    # Attach validation metrics to the closest *previous* epoch (simple heuristic):
    # If your logs print validation once per epoch, this will align well.
    if not df_val.empty:
        # if multiple val lines, we just map them in order to epoch order
        epochs = sorted(summary["epoch"].tolist())
        for k in ["giou","ciou","dice"]:
            summary[k] = float("nan")
        for idx, (_, vrow) in enumerate(df_val.iterrows()):
            if idx < len(epochs):
                summary.loc[summary["epoch"]==epochs[idx], ["giou","ciou","dice"]] = [vrow["giou"], vrow["ciou"], vrow["dice"]]

    df_iters.to_csv(os.path.join(os.path.dirname(args.log_path), f"epoch{len(epochs)-1}_{args.out_csv}"), index=False)
    summary.to_csv(os.path.join(os.path.dirname(args.log_path), f"epoch{len(epochs)-1}_{args.out_summary_csv}"), index=False)

    print(f"[OK] wrote per-iteration: epoch{len(epochs)-1}_{args.out_csv}  (rows={len(df_iters)})")
    print(f"[OK] wrote per-epoch summary: epoch{len(epochs)-1}_{args.out_summary_csv}  (epochs={len(summary)})")
if __name__ == "__main__":
    main()
