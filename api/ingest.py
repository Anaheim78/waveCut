# pout_auto.py — 全自動：robust 門檻 + 回滯 + 段落擴展（零手調）
# 需 CSV 欄位：time_seconds, height_width_ratio

import os, math, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== 只改這一行就好 =====
CSV_PATH = r"C:\Users\plus1\Downloads\FaceTraining_POUT_LIPS_20250901_031623.csv"
# =========================

PLOT          = True
WRITE_JSON    = False
OUT_JSON_PATH = r"C:\Users\plus1\Downloads\pout_result_auto.json"

def _mad_sigma(x: np.ndarray) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad

def _estimate_fs(t: np.ndarray) -> float:
    if len(t) < 2: return 20.0
    dt = float(np.median(np.diff(t)))
    return 1.0 / dt if dt > 1e-9 else 20.0

def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1: return x.copy()
    ker = np.ones(int(k), dtype=float) / float(k)
    return np.convolve(x, ker, mode="same")

def _exp_smooth(x: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0: return x.copy()
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = (1 - alpha) * y[i-1] + alpha * x[i]
    return y

def _hysteresis_segments(y: np.ndarray, th_hi: float, th_lo: float):
    segs, on, s, n = [], False, None, len(y)
    for i in range(n):
        if not on:
            if y[i] >= th_hi: on, s = True, i
        else:
            if y[i] < th_lo:
                e = i
                segs.append((s, max(e, s+1)))
                on, s = False, None
    if on and s is not None:
        segs.append((s, n))
    return segs

def _pad_merge_cut(segs, fs, n, pad_sec, min_gap_sec, max_seg_sec):
    if not segs: return []
    pad = int(round(pad_sec * fs))
    segs = [(max(0, s-pad), min(n, e+pad)) for (s, e) in segs]

    # merge small gaps
    merged = []
    min_gap = int(round(min_gap_sec * fs))
    for s, e in sorted(segs):
        if not merged:
            merged.append([s, e]); continue
        ps, pe = merged[-1]
        if s - pe <= min_gap:
            merged[-1][1] = max(pe, e)
        else:
            merged.append([s, e])

    # enforce max length
    out = []
    max_len = int(round(max_seg_sec * fs))
    for s, e in merged:
        L = e - s
        if L <= max_len:
            out.append((s, e))
        else:
            k = max(1, int(math.ceil(L / max_len)))
            step = int(round(L / k))
            cur = s
            for _ in range(k-1):
                out.append((cur, cur+step)); cur += step
            out.append((cur, e))
    return out

def _build_segments(t: np.ndarray, idx_pairs):
    n = len(t); segs = []
    for i, (s, e) in enumerate(idx_pairs):
        st = float(t[s])
        ed = float(t[e-1]) if e-1 < n else float(t[-1])
        segs.append({"index": i, "start_time": round(st,3),
                     "end_time": round(ed,3),
                     "duration": round(ed-st,3)})
    return segs

def auto_params_from_data(y_s: np.ndarray):
    """
    從平滑後訊號自動推門檻與回滯：
    - 低 40% 的中位數 ≈ baseline
    - 高 25% 的中位數 ≈ 目標平台
    - 以 robust σ 估噪聲，依分離度自動選 k_sigma 與 k_delta
    """
    q40 = np.percentile(y_s, 40)
    q75 = np.percentile(y_s, 75)
    base_med = np.median(y_s[y_s <= q40]) if np.any(y_s <= q40) else float(np.median(y_s))
    top_med  = np.median(y_s[y_s >= q75]) if np.any(y_s >= q75) else float(np.median(y_s))
    sigma = max(_mad_sigma(y_s), 1e-6)

    sep = (top_med - base_med) / sigma if sigma > 0 else 0.0
    # 根據分離度挑高門檻倍數
    if   sep >= 5.0: k_sigma = 3.5
    elif sep >= 3.0: k_sigma = 3.0
    elif sep >= 1.8: k_sigma = 2.5
    else:            k_sigma = 2.0

    # 低門檻 = 高門檻 - kDelta*sigma（回滯）
    k_delta = 0.9
    thr_hi = base_med + k_sigma * sigma
    thr_lo = thr_hi  - k_delta * sigma

    return dict(base_med=base_med, top_med=top_med, sigma=sigma,
                k_sigma=k_sigma, k_delta=k_delta,
                thr_hi=thr_hi, thr_lo=thr_lo)

def analyze(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    if "time_seconds" not in cols or "height_width_ratio" not in cols:
        raise ValueError("CSV 需含 time_seconds 與 height_width_ratio")

    t = pd.to_numeric(df[cols["time_seconds"]], errors="coerce").to_numpy()
    r = pd.to_numeric(df[cols["height_width_ratio"]], errors="coerce").to_numpy()
    m = np.isfinite(t) & np.isfinite(r)
    t, r = t[m], r[m]
    if len(t) < 2: raise ValueError("有效資料太少")

    fs = _estimate_fs(t)
    dt = 1.0/fs

    # 自動平滑（固定時間常數，依 fs 轉為窗長/alpha）
    box_win_sec = 0.12
    k = max(1, int(round(box_win_sec * fs)))
    r_ma = _moving_average(r, k)

    tau = 0.20  # 指數平滑時間常數
    alpha = dt / (tau + dt)   # 由 tau 推 alpha
    r_s  = _exp_smooth(r_ma, alpha)

    # 自動門檻/回滯
    th = auto_params_from_data(r_s)
    thr_hi, thr_lo = th["thr_hi"], th["thr_lo"]

    # 第一次切段
    raw = _hysteresis_segments(r_s, thr_hi, thr_lo)

    # 若完全抓不到，放寬一次（降低高門檻 15%）
    if len(raw) == 0:
        thr_hi2 = th["base_med"] + 0.85 * (thr_hi - th["base_med"])
        thr_lo2 = thr_hi2 - th["k_delta"] * th["sigma"]
        raw = _hysteresis_segments(r_s, thr_hi2, thr_lo2)

    # 依初步段落自動決定 pad/min_gap/max_len
    if raw:
        durs = np.array([(e-s)/fs for (s,e) in raw])
        med_dur = float(np.median(durs))
    else:
        med_dur = 3.0

    pad_sec     = min(0.15, 0.12 * med_dur)
    min_gap_sec = float(np.clip(0.30 * med_dur, 0.40, 1.00))
    max_seg_sec = float(np.clip(1.70 * med_dur, 2.50, 6.00))

    seg_idx = _pad_merge_cut(raw, fs, len(r_s),
                             pad_sec=pad_sec,
                             min_gap_sec=min_gap_sec,
                             max_seg_sec=max_seg_sec)
    segments = _build_segments(t, seg_idx)
    bk_times = [seg["end_time"] for seg in segments]

    # 嘟嘴次數與維持時間（扣頭 → /2 進位；扣頭後奇數段相加）
    pout_count = int(math.ceil(max(0, len(segments)-1) / 2.0))
    total_hold = 0.0
    for idx, seg in enumerate(segments[1:], start=1):
        if idx % 2 == 1:
            total_hold += seg["duration"]

    result = {
        "message": "LOCAL AUTO OK",
        "motion": "poutLip",
        "pout_count": pout_count,
        "total_hold_time": round(float(total_hold), 3),
        "breakpoints": [round(x, 2) for x in bk_times],
        "segments": segments,
        "debug": {
            "fs_hz": round(float(fs), 3),
            "thr_hi": round(float(thr_hi), 4),
            "thr_lo": round(float(thr_lo), 4),
            "pad_sec": round(pad_sec, 3),
            "min_gap_sec": round(min_gap_sec, 3),
            "max_seg_sec": round(max_seg_sec, 3),
            "raw_segments": len(raw),
            "final_segments": len(segments),
        }
    }

    # 視覺化
    if PLOT:
        plt.figure(figsize=(12,5))
        plt.plot(t, r,   label="raw", alpha=0.35, lw=1.0)
        plt.plot(t, r_s, label="smoothed", lw=1.6)
        plt.axhline(thr_hi, color="red", linestyle="--", alpha=0.7, label="thr_hi")
        plt.axhline(thr_lo, color="orange", linestyle="--", alpha=0.7, label="thr_lo")
        for seg in segments:
            plt.axvspan(seg["start_time"], seg["end_time"], color="#a0d8ff", alpha=0.25)
            mid = 0.5*(seg["start_time"]+seg["end_time"])
            plt.text(mid, thr_hi, f'{seg["duration"]:.2f}s',
                     ha="center", va="bottom", fontsize=8, color="#333")
        plt.title("height_width_ratio — AUTO hysteresis segmentation")
        plt.xlabel("Time (s)"); plt.ylabel("height_width_ratio")
        plt.legend(); plt.tight_layout(); plt.show()

    if WRITE_JSON and OUT_JSON_PATH:
        with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result

if __name__ == "__main__":
    analyze(CSV_PATH)
