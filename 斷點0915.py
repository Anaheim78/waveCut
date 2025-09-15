# pout_rup_len_snap_nofilter.py
# 依賴: pip install ruptures pandas numpy matplotlib
# 目的: 不濾波，使用 ruptures 的 L2 斷點偵測，依「預期段長」決定段數，適用於嘟嘴 height_width_ratio。

import os, math, json
import numpy as np
import pandas as pd
import ruptures as rpt
import matplotlib.pyplot as plt

# ======= 必改：你的 CSV 路徑（需包含: time_seconds, height_width_ratio）=======
CSV_PATH = r"C:\Users\plus1\Downloads\FaceTraining_POUT_LIPS_20250901_222635.csv"

# ======= 可調參數 ===========================================================
TARGET_LEN_SEC = 3.0   # 預期每段 ~3 秒（用來估計段數）
MIN_LEN_SEC    = 1.0   # 每段最短（平台穩可以拉高；要救矮峰可略降，但別太小）
PLOT           = True
WRITE_JSON     = False
OUT_JSON_PATH  = r"C:\Users\plus1\Downloads\pout_result_rup_nofilter.json"

# —— 讓它「稍微抓多一點」的兩個旋鈕（抓多後會用保險再合併短段）——
SENSITIVITY    = 1.25  # 預期段數 * 係數（例如 1.25 代表多抓 25%）
NBKPS_BONUS    = 1     # 在估計的 n_bkps 上 +1（或 +2）。0=不用

# 合併保險：把極短誤切的小段併回前段，避免一個動作被切兩半
MIN_TINY_SEC   = 0.7   # 小於此秒數視為誤切小段 → 併回前段
MAX_MERGE_GAP  = 0.5   # 小段與前段的間隙 ≤ 這個秒數才合併（避免跨太遠）

# snap 相關（平台中心）
SPEED_WIN_SEC  = 0.6   # 速度平滑視窗（秒）
EDGE_KEEP      = 0.10  # 中心搜尋時，每段兩端各保留 10%

# 視覺化顏色
SPAN_COLOR     = "#a0d8ff"
CENTER_COLOR   = "#cc0000"

# ---------------------------------------------------------------------------

def _estimate_fs(t: np.ndarray) -> float:
    if len(t) < 2: return 20.0
    dt = float(np.median(np.diff(t)))
    return 1.0 / dt if dt > 1e-9 else 20.0

def _rolling_mean(x: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k))
    ker = np.ones(k, dtype=float) / float(k)
    return np.convolve(x, ker, mode="same")

def _snap_centers_min_speed_1d(y: np.ndarray, bkps: list, fs: float,
                               win_sec: float = 0.6, edge_keep: float = 0.10):
    dy = np.gradient(y)
    k = int(max(3, round(fs * win_sec)))
    speed = _rolling_mean(np.abs(dy), k)

    centers = []
    starts = [0] + bkps[:-1]
    for s, e in zip(starts, bkps):
        if e - s <= 3:
            centers.append((s+e)//2); continue
        L = e - s
        left  = s + int(edge_keep * L)
        right = e - int(edge_keep * L)
        if right <= left:
            centers.append((s+e)//2); continue
        idx = int(np.argmin(speed[left:right]) + left)
        centers.append(idx)
    return np.array(centers, dtype=int)

def _build_segments_time(t: np.ndarray, bkps: list):
    segs, s = [], 0
    n = len(t)
    for i, e in enumerate(bkps):
        st = float(t[s])
        ed = float(t[e-1]) if e-1 < n else float(t[-1])
        segs.append({
            "index": i,
            "start_idx": int(s),
            "end_idx": int(e),
            "start_time": round(st, 3),
            "end_time": round(ed, 3),
            "duration": round(ed - st, 3),
        })
        s = e
    return segs

def _n_bkps_by_target_len(n: int, fs: float, target_len_sec: float, min_len_sec: float):
    total_dur = max(0.0, (n-1) / fs)
    exp_segments = max(1, int(math.floor(total_dur / max(1e-9, target_len_sec))))
    min_size = max(2, int(round(min_len_sec * fs)))
    k_max = max(0, n // max(2, min_size) - 1)
    n_bkps = max(0, min(k_max, exp_segments - 1))
    return n_bkps, min_size, total_dur, exp_segments

def _pout_count_and_hold(segments):
    n = len(segments)
    pout_count = int(math.ceil(max(0, n - 1) / 2.0))
    total_hold = 0.0
    for idx, seg in enumerate(segments[1:], start=1):
        if idx % 2 == 1:
            total_hold += float(seg["duration"])
    return pout_count, round(total_hold, 3)

def _merge_tiny_segments(bkps: list, fs: float,
                         min_tiny_sec: float = 0.7,
                         max_merge_gap_sec: float = 0.5) -> list:
    if not bkps: return bkps
    min_len = int(round(min_tiny_sec * fs))
    max_gap = int(round(max_merge_gap_sec * fs))
    segs, s = [], 0
    for e in bkps:
        segs.append([s, e])
        s = e
    out = []
    for i, (s, e) in enumerate(segs):
        if out:
            prev_s, prev_e = out[-1]
            gap = s - prev_e
            if (e - s) < min_len and gap <= max_gap:
                out[-1][1] = e
                continue
        out.append([s, e])
    bkps_new = [e for _, e in out]
    return bkps_new

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
    if len(t) < 3:
        raise ValueError("有效資料太少")

    # ===== 跳過前 5 秒 =====
    mask = t >= 5.0
    t, r = t[mask], r[mask]
    if len(t) < 3:
        raise ValueError("有效資料太少 (扣掉前5秒後)")

    # 平移時間軸，讓剩下的從 0 開始
    t = t - t[0]

    # ===== 不濾波：直接用原始 r =====
    fs = _estimate_fs(t)
    r_s = r.copy()

    n_bkps, min_size, total_dur, exp_segments = _n_bkps_by_target_len(
        n=len(r_s), fs=fs,
        target_len_sec=TARGET_LEN_SEC,
        min_len_sec=MIN_LEN_SEC
    )

    n_bkps_sens = int(round(n_bkps * SENSITIVITY)) + int(NBKPS_BONUS)
    k_max = max(0, len(r_s) // max(2, int(round(MIN_LEN_SEC * fs))) - 1)
    n_bkps_sens = max(0, min(k_max, n_bkps_sens))

    print(f"n={len(r_s)}, fs≈{fs:.3f} Hz, total_dur≈{total_dur:.2f}s")
    print(f"TARGET_LEN_SEC={TARGET_LEN_SEC}s → 預期段數≈{exp_segments} → base n_bkps={n_bkps} → sens n_bkps={n_bkps_sens}")
    print(f"MIN_LEN_SEC={MIN_LEN_SEC}s → min_size={int(round(MIN_LEN_SEC*fs))} samples")

    signal = r_s.reshape(-1, 1)
    algo = rpt.Binseg(model="l2", min_size=int(round(MIN_LEN_SEC * fs))).fit(signal)
    bkps = algo.predict(n_bkps=n_bkps_sens)

    bkps = _merge_tiny_segments(bkps, fs,
                                min_tiny_sec=MIN_TINY_SEC,
                                max_merge_gap_sec=MAX_MERGE_GAP)

    centers = _snap_centers_min_speed_1d(r_s, bkps, fs=fs,
                                         win_sec=SPEED_WIN_SEC,
                                         edge_keep=EDGE_KEEP)

    segments = _build_segments_time(t, bkps)
    pout_count, total_hold = _pout_count_and_hold(segments)

    out_dir = os.path.dirname(csv_path) or "."
    seg_csv = os.path.join(out_dir, "pout_segments_len_nofilter.csv")
    pd.DataFrame(segments).to_csv(seg_csv, index=False)

    center_csv = os.path.join(out_dir, "pout_centers_len_nofilter.csv")
    pd.DataFrame({"center_idx": centers, "center_time": t[centers]}).to_csv(center_csv, index=False)

    print(f"✅ 段落資訊：{seg_csv}")
    print(f"✅ 平台中心：{center_csv}")
    print(f"breakpoints (end indices): {bkps}")
    print(f"centers (indices): {centers}")

    if PLOT:
        n = len(t)
        plt.figure(figsize=(12, 5))
        plt.plot(t, r, label="raw", linewidth=1.4)
        for b in bkps[:-1]:
            xb = t[min(max(b-1, 0), n-1)]
            plt.axvline(xb, color="#666666", linestyle="--", alpha=0.6, linewidth=1.2)
        s = 0
        for e in bkps:
            x0 = t[s]
            x1 = t[e-1] if e-1 < n else t[-1]
            plt.axvspan(x0, x1, color=SPAN_COLOR, alpha=0.22)
            s = e
        for c in centers:
            xc = t[min(max(c, 0), n-1)]
            plt.axvline(xc, color=CENTER_COLOR, linestyle="-", alpha=0.85, linewidth=1.6)

        plt.title(f"pout height_width_ratio — ruptures L2 (no filter, skip first 5s) "
                  f"(target≈{TARGET_LEN_SEC}s, min={MIN_LEN_SEC}s, sens×{SENSITIVITY}, bonus+{NBKPS_BONUS})")
        plt.xlabel("Time (s)"); plt.ylabel("height_width_ratio"); plt.legend()
        out_png = os.path.join(out_dir, "pout_timeseries_len_snap_nofilter.png")
        plt.tight_layout(); plt.savefig(out_png, dpi=300, bbox_inches="tight"); plt.show()
        print(f"✅ 圖檔：{out_png}")

    result = {
        "message": "LOCAL RUPTURES OK (no filter, skip first 5s)",
        "motion": "poutLip",
        "pout_count": pout_count,
        "total_hold_time": total_hold,
        "breakpoints_time": [round(float(t[min(max(b-1,0), len(t)-1)]), 3) for b in bkps[:-1]],
        "centers_time": [round(float(t[c]), 3) for c in centers],
        "segments": segments,
        "debug": {
            "fs_hz": round(float(fs), 3),
            "n_bkps_base": int(n_bkps),
            "n_bkps_used": int(n_bkps_sens),
            "min_size_samples": int(round(MIN_LEN_SEC * fs)),
            "target_len_sec": float(TARGET_LEN_SEC),
            "min_len_sec": float(MIN_LEN_SEC),
            "model": "l2",
            "filter": "none",
            "merge_tiny_sec": float(MIN_TINY_SEC),
            "merge_gap_sec": float(MAX_MERGE_GAP),
            "skip_first_sec": 5.0
        }
    }

    if WRITE_JSON and OUT_JSON_PATH:
        with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    analyze(CSV_PATH)
