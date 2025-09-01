# api/ingest.py
# FastAPI 版（零手調自動分段）
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import io, csv, math
import numpy as np
import pandas as pd

app = FastAPI()

# --------- 核心工具（與你本地 pout_auto 相同邏輯，但不畫圖）---------
def _mad_sigma(x: np.ndarray) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad

def _estimate_fs(t: np.ndarray) -> float:
    if len(t) < 2:
        return 20.0
    dt = float(np.median(np.diff(t)))
    return 1.0 / dt if dt > 1e-9 else 20.0

def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x.copy()
    ker = np.ones(int(k), dtype=float) / float(k)
    return np.convolve(x, ker, mode="same")

def _exp_smooth(x: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0:
        return x.copy()
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = (1 - alpha) * y[i-1] + alpha * x[i]
    return y

def _hysteresis_segments(y: np.ndarray, th_hi: float, th_lo: float):
    segs, on, s, n = [], False, None, len(y)
    for i in range(n):
        if not on:
            if y[i] >= th_hi:
                on, s = True, i
        else:
            if y[i] < th_lo:
                e = i
                segs.append((s, max(e, s+1)))
                on, s = False, None
    if on and s is not None:
        segs.append((s, n))
    return segs

def _pad_merge_cut(segs, fs, n, pad_sec, min_gap_sec, max_seg_sec):
    if not segs:
        return []
    pad = int(round(pad_sec * fs))
    segs = [(max(0, s - pad), min(n, e + pad)) for (s, e) in segs]

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
            for _ in range(k - 1):
                out.append((cur, cur + step)); cur += step
            out.append((cur, e))
    return out

def _build_segments(t: np.ndarray, idx_pairs):
    n = len(t); segs = []
    for i, (s, e) in enumerate(idx_pairs):
        st = float(t[s])
        ed = float(t[e - 1]) if e - 1 < n else float(t[-1])
        segs.append({
            "index": i,
            "start_time": round(st, 3),
            "end_time": round(ed, 3),
            "duration": round(ed - st, 3),
        })
    return segs

def _auto_params_from_data(y_s: np.ndarray):
    # 低 40% 當 baseline，中位；高 25% 當目標平台，中位；MAD 估噪
    q40 = np.percentile(y_s, 40)
    q75 = np.percentile(y_s, 75)
    base_med = np.median(y_s[y_s <= q40]) if np.any(y_s <= q40) else float(np.median(y_s))
    top_med  = np.median(y_s[y_s >= q75]) if np.any(y_s >= q75) else float(np.median(y_s))
    sigma = max(_mad_sigma(y_s), 1e-6)

    sep = (top_med - base_med) / sigma if sigma > 0 else 0.0
    if   sep >= 5.0: k_sigma = 3.5
    elif sep >= 3.0: k_sigma = 3.0
    elif sep >= 1.8: k_sigma = 2.5
    else:            k_sigma = 2.0

    k_delta = 0.9  # 回滯量
    thr_hi = base_med + k_sigma * sigma
    thr_lo = thr_hi  - k_delta * sigma
    return dict(base_med=base_med, sigma=sigma, k_delta=k_delta, thr_hi=thr_hi, thr_lo=thr_lo)

def analyze_arrays(t: np.ndarray, r: np.ndarray):
    # 估 fs
    fs = _estimate_fs(t)
    dt = 1.0 / fs

    # 平滑：盒型 + 指數
    box_win_sec = 0.12
    k = max(1, int(round(box_win_sec * fs)))
    r_ma = _moving_average(r, k)

    tau = 0.20
    alpha = dt / (tau + dt)
    r_s = _exp_smooth(r_ma, alpha)

    # 自動門檻/回滯
    th = _auto_params_from_data(r_s)
    thr_hi, thr_lo = th["thr_hi"], th["thr_lo"]

    # 第一次切段
    raw = _hysteresis_segments(r_s, thr_hi, thr_lo)

    # 如果抓不到，放寬一次
    if len(raw) == 0:
        thr_hi2 = th["base_med"] + 0.85 * (thr_hi - th["base_med"])
        thr_lo2 = thr_hi2 - th["k_delta"] * th["sigma"]
        raw = _hysteresis_segments(r_s, thr_hi2, thr_lo2)

    # 依初步段落推 pad/min_gap/max_len
    if raw:
        durs = np.array([(e - s) / fs for (s, e) in raw])
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

    # breakpoints：回傳每段結束時間（不含最後一段）
    breakpoints = [round(seg["end_time"], 2) for seg in segments[:-1]]

    # 嘟嘴次數（扣頭 /2 進位）
    pout_count = int(math.ceil(max(0, len(segments) - 1) / 2.0))

    # 維持總時間：扣頭後奇數段(1,3,5,…)相加
    total_hold = sum(seg["duration"] for idx, seg in enumerate(segments[1:], start=1) if idx % 2 == 1)

    return {
        "motion": "poutLip",
        "pout_count": pout_count,
        "total_hold_time": round(float(total_hold), 3),
        "breakpoints": breakpoints,
        "segments": segments
    }

# --------- API 入口 ---------
@app.post("/")
async def ingest(req: Request):
    body = await req.json()
    lines = body.get("lines") or []
    if not lines or not isinstance(lines, list):
        return JSONResponse({"message": "no lines", "receivedCount": 0}, status_code=400)

    header = lines[0]
    cols_lower = [c.strip().lower() for c in header.split(",")]

    # 嘟嘴：看 header 是否含 height_width_ratio
    if "height_width_ratio" in cols_lower:
        # 轉 DataFrame
        reader = csv.DictReader(io.StringIO("\n".join(lines)))
        df = pd.DataFrame(reader)

        # 容錯（大小寫/空白）
        lowmap = {c.strip().lower(): c for c in df.columns}
        if "time_seconds" not in lowmap or "height_width_ratio" not in lowmap:
            return JSONResponse({"message": "bad schema", "motion": "poutLip"}, status_code=400)

        t = pd.to_numeric(df[lowmap["time_seconds"]], errors="coerce").to_numpy()
        r = pd.to_numeric(df[lowmap["height_width_ratio"]], errors="coerce").to_numpy()
        m = np.isfinite(t) & np.isfinite(r)
        t, r = t[m], r[m]
        if len(t) < 2:
            return JSONResponse({
                "message": "APITEST OK",
                "motion": "poutLip",
                "pout_count": 0,
                "total_hold_time": 0.0,
                "breakpoints": [],
                "segments": []
            }, status_code=200)

        result_core = analyze_arrays(t, r)
        result = {
            "message": "APITEST OK",
            **result_core
        }
        # 方便你在 Railway「Logs」看到
        print("APITEST result", result)
        return JSONResponse(result, status_code=200)

    # 非嘟嘴
    return JSONResponse({
        "message": "APITEST OK",
        "motion": "unknown",
        "reason": "missing height_width_ratio in header",
        "receivedCount": max(0, len(lines) - 1)
    }, status_code=200)
