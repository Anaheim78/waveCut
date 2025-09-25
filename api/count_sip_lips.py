from fastapi import Request
import io, csv
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# ===== 參數（與你現有邏輯一致） =====
FS = 20.0
CUTOFF = 0.8
ORDER = 4

# ===== 低通濾波器 =====
def lowpass_filter(x, fs=FS, cutoff=CUTOFF, order=ORDER):
    b, a = butter(order, cutoff / (fs / 2), btype='low')
    y = filtfilt(b, a, x)
    return y

# ===== 移動平均（基線估計）=====
def moving_average(x, win_samples):
    if win_samples < 1:
        win_samples = 1
    kernel = np.ones(win_samples) / win_samples
    pad_width = win_samples // 2
    x_padded = np.pad(x, pad_width, mode='edge')
    baseline_full = np.convolve(x_padded, kernel, mode='same')
    baseline = baseline_full[pad_width:-pad_width]
    return baseline

# ===== 零交叉檢測 =====
def zero_crossings(x, t, deadband=0.0, min_interval=10):
    crossings_all, crossings_up, crossings_down = [], [], []
    last_idx = -min_interval
    for i in range(1, len(x)):
        if np.isnan(x[i-1]) or np.isnan(x[i]):
            continue
        # 負 -> 正
        if x[i-1] < 0 and x[i] >= 0 and abs(x[i]) > deadband:
            if i - last_idx >= min_interval:
                crossings_all.append(i)
                crossings_up.append(i)
                last_idx = i
        # 正 -> 負
        elif x[i-1] > 0 and x[i] <= 0 and abs(x[i]) > deadband:
            if i - last_idx >= min_interval:
                crossings_all.append(i)
                crossings_down.append(i)
                last_idx = i
    return crossings_all, crossings_up, crossings_down

async def run(req: Request) -> dict:
    """
    子程式自己解析 req：
    - 讀 lines → DataFrame
    - 取 time_seconds、height_width_ratio
    - lowpass → baseline 扣除 → zero crossing
    - 建立 segments（全段保留），但 action_count / total_action_time 只算「>0 的段」
    - 維持舊 API 欄位：action_count / total_action_time / breakpoints / segments / debug
    """
    try:
        body = await req.json()
        lines = body.get("lines") or []
        if not lines or not isinstance(lines, list):
            return {"message": "no lines", "receivedCount": 0, "status": "ERROR"}

        # CSV 轉 DataFrame（容忍表頭空白大小寫）
        df = pd.DataFrame(csv.DictReader(io.StringIO("\n".join(lines))))
        lowmap = {str(c).strip().lower(): c for c in df.columns if c is not None}

        # 必要欄位檢查
        if "time_seconds" not in lowmap:
            return {"status": "ERROR", "error": "Missing time_seconds column"}
        if "total_lip_area" not in lowmap:
            return {"status": "ERROR", "error": "Missing total_lip_area column for pout detection"}

        # 讀資料 + 過濾 NaN
        t_raw = pd.to_numeric(df[lowmap["time_seconds"]], errors="coerce").to_numpy()
        r_raw = pd.to_numeric(df[lowmap["total_lip_area"]], errors="coerce").to_numpy()
        m = np.isfinite(t_raw) & np.isfinite(r_raw)
        t, r = t_raw[m], r_raw[m]

        if len(t) < 2:
            return {
                "status": "OK",
                "action_count": 0,
                "total_action_time": 0.0,
                "breakpoints": [],
                "segments": [],
                "debug": {"fs_hz": FS, "cutoff": CUTOFF, "order": ORDER, "note": "insufficient data"}
            }

        # 低通
        r_filt = lowpass_filter(r, fs=FS, cutoff=CUTOFF, order=ORDER)

        # 基線扣除
        win = int(4.0 * FS)  # 與你現行邏輯一致
        baseline = moving_average(r_filt, win)
        r_detrend = r_filt - baseline

        # 零交叉
        deadband = 0.005 * float(np.std(r_detrend)) if np.std(r_detrend) > 0 else 0.0
        min_interval = int(0.5 * FS)
        zc_all, zc_up, zc_down = zero_crossings(r_detrend, t, deadband=deadband, min_interval=min_interval)

        # 組全段 segments（相鄰零交叉為一段）
        segments = []
        negative_segments = []
        if len(zc_all) >= 2:
            for i, (s, e) in enumerate(zip(zc_all[:-1], zc_all[1:])):
                st, ed = float(t[s]), float(t[e])
                dur = round(ed - st, 3)
                seg = {
                    "index": i,
                    "start_time": round(st, 3),
                    "end_time": round(ed, 3),
                    "duration": dur
                }
                segments.append(seg)
                # 只統計「起點在 >0」的段
                if r_detrend[s] < 0:
                    negative_segments.append(seg)

        breakpoints = [seg["end_time"] for seg in segments]
        total_action_time = round(sum(seg["duration"] for seg in negative_segments), 3)

        return {
            "status": "OK",
            "action_count": len(negative_segments),     # 只算正區段
            "total_action_time": total_action_time,
            "breakpoints": breakpoints,                  # 全段的 end_time
            "segments": segments,                        # 全段
            "debug": {
                "fs_hz": FS,
                "cutoff": CUTOFF,
                "order": ORDER,
                "zc_all": len(zc_all),
                "zc_up": len(zc_up),
                "zc_down": len(zc_down),
                "deadband": round(deadband, 6),
                "min_interval": min_interval
            }
        }

    except Exception as e:
        return {"status": "ERROR", "error": str(e)}
