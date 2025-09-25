from fastapi import Request
import io, csv, math
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# 臉頰點位（左右各 18 個）
LEFT_CHEEK_IDXS  = [117,118,101,36,203,212,214,192,147,123,98,97,164,0,37,39,40,186]
RIGHT_CHEEK_IDXS = [164,0,267,269,270,410,423,327,326,432,434,416,376,352,346,347,330,266]

# 濾波參數（與其他模組一致）
FS = 20.0
CUTOFF = 0.8
ORDER = 4

# ===== 濾波 & 前處理 =====
def lowpass_filter(x, fs=FS, cutoff=CUTOFF, order=ORDER):
    x = np.asarray(x, dtype=float)
    # filtfilt 需要最小長度；不足時直接回原訊號
    min_len = max(order * 3, 8)
    if x.size < min_len:
        return x
    b, a = butter(order, cutoff / (fs / 2), btype='low')
    return filtfilt(b, a, x)

def moving_average(x, win_samples):
    win = max(1, int(win_samples))
    ker = np.ones(win, dtype=float) / win
    pad = win // 2
    xpad = np.pad(x, pad_width=pad, mode="edge")
    base_full = np.convolve(xpad, ker, mode="same")
    return base_full[pad:-pad] if pad > 0 else base_full

def zero_crossings(x, t, deadband=0.0, min_interval=10):
    z_all, z_up, z_dn = [], [], []
    last = -min_interval
    for i in range(1, len(x)):
        xi_1, xi = x[i-1], x[i]
        if not np.isfinite(xi_1) or not np.isfinite(xi):
            continue
        # 負→正
        if xi_1 < 0 and xi >= 0 and abs(xi) > deadband:
            if i - last >= min_interval:
                z_all.append(i); z_up.append(i); last = i
        # 正→負
        elif xi_1 > 0 and xi <= 0 and abs(xi) > deadband:
            if i - last >= min_interval:
                z_all.append(i); z_dn.append(i); last = i
    return z_all, z_up, z_dn

# ===== 曲率（依你本地算法）=====
def fit_quadratic_surface_xyz(x, y, z):
    n = x.shape[0]
    if n < 6:
        return (0., 0., 0., 0., 0., 0.)
    A = np.column_stack([x*x, x*y, y*y, x, y, np.ones_like(x)])
    try:
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        return tuple(coef.tolist())
    except np.linalg.LinAlgError:
        return (0., 0., 0., 0., 0., 0.)

def curvature_proxy_from_quad(a, b, c):
    # 只取二次項當作曲率 proxy
    return math.sqrt((2*a)**2 + (2*c)**2 + 2*(b**2))

def cheek_patch_curvature(points3d):
    """
    points3d: shape (N, 3), columns = [x, y, z]
    """
    points3d = np.asarray(points3d, dtype=np.float32)
    if points3d.shape[0] < 6:
        return 0.0
    x = points3d[:, 0]
    y = points3d[:, 1]
    z = points3d[:, 2]
    a, b, c, d, e, f = fit_quadratic_surface_xyz(x, y, z)
    return curvature_proxy_from_quad(a, b, c)

# ===== 行別工具 =====
def _row_points3d(row, idxs):
    pts = []
    for k in idxs:
        px = float(row[f"point{k}_x"])
        py = float(row[f"point{k}_y"])
        pz = float(row[f"point{k}_z"])
        pts.append([px, py, pz])
    return np.asarray(pts, dtype=float)  # (N, 3)

# ===== 主流程（只做鼓臉，計「> 0」的段）=====
async def run(req: Request) -> dict:
    try:
        body = await req.json()
        lines = body.get("lines") or []
        if not lines or not isinstance(lines, list):
            return {"status": "ERROR", "error": "no lines"}

        # 轉 DataFrame（從 API rows，而非讀檔）
        df = pd.DataFrame(csv.DictReader(io.StringIO("\n".join(lines))))

        # 欄位檢查
        need_cols = [f"point{k}_{ax}" for k in LEFT_CHEEK_IDXS + RIGHT_CHEEK_IDXS for ax in ("x", "y", "z")]
        need_cols = ["time_seconds", "state"] + need_cols + ["img_w", "img_h"]
        missing = [c for c in need_cols if c not in df.columns]
        if missing:
            return {"status": "ERROR", "error": f"missing columns ({len(missing)})", "missing_preview": missing[:10]}

        if len(df) < 2:
            return {
                "status": "OK",
                "action_count": 0,
                "total_action_time": 0.0,
                "breakpoints": [],
                "segments": [],
                "debug": {"fs_hz": FS, "cutoff": CUTOFF, "order": ORDER, "note": "insufficient rows"}
            }

        # 每列重建 P_L/P_R → 曲率 → 時序
        curv_L, curv_R, t_list = [], [], []
        for _, row in df.iterrows():
            P_L = _row_points3d(row, LEFT_CHEEK_IDXS)
            P_R = _row_points3d(row, RIGHT_CHEEK_IDXS)
            cL = cheek_patch_curvature(P_L)
            cR = cheek_patch_curvature(P_R)
            curv_L.append(cL)
            curv_R.append(cR)
            t_list.append(float(row["time_seconds"]))

        t = np.asarray(t_list, dtype=float)
        s = np.asarray(curv_L, dtype=float) + np.asarray(curv_R, dtype=float)  # L+R

        # 前處理：低通 → 基線 → 扣除
        s_f = lowpass_filter(s, fs=FS, cutoff=CUTOFF, order=ORDER)
        baseline = moving_average(s_f, int(4.0 * FS))
        # 對齊長度保護（極短序列時 moving_average 會回原長或近原長）
        L = min(len(s_f), len(baseline))
        s_d = s_f[:L] - baseline[:L]
        t = t[:L]

        # 零交叉（取 >0 的段）
        std = float(np.std(s_d)) if len(s_d) else 0.0
        deadband = 0.005 * std if std > 0 else 0.0
        min_interval = int(0.5 * FS)
        zc_all, zc_up, zc_down = zero_crossings(s_d, t, deadband=deadband, min_interval=min_interval)

        segments = []
        positive_segments = []
        if len(zc_all) >= 2:
            for i, (s_idx, e_idx) in enumerate(zip(zc_all[:-1], zc_all[1:])):
                st, ed = float(t[s_idx]), float(t[e_idx])
                dur = round(ed - st, 3)
                seg = {"index": i, "start_time": round(st, 3), "end_time": round(ed, 3), "duration": dur}
                segments.append(seg)
                # 只計「>0」區間
                if s_d[s_idx] >= 0:
                    positive_segments.append(seg)

        action_count = len(positive_segments)
        total_action_time = round(sum(seg["duration"] for seg in positive_segments), 3)
        breakpoints = [seg["end_time"] for seg in segments]

        return {
            "status": "OK",
            "action_count": action_count,
            "total_action_time": total_action_time,
            "breakpoints": breakpoints,
            "segments": segments,
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












# 手寫板
# # api/count_puff_cheek.py
# from fastapi import Request
# import io, csv
# import numpy as np
# import math
# import pandas as pd
# import mediapipe as mp
# from scipy.signal import savgol_filter
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from scipy.signal import find_peaks
# import matplotlib.pyplot as plt
# from scipy.signal import butter, filtfilt


# # 兩組臉頰點
# LEFT_CHEEK_IDXS  = [117,118,101,36,203,212,214,192,147,123,98,97,164,0,37,39,40,186]
# RIGHT_CHEEK_IDXS = [164,0,267,269,270,410,423,327,326,432,434,416,376,352,346,347,330,266]

# # 參數設定
# NORMALIZE_MODE = 'raw'
# BASELINE_SEC = 2.0
# SMOOTH_WINDOW = 9
# SMOOTH_POLY = 2
# PLOT_WINDOW_SEC = 4


# # ===== 低通濾波器 =====
# def lowpass_filter(x, fs=20.0, cutoff=0.5, order=4):
#     b, a = butter(order, cutoff / (fs / 2), btype='low')
#     y = filtfilt(b, a, x)
#     print(f"[DEBUG] Lowpass: cutoff={cutoff} Hz, order={order}, fs={fs}")
#     return y

# # ===== 移動平均 (基線估計) =====
# def moving_average(x, win_samples=60):
#     kernel = np.ones(win_samples) / win_samples
#     pad_width = win_samples // 2
#     x_padded = np.pad(x, pad_width, mode='edge')
#     baseline_full = np.convolve(x_padded, kernel, mode='same')
#     baseline = baseline_full[pad_width:-pad_width]
#     return baseline

# # ===== 零交叉檢測 =====
# def zero_crossings(x, t, deadband=0.0, min_interval=10):
#     crossings_all = []
#     crossings_up = []    # 負→正
#     crossings_down = []  # 正→負

#     last_idx = -min_interval
#     for i in range(1, len(x)):
#         # 負 -> 正
#         if x[i-1] < 0 and x[i] >= 0 and abs(x[i]) > deadband:
#             if i - last_idx >= min_interval:
#                 crossings_all.append(i)
#                 crossings_up.append(i)
#                 last_idx = i
#         # 正 -> 負
#         elif x[i-1] > 0 and x[i] <= 0 and abs(x[i]) > deadband:
#             if i - last_idx >= min_interval:
#                 crossings_all.append(i)
#                 crossings_down.append(i)
#                 last_idx = i

#     return crossings_all, crossings_up, crossings_down

# # ------------------ 新版曲率計算 ------------------
# def fit_quadratic_surface_xyz(x, y, z):
#     n = x.shape[0]
#     if n < 6:
#         return (0., 0., 0., 0., 0., 0.)
#     # z ≈ a*x^2 + b*x*y + c*y^2 + d*x + e*y + f
#     A = np.column_stack([x*x, x*y, y*y, x, y, np.ones_like(x)])
#     #回傳六組系數
#     try:
#         coef, *_ = np.linalg.lstsq(A, z, rcond=None)
#         return tuple(coef.tolist())
#     except np.linalg.LinAlgError:
#         return (0., 0., 0., 0., 0., 0.)

# def curvature_proxy_from_quad(a, b, c):
#     return math.sqrt((2*a)**2 + (2*c)**2 + 2*(b**2))


# def cheek_patch_curvature(points3d):
#     """
#     points3d: shape (N, 3), columns = [x,y,z]
#     """
#     points3d = np.asarray(points3d, dtype=np.float32)
#     if points3d.shape[0] < 6:
#         return 0.0
#     x = points3d[:,0]
#     y = points3d[:,1]
#     z = points3d[:,2]
#     a,b,c,d,e,f = fit_quadratic_surface_xyz(x, y, z)
#     return curvature_proxy_from_quad(a,b,c)
# # ---------------------------------------------------

# def smooth_series(x, win=SMOOTH_WINDOW, poly=SMOOTH_POLY):
#     x = np.asarray(x, dtype=float)
#     # if len(x) >= win and win % 2 == 1:
#     #     return savgol_filter(x, win, poly, mode='interp')
#     return x

# def _row_points3d(row, idxs):
#     pts = []
#     for k in idxs:
#         px = float(row[f"point{k}_x"])
#         py = float(row[f"point{k}_y"])
#         pz = float(row[f"point{k}_z"])
#         pts.append([px, py, pz])
#     return np.asarray(pts, dtype=float)  # (N, 3)

# async def run(req: Request, mode="PUFF"):
#     """
#     只解析 payload，檢查第一列 row 組出 P_L / P_R，印出 shape，並回傳 shape。
#     """
#     body = await req.json()
#     lines = body.get("lines") or []
#     if not lines or not isinstance(lines, list):
#         print("[shapecheck] no lines")
#         return {"status": "ERROR", "error": "no lines"}

#     # 轉 DataFrame
#     df = pd.DataFrame(csv.DictReader(io.StringIO("\n".join(lines))))
#     print(f"[shapecheck] rows={len(df)}, cols={len(df.columns)}")

#     # 檢查必要欄位
#     need = [f"point{k}_{ax}" for k in LEFT_CHEEK_IDXS+RIGHT_CHEEK_IDXS for ax in ("x","y","z")]
#     need = ["time_seconds", "state"] + need + ["img_w", "img_h"]
#     missing = [c for c in need if c not in df.columns]
#     if missing:
#         print(f"[shapecheck] missing columns: {missing[:6]} ... total {len(missing)}")
#         return {"status": "ERROR", "error": f"missing columns ({len(missing)})", "missing_preview": missing[:10]}

#     if len(df) == 0:
#         return {"status": "ERROR", "error": "no data rows"}
    
#     curv_L, curv_R, t_list = [], [], []
#     for _, row in df.iterrows():
#         P_L = _row_points3d(row, LEFT_CHEEK_IDXS)
#         P_R = _row_points3d(row, RIGHT_CHEEK_IDXS)
#         cL = cheek_patch_curvature(P_L)
#         cR = cheek_patch_curvature(P_R)
#         curv_L.append(cL)
#         curv_R.append(cR)
#         t_list.append(float(row["time_seconds"]))

#     Ls = smooth_series(np.array(curv_L))
#     Rs = smooth_series(np.array(curv_R))
#     Sums = Ls + Rs

#     fs = 30.0  # 假設 30fps
#     # 低通濾波
#     r_filt = lowpass_filter(Sums, fs=fs, cutoff=0.8, order=4)

#     # 基線 (移動平均)
#     baseline = moving_average(r_filt, int(4.0 * fs))
#     r_detrend = r_filt - baseline

#     # 零交叉
#     zc_all, zc_up, zc_down = zero_crossings(
#         r_detrend, np.array(t_list),
#         deadband=0.005*np.std(r_detrend),
#         min_interval=int(0.5*fs)
#     )

#     return {
#         "status": "OK",
#         "mode": mode,
#         "rows": len(df),
#         "cols": len(df.columns),
#         "count_total": len(zc_all),
#         "count_up": len(zc_up),
#         "count_down": len(zc_down),
#         "zc_up_times": [float(t_list[i]) for i in zc_up],
#         "zc_down_times": [float(t_list[i]) for i in zc_down],
#     }


#     # # 只拿第一列做 shape 檢查
#     # row0 = df.iloc[0]
#     # P_L = _row_points3d(row0, LEFT_CHEEK_IDXS)   # (18, 3)
#     # P_R = _row_points3d(row0, RIGHT_CHEEK_IDXS)  # (18, 3)

#     # # print(f"[shapecheck] P_L shape: {P_L.shape}")
#     # # print(f"[shapecheck] P_R shape: {P_R.shape}")
#     # # # 若想快速肉眼確認，印一點點數值：
#     # # print(f"[shapecheck] P_L sample first 2: {P_L[:2].tolist()}")
#     # # print(f"[shapecheck] P_R sample first 2: {P_R[:2].tolist()}")

#     # # return {
#     # #     "status": "OK",
#     # #     "mode": mode,
#     # #     "rows": len(df),
#     # #     "cols": len(df.columns),
#     # #     "left_shape": list(P_L.shape),
#     # #     "right_shape": list(P_R.shape),
#     # # }
