from fastapi import FastAPI, Request
import pandas as pd
import io

# 匯入五個子模組
import count_tongue
import count_puff_cheek
import count_reduce_cheek
import count_pout_lips
import count_sip_lips

app = FastAPI()

# 註冊表：trainingType → 模組
ANALYZERS = {
    # 舌頭
    "TONGUE_RIGHT": count_tongue,
    "TONGUE_LEFT": count_tongue,
    "TONGUE_UP": count_tongue,
    "TONGUE_DOWN": count_tongue,
    # 鼓臉
    "PUFF_CHEEK": count_puff_cheek,
    # 縮臉
    "REDUCE_CHEEK": count_reduce_cheek,
    # 嘟嘴
    "POUT_LIPS": count_pout_lips,
    # 抿嘴
    "SIP_LIPS": count_sip_lips,
}

@app.post("/")
async def ingest(request: Request):
    payload = await request.json()
    training_type = payload.get("trainingType")
    lines = payload.get("lines")

    if training_type not in ANALYZERS:
        return {"error": f"Unknown trainingType: {training_type}"}

    # 將 CSV 轉成 DataFrame
    csv_str = "\n".join(lines)
    df = pd.read_csv(io.StringIO(csv_str))

    analyzer = ANALYZERS[training_type]
    result = analyzer.run(df, training_type=training_type)

    return {"trainingType": training_type, "result": result}


# # api/ingest.py
# # FastAPI — 原始robust算法 + 双向检测 + 修正计数逻辑
# from fastapi import FastAPI, Request
# from fastapi.responses import JSONResponse
# import io, csv, math
# import numpy as np
# import pandas as pd

# app = FastAPI()

# # ===== 原始robust算法核心 =====
# class RobustDetector:
#     @staticmethod
#     def _mad_sigma(x: np.ndarray) -> float:
#         """使用MAD估算robust标准差"""
#         med = np.median(x)
#         mad = np.median(np.abs(x - med))
#         return 1.4826 * mad

#     @staticmethod
#     def _estimate_fs(t: np.ndarray) -> float:
#         """估算采样率"""
#         #列出全部相鄰間格的陣列，中位數
#         if len(t) < 2: 
#             return 20.0
#         dt = float(np.median(np.diff(t)))
#         return 1.0 / dt if dt > 1e-9 else 20.0

#     @staticmethod
#     def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
#         """移动平均滤波"""
#         if k <= 1: 
#             return x.copy()
#         ker = np.ones(int(k), dtype=float) / float(k)
#         return np.convolve(x, ker, mode="same")

#     @staticmethod
#     def _exp_smooth(x: np.ndarray, alpha: float) -> np.ndarray:
#         """指数平滑滤波"""
#         if alpha <= 0: 
#             return x.copy()
#         y = np.empty_like(x, dtype=float)
#         y[0] = x[0]
#         for i in range(1, len(x)):
#             y[i] = (1 - alpha) * y[i-1] + alpha * x[i]
#         return y

#     @staticmethod
#     def _hysteresis_segments(y: np.ndarray, th_hi: float, th_lo: float, detect_high=True):
#         """双向回滞检测"""
#         segs, on, s, n = [], False, None, len(y)
        
#         for i in range(n):
#             if not on:
#                 # 寻找动作开始
#                 if detect_high:
#                     condition_met = y[i] >= th_hi
#                 else:
#                     condition_met = y[i] <= th_hi  # 注意：低谷检测时th_hi是低阈值
                
#                 if condition_met:
#                     on, s = True, i
#             else:
#                 # 寻找动作结束
#                 if detect_high:
#                     condition_met = y[i] < th_lo
#                 else:
#                     condition_met = y[i] > th_lo  # 注意：低谷检测时th_lo是高阈值
                
#                 if condition_met:
#                     e = i
#                     segs.append((s, max(e, s+1)))
#                     on, s = False, None
        
#         if on and s is not None:
#             segs.append((s, n))
#         return segs

#     @staticmethod
#     def _pad_merge_cut(segs, fs, n, pad_sec, min_gap_sec, max_seg_sec):
#         """原始的段落后处理逻辑"""
#         if not segs: 
#             return []

#         # 第一阶段：预合并（在padding前）
#         merged = []
#         min_gap = int(round(min_gap_sec * fs))
#         for s, e in sorted(segs):
#             if not merged:
#                 merged.append([s, e])
#                 continue
#             ps, pe = merged[-1]
#             if s - pe <= min_gap:
#                 merged[-1][1] = max(pe, e)
#             else:
#                 merged.append([s, e])

#         # 第二阶段：强制合并
#         FIX_MIN_GAP_SEC = 0.3
#         fix_gap = int(round(FIX_MIN_GAP_SEC * fs))
#         final_merged = []
#         for s, e in merged:
#             if not final_merged:
#                 final_merged.append([s, e])
#                 continue
#             ps, pe = final_merged[-1]
#             if s - pe <= fix_gap:
#                 final_merged[-1][1] = max(pe, e)
#             else:
#                 final_merged.append([s, e])

#         # 第三阶段：添加padding
#         pad = int(round(pad_sec * fs))
#         padded_segs = [(max(0, s-pad), min(n, e+pad)) for (s, e) in final_merged]

#         # 第四阶段：长度限制
#         final = []
#         max_len = int(round(max_seg_sec * fs))
#         for s, e in padded_segs:
#             L = e - s
#             if L <= max_len:
#                 final.append((s, e))
#             else:
#                 k = max(1, int(math.ceil(L / max_len)))
#                 step = int(round(L / k))
#                 cur = s
#                 for _ in range(k-1):
#                     final.append((cur, cur+step))
#                     cur += step
#                 final.append((cur, e))
#         return final

#     @staticmethod
#     def _final_time_merge(segments, timestamps, max_gap_sec=0.2):
#         """基于实际时间的最终合并"""
#         if len(segments) < 2:
#             return segments
        
#         final_merged = []
#         current_start, current_end = segments[0]
        
#         for start, end in segments[1:]:
#             gap_time = timestamps[start] - timestamps[current_end]
#             if gap_time <= max_gap_sec:
#                 current_end = end  # 合并段落
#             else:
#                 final_merged.append((current_start, current_end))
#                 current_start, current_end = start, end
        
#         final_merged.append((current_start, current_end))
#         return final_merged

#     @staticmethod
#     def _build_segments(t: np.ndarray, idx_pairs):
#         """构建段落输出格式"""
#         n = len(t)
#         segs = []
#         for i, (s, e) in enumerate(idx_pairs):
#             st = float(t[s])
#             ed = float(t[e-1]) if e-1 < n else float(t[-1])
#             segs.append({
#                 "index": i, 
#                 "start_time": round(st, 3),
#                 "end_time": round(ed, 3),
#                 "duration": round(ed - st, 3)
#             })
#         return segs

#     @staticmethod
#     def auto_params_from_data(y_s: np.ndarray, detect_high=True):
#         """原始的自适应参数推导 + 双向支持 + 提高敏感度"""
#         if detect_high:
#             # 嘟嘴：检测高平台（调整后的逻辑）
#             q40 = np.percentile(y_s, 40)
#             q75 = np.percentile(y_s, 75)
#             base_med = np.median(y_s[y_s <= q40]) if np.any(y_s <= q40) else float(np.median(y_s))
#             top_med = np.median(y_s[y_s >= q75]) if np.any(y_s >= q75) else float(np.median(y_s))
#             sigma = max(RobustDetector._mad_sigma(y_s), 1e-6)

#             sep = (top_med - base_med) / sigma if sigma > 0 else 0.0
#             # 更敏感的k_sigma选择（降低所有档位）
#             if   sep >= 5.0: k_sigma = 3.2  # 原3.5 → 3.2
#             elif sep >= 3.0: k_sigma = 2.7  # 原3.0 → 2.7
#             elif sep >= 1.8: k_sigma = 2.2  # 原2.5 → 2.2
#             elif sep >= 1.2: k_sigma = 1.9  # 新增档位
#             else:            k_sigma = 1.7  # 原2.0 → 1.7

#             k_delta = 0.9
#             thr_hi = base_med + k_sigma * sigma
#             thr_lo = thr_hi - k_delta * sigma

#         else:
#             # 闭嘴唇：检测低谷（镜像逻辑，同样提高敏感度）
#             q25 = np.percentile(y_s, 25)
#             q60 = np.percentile(y_s, 60)
#             base_med = np.median(y_s[y_s >= q60]) if np.any(y_s >= q60) else float(np.median(y_s))
#             bottom_med = np.median(y_s[y_s <= q25]) if np.any(y_s <= q25) else float(np.median(y_s))
#             sigma = max(RobustDetector._mad_sigma(y_s), 1e-6)

#             sep = (base_med - bottom_med) / sigma if sigma > 0 else 0.0
#             # 同样更敏感的k_sigma
#             if   sep >= 5.0: k_sigma = 3.2
#             elif sep >= 3.0: k_sigma = 2.7
#             elif sep >= 1.8: k_sigma = 2.2
#             elif sep >= 1.2: k_sigma = 1.9
#             else:            k_sigma = 1.7

#             k_delta = 0.9
#             thr_hi = base_med - k_sigma * sigma  # 向下的激活阈值
#             thr_lo = thr_hi + k_delta * sigma    # 向上的回滞阈值

#         return dict(
#             base_med=base_med, 
#             sigma=sigma,
#             k_sigma=k_sigma, 
#             k_delta=k_delta,
#             thr_hi=thr_hi, 
#             thr_lo=thr_lo
#         )

#     @staticmethod
#     def analyze_robust(t: np.ndarray, r: np.ndarray, detect_high=True, motion_name=""):
#         """原始robust算法主函数 + 双向支持"""
#         fs = RobustDetector._estimate_fs(t)
#         dt = 1.0/fs

#         # 原始的信号预处理参数
#         box_win_sec = 0.12
#         k = max(1, int(round(box_win_sec * fs)))
#         r_ma = RobustDetector._moving_average(r, k)

#         tau = 0.20  # 原始的时间常数
#         alpha = dt / (tau + dt)
#         r_s = RobustDetector._exp_smooth(r_ma, alpha)

#         # 自适应阈值计算
#         th = RobustDetector.auto_params_from_data(r_s, detect_high)
#         thr_hi, thr_lo = th["thr_hi"], th["thr_lo"]

#         # 第一次检测
#         raw = RobustDetector._hysteresis_segments(r_s, thr_hi, thr_lo, detect_high)

#         # 如果检测不到，放宽阈值（原始逻辑）
#         if len(raw) == 0:
#             if detect_high:
#                 thr_hi2 = th["base_med"] + 0.85 * (thr_hi - th["base_med"])
#                 thr_lo2 = thr_hi2 - th["k_delta"] * th["sigma"]
#             else:
#                 thr_hi2 = th["base_med"] - 0.85 * (th["base_med"] - thr_hi)
#                 thr_lo2 = thr_hi2 + th["k_delta"] * th["sigma"]
            
#             raw = RobustDetector._hysteresis_segments(r_s, thr_hi2, thr_lo2, detect_high)
#             thr_hi, thr_lo = thr_hi2, thr_lo2

#         # 原始的自适应参数推导
#         if raw:
#             durs = np.array([(e-s)/fs for (s,e) in raw])
#             med_dur = float(np.median(durs))
#         else:
#             med_dur = 3.0

#         pad_sec = min(0.15, 0.12 * med_dur)
#         min_gap_sec = float(np.clip(0.30 * med_dur, 0.40, 1.00))
#         max_seg_sec = float(np.clip(1.70 * med_dur, 2.50, 6.00))

#         # 段落后处理
#         seg_idx = RobustDetector._pad_merge_cut(raw, fs, len(r_s),
#                                                pad_sec=pad_sec,
#                                                min_gap_sec=min_gap_sec,
#                                                max_seg_sec=max_seg_sec)
        
#         # 方案A：基于时间的最终合并
#         final_idx = RobustDetector._final_time_merge(seg_idx, t, max_gap_sec=0.2)
        
#         segments = RobustDetector._build_segments(t, final_idx)

#         # 修正的计数逻辑：每个段落就是一次动作
#         action_count = len(segments)
#         total_time = sum(seg["duration"] for seg in segments)
#         breakpoints = [round(seg["end_time"], 2) for seg in segments]

#         direction = "高平台" if detect_high else "低谷"
#         print(f"Robust {motion_name}检测({direction}): {action_count}次, 总时长{total_time:.2f}s")

#         return {
#             "action_count": action_count,
#             "total_action_time": round(float(total_time), 3),
#             "breakpoints": breakpoints,
#             "segments": segments,
#             "debug": {
#                 "fs_hz": round(float(fs), 3),
#                 "thr_hi": round(float(thr_hi), 4),
#                 "thr_lo": round(float(thr_lo), 4),
#                 "pad_sec": round(pad_sec, 3),
#                 "min_gap_sec": round(min_gap_sec, 3),
#                 "max_seg_sec": round(max_seg_sec, 3),
#                 "raw_segments": len(raw),
#                 "final_segments": len(segments),
#             }
#         }

# # ===== 动作识别器 =====
# class MotionDetector:
#     @staticmethod
#     def detect_motion_type(header_columns):
#         # 防御性处理，确保所有元素都不是None
#         cols_lower = []
#         for col in header_columns:
#             if col is not None:
#                 cols_lower.append(str(col).strip().lower())
        
#         if "height_width_ratio" in cols_lower:
#             return "poutLip"
#         elif "total_lip_area" in cols_lower:
#             return "closeLip"
#         else:
#             return "unknown"

# # ===== 动作分析器 =====
# class PoutLipAnalyzer:
#     @staticmethod
#     def analyze(timestamps, height_width_ratios):
#         result = RobustDetector.analyze_robust(
#             timestamps, height_width_ratios, 
#             detect_high=True, motion_name="嘟嘴"
#         )
#         return {
#             "motion": "poutLip",
#             "pout_count": result["action_count"],
#             "total_hold_time": result["total_action_time"],
#             "breakpoints": result["breakpoints"],
#             "segments": result["segments"],
#             "debug": result["debug"]
#         }

# class CloseLipAnalyzer:
#     @staticmethod
#     def analyze(timestamps, total_lip_areas):
#         result = RobustDetector.analyze_robust(
#             timestamps, total_lip_areas,
#             detect_high=False, motion_name="闭嘴唇"
#         )
#         return {
#             "motion": "closeLip", 
#             "close_count": result["action_count"],
#             "total_close_time": result["total_action_time"],
#             "breakpoints": result["breakpoints"],
#             "segments": result["segments"],
#             "debug": result["debug"]
#         }

# # ===== 数据处理工具 =====
# class DataProcessor:
#     @staticmethod
#     def validate_and_extract(df, motion_type):
#         # 防御性处理列名，避免None值
#         # 存入column名稱
#         lowmap = {}
#         for c in df.columns:
#             if c is not None:
#                 clean_col = str(c).strip().lower()
#                 lowmap[clean_col] = c
        
#         if "time_seconds" not in lowmap:
#             raise ValueError("Missing time_seconds column")
        
#         timestamps = pd.to_numeric(df[lowmap["time_seconds"]], errors="coerce").to_numpy()
        
#         #存取個別動作的專用指標
#         if motion_type == "poutLip":
#             if "height_width_ratio" not in lowmap:
#                 raise ValueError("Missing height_width_ratio column for pout detection")
#             values = pd.to_numeric(df[lowmap["height_width_ratio"]], errors="coerce").to_numpy()
        
#         elif motion_type == "closeLip":
#             if "total_lip_area" not in lowmap:
#                 raise ValueError("Missing total_lip_area column for close lip detection")
#             values = pd.to_numeric(df[lowmap["total_lip_area"]], errors="coerce").to_numpy()
        
#         else:
#             raise ValueError(f"Unsupported motion type: {motion_type}")
        
#         # 清理无效数据
#         valid_mask = np.isfinite(timestamps) & np.isfinite(values)
#         timestamps = timestamps[valid_mask]
#         values = values[valid_mask]
        
#         if len(timestamps) < 2:
#             raise ValueError("Insufficient valid data points")
        
#         return timestamps, values

# # ===== FastAPI路由 =====
# @app.post("/")
# async def ingest(req: Request):
#     try:
#         body = await req.json()
#         lines = body.get("lines") or []
        
#         if not lines or not isinstance(lines, list):
#             return JSONResponse({"message": "no lines", "receivedCount": 0}, status_code=400)

#         header = lines[0]
#         # 防御性处理header，分割后过滤空值
#         header_columns = []
#         for col in header.split(","):
#             if col is not None and str(col).strip():
#                 header_columns.append(str(col).strip())
        
#         # 识别动作类型
#         motion_type = MotionDetector.detect_motion_type(header_columns)
        
#         if motion_type == "unknown":
#             return JSONResponse({
#                 "message": "ROBUST API OK",
#                 "motion": "unknown",
#                 "reason": "no supported motion columns found",
#                 "supported_motions": ["poutLip (height_width_ratio)", "closeLip (total_lip_area)"],
#                 "receivedCount": max(0, len(lines) - 1)
#             }, status_code=200)

#         try:
#             # 解析CSV数据
#             reader = csv.DictReader(io.StringIO("\n".join(lines)))
#             df = pd.DataFrame(reader)
            
#             # 验证并提取数据
#             #DataProcessor處理工具，切出波形圖的時間與值
#             timestamps, values = DataProcessor.validate_and_extract(df, motion_type)
            
#             # 根据动作类型调用相应分析器
#             if motion_type == "poutLip":
#                 result_core = PoutLipAnalyzer.analyze(timestamps, values)
#             elif motion_type == "closeLip":
#                 result_core = CloseLipAnalyzer.analyze(timestamps, values)
#             else:
#                 raise ValueError(f"Motion type {motion_type} not implemented")
            
#             result = {"message": "ROBUST API OK", **result_core}
            
#             print(f"Robust检测API结果 [{motion_type}]:", result)
#             return JSONResponse(result, status_code=200)
            
#         except Exception as e:
#             print(f"动作分析错误 [{motion_type}]: {str(e)}")
#             return JSONResponse({
#                 "message": "ROBUST API ERROR",
#                 "motion": motion_type,
#                 "error": str(e),
#                 **({"pout_count": 0, "total_hold_time": 0.0} if motion_type == "poutLip" 
#                    else {"close_count": 0, "total_close_time": 0.0} if motion_type == "closeLip"
#                    else {}),
#                 "breakpoints": [],
#                 "segments": []
#             }, status_code=500)
        
#     except Exception as e:
#         print(f"API总体错误: {str(e)}")
#         return JSONResponse({
#             "message": "ROBUST API ERROR",
#             "error": str(e)
#         }, status_code=500)

# @app.get("/health")
# async def health_check():
#     return {
#         "status": "healthy", 
#         "version": "robust_dual_v1.0",
#         "algorithm": "original_robust_hysteresis_with_dual_direction",
#         "supported_motions": ["poutLip", "closeLip"]
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)