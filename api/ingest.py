# api/ingest.py
# FastAPI — 改进版嘟嘴检测：更合理的阈值策略 + 简化段落处理 + 直观计数
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import io, csv, math
import numpy as np
import pandas as pd

app = FastAPI()

# ===== 改进版核心参数 =====
class DetectionConfig:
    # 信号预处理
    SMOOTH_WINDOW_SEC = 0.1      # 移动平均窗口
    EXP_SMOOTH_ALPHA = 0.3       # 指数平滑系数
    
    # 阈值策略
    BASELINE_PERCENTILE = 30     # 基线百分位数
    ACTIVATION_PERCENTILE = 75   # 激活百分位数
    NOISE_MULTIPLIER = 2.5       # 噪声倍数
    HYSTERESIS_RATIO = 0.7       # 回滞比例
    
    # 段落处理
    MIN_POUT_DURATION = 0.3      # 最小嘟嘴持续时间(秒)
    MAX_GAP_MERGE = 0.5          # 最大合并间隔(秒)
    PADDING_SEC = 0.1            # 段落边缘扩展(秒)

# ===== 核心工具函数 =====
def _estimate_fs(t: np.ndarray) -> float:
    """估算采样率"""
    if len(t) < 2:
        return 20.0
    dt = float(np.median(np.diff(t)))
    return 1.0 / dt if dt > 1e-9 else 20.0

def _moving_average(x: np.ndarray, window_size: int) -> np.ndarray:
    """移动平均滤波"""
    if window_size <= 1:
        return x.copy()
    kernel = np.ones(int(window_size), dtype=float) / float(window_size)
    return np.convolve(x, kernel, mode="same")

def _exp_smooth(x: np.ndarray, alpha: float) -> np.ndarray:
    """指数平滑滤波"""
    if alpha <= 0:
        return x.copy()
    result = np.zeros_like(x, dtype=float)
    result[0] = x[0]
    for i in range(1, len(x)):
        result[i] = alpha * x[i] + (1 - alpha) * result[i-1]
    return result

def _preprocess_signal(signal: np.ndarray, fs: float) -> np.ndarray:
    """信号预处理：移动平均 + 指数平滑"""
    # 移动平均去噪
    window_size = max(1, int(DetectionConfig.SMOOTH_WINDOW_SEC * fs))
    smoothed = _moving_average(signal, window_size)
    
    # 指数平滑稳定信号
    processed = _exp_smooth(smoothed, DetectionConfig.EXP_SMOOTH_ALPHA)
    return processed

def _calculate_adaptive_thresholds(signal: np.ndarray) -> dict:
    """计算自适应阈值"""
    # 计算基线和激活水平
    baseline = np.percentile(signal, DetectionConfig.BASELINE_PERCENTILE)
    activation = np.percentile(signal, DetectionConfig.ACTIVATION_PERCENTILE)
    
    # 估算噪声水平 (MAD方法)
    median_val = np.median(signal)
    mad = np.median(np.abs(signal - median_val))
    noise_std = 1.4826 * mad  # MAD到标准差转换
    
    # 动态阈值计算
    signal_range = activation - baseline
    if signal_range > DetectionConfig.NOISE_MULTIPLIER * noise_std:
        # 信号足够强，使用百分位数阈值
        high_thresh = baseline + 0.6 * signal_range
    else:
        # 信号较弱，使用噪声基础阈值
        high_thresh = baseline + DetectionConfig.NOISE_MULTIPLIER * noise_std
    
    # 回滞阈值
    low_thresh = baseline + DetectionConfig.HYSTERESIS_RATIO * (high_thresh - baseline)
    
    return {
        'high_threshold': high_thresh,
        'low_threshold': low_thresh,
        'baseline': baseline,
        'noise_std': noise_std
    }

def _detect_pout_segments(signal: np.ndarray, thresholds: dict) -> list:
    """使用回滞检测嘟嘴段落"""
    high_thresh = thresholds['high_threshold']
    low_thresh = thresholds['low_threshold']
    
    segments = []
    in_pout = False
    start_idx = 0
    
    for i, value in enumerate(signal):
        if not in_pout:
            # 寻找嘟嘴开始
            if value >= high_thresh:
                in_pout = True
                start_idx = i
        else:
            # 寻找嘟嘴结束
            if value < low_thresh:
                segments.append((start_idx, i))
                in_pout = False
    
    # 处理结尾仍在嘟嘴状态
    if in_pout:
        segments.append((start_idx, len(signal) - 1))
    
    return segments

def _merge_close_segments(segments: list, timestamps: np.ndarray, max_gap_sec: float) -> list:
    """合并相近的段落"""
    if len(segments) < 2:
        return segments
    
    merged = []
    current_start, current_end = segments[0]
    
    for start, end in segments[1:]:
        gap_time = timestamps[start] - timestamps[current_end]
        
        if gap_time <= max_gap_sec:
            # 合并段落
            current_end = end
        else:
            # 保存当前段落，开始新段落
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    
    # 添加最后一个段落
    merged.append((current_start, current_end))
    return merged

def _filter_by_duration(segments: list, timestamps: np.ndarray, min_duration_sec: float) -> list:
    """过滤太短的段落"""
    filtered = []
    for start, end in segments:
        duration = timestamps[end] - timestamps[start]
        if duration >= min_duration_sec:
            filtered.append((start, end))
    return filtered

def _apply_padding(segments: list, signal_length: int, fs: float, padding_sec: float) -> list:
    """为段落添加边缘扩展"""
    padding_samples = int(padding_sec * fs)
    padded = []
    
    for start, end in segments:
        new_start = max(0, start - padding_samples)
        new_end = min(signal_length - 1, end + padding_samples)
        padded.append((new_start, new_end))
    
    return padded

def _segments_to_results(segments: list, timestamps: np.ndarray) -> list:
    """将段落索引转换为结果格式"""
    results = []
    for i, (start, end) in enumerate(segments):
        start_time = timestamps[start]
        end_time = timestamps[end]
        duration = end_time - start_time
        
        results.append({
            "index": i,
            "start_time": round(float(start_time), 3),
            "end_time": round(float(end_time), 3),
            "duration": round(float(duration), 3)
        })
    
    return results

# ===== 主分析函数 =====
def analyze_improved_pout(t: np.ndarray, r: np.ndarray) -> dict:
    """改进版嘟嘴分析算法"""
    # 估算采样率
    fs = _estimate_fs(t)
    
    # 信号预处理
    processed_signal = _preprocess_signal(r, fs)
    
    # 计算自适应阈值
    thresholds = _calculate_adaptive_thresholds(processed_signal)
    
    # 检测初步段落
    raw_segments = _detect_pout_segments(processed_signal, thresholds)
    
    # 如果完全检测不到，放宽阈值再试一次
    if len(raw_segments) == 0:
        # 降低高阈值15%
        relaxed_high = thresholds['baseline'] + 0.85 * (thresholds['high_threshold'] - thresholds['baseline'])
        relaxed_low = thresholds['baseline'] + DetectionConfig.HYSTERESIS_RATIO * (relaxed_high - thresholds['baseline'])
        relaxed_thresholds = {
            'high_threshold': relaxed_high,
            'low_threshold': relaxed_low,
            'baseline': thresholds['baseline'],
            'noise_std': thresholds['noise_std']
        }
        raw_segments = _detect_pout_segments(processed_signal, relaxed_thresholds)
        thresholds = relaxed_thresholds  # 更新用于输出的阈值
    
    # 段落后处理流程
    # 1. 合并相近段落
    merged_segments = _merge_close_segments(raw_segments, t, DetectionConfig.MAX_GAP_MERGE)
    
    # 2. 过滤短段落
    filtered_segments = _filter_by_duration(merged_segments, t, DetectionConfig.MIN_POUT_DURATION)
    
    # 3. 添加边缘扩展
    final_segments = _apply_padding(filtered_segments, len(processed_signal), fs, DetectionConfig.PADDING_SEC)
    
    # 转换为输出格式
    segments = _segments_to_results(final_segments, t)
    
    # 计算指标
    pout_count = len(segments)
    total_hold_time = sum(seg["duration"] for seg in segments)
    breakpoints = [round(seg["end_time"], 2) for seg in segments]
    
    # 添加调试信息（用于日志）
    print(f"🔍 改进版检测结果: {pout_count}次嘟嘴, 总时长{total_hold_time:.2f}s")
    print(f"📊 阈值: 高={thresholds['high_threshold']:.3f}, 低={thresholds['low_threshold']:.3f}")
    print(f"🎯 原始段落: {len(raw_segments)}, 最终段落: {len(final_segments)}")
    
    return {
        "motion": "poutLip",
        "pout_count": pout_count,
        "total_hold_time": round(float(total_hold_time), 3),
        "breakpoints": breakpoints,
        "segments": segments
    }

# ===== FastAPI路由 =====
@app.post("/")
async def ingest(req: Request):
    try:
        body = await req.json()
        lines = body.get("lines") or []
        
        if not lines or not isinstance(lines, list):
            return JSONResponse({"message": "no lines", "receivedCount": 0}, status_code=400)

        header = lines[0]
        cols_lower = [c.strip().lower() for c in header.split(",")]

        # 检查是否是嘟嘴数据
        if "height_width_ratio" in cols_lower:
            try:
                # 转换为DataFrame
                reader = csv.DictReader(io.StringIO("\n".join(lines)))
                df = pd.DataFrame(reader)

                # 建立列名映射（容错大小写和空白）
                lowmap = {c.strip().lower(): c for c in df.columns}
                
                if "time_seconds" not in lowmap or "height_width_ratio" not in lowmap:
                    return JSONResponse({
                        "message": "IMPROVED API OK", 
                        "motion": "poutLip",
                        "error": "missing required columns"
                    }, status_code=400)

                # 提取并清理数据
                t = pd.to_numeric(df[lowmap["time_seconds"]], errors="coerce").to_numpy()
                r = pd.to_numeric(df[lowmap["height_width_ratio"]], errors="coerce").to_numpy()
                
                # 过滤有效数据
                valid_mask = np.isfinite(t) & np.isfinite(r)
                t, r = t[valid_mask], r[valid_mask]

                if len(t) < 2:
                    return JSONResponse({
                        "message": "IMPROVED API OK",
                        "motion": "poutLip",
                        "pout_count": 0,
                        "total_hold_time": 0.0,
                        "breakpoints": [],
                        "segments": []
                    }, status_code=200)

                # 使用改进版算法分析
                result_core = analyze_improved_pout(t, r)
                result = {"message": "IMPROVED API OK", **result_core}

                # 输出到Railway日志便于调试
                print("✅ 改进版API结果:", result)
                
                return JSONResponse(result, status_code=200)
                
            except Exception as e:
                print(f"❌ 嘟嘴分析错误: {str(e)}")
                return JSONResponse({
                    "message": "IMPROVED API ERROR",
                    "motion": "poutLip",
                    "error": str(e),
                    "pout_count": 0,
                    "total_hold_time": 0.0,
                    "breakpoints": [],
                    "segments": []
                }, status_code=500)

        # 非嘟嘴数据
        return JSONResponse({
            "message": "IMPROVED API OK",
            "motion": "unknown",
            "reason": "missing height_width_ratio in header",
            "receivedCount": max(0, len(lines) - 1)
        }, status_code=200)
        
    except Exception as e:
        print(f"❌ API总体错误: {str(e)}")
        return JSONResponse({
            "message": "IMPROVED API ERROR",
            "error": str(e)
        }, status_code=500)

# ===== 健康检查路由 =====
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "improved_v1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)