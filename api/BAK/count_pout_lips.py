# api/ingest.py

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import io, csv, math
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

app = FastAPI()
fs: float = 30.0
cutoff: float = 0.8
order: int = 4

# 輸入格式
#{
# "trainingType":"POUT_LIPS",
# "lines":[
# 	"time_seconds, state, mouth_height, mouth_width, height_width_ratio",
# 	"1.010,CALIBRATING,827.640,376.379,2.199",
# 	"1.220,CALIBRATING,787.822,360.203,2.187",
# 	"1.282,CALIBRATING,783.903,362.270,2.164","

def run(df, training_type=None):
    return {"status": "OK", "module": "count_pout_lips", "rows": len(df)}


# ===== 低通濾波器 =====
def lowpass_filter(x, fs=30.0, cutoff=0.5, order=4):
    b, a = butter(order, cutoff / (fs / 2), btype='low')
    y = filtfilt(b, a, x)
    print(f"[DEBUG] Lowpass: cutoff={cutoff} Hz, order={order}, fs={fs}")
    return y

# ===== 移動平均 (基線估計) =====
def moving_average(x, win_samples=60):
    kernel = np.ones(win_samples) / win_samples
    pad_width = win_samples // 2
    x_padded = np.pad(x, pad_width, mode='edge')
    baseline_full = np.convolve(x_padded, kernel, mode='same')
    baseline = baseline_full[pad_width:-pad_width]
    return baseline

# ===== 零交叉檢測 =====
def zero_crossings(x, t, deadband=0.0, min_interval=10):
    crossings_all = []
    crossings_up = []    # 負→正
    crossings_down = []  # 正→負

    last_idx = -min_interval
    for i in range(1, len(x)):
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

@app.post("/")
async def ingest(req: Request):
    
    try:
        body = await req.json()
        lines = body.get("lines") or []
        if not lines or not isinstance(lines, list):
            return JSONResponse({"message": "no lines", "receivedCount": 0}, status_code=400)
        
        header = lines[0]

        try:
            # 解析CSV数据
            reader = csv.DictReader(io.StringIO("\n".join(lines)))
            df = pd.DataFrame(reader)
  
            # 取得列名映射（忽略大小写和空格）
            lowmap = {}
            for c in df.columns:
                    clean_col = str(c).strip().lower()
                    lowmap[clean_col] = c
            
            # 欄位檢查
            if "time_seconds" not in lowmap:
                raise ValueError("Missing time_seconds column")
            if "height_width_ratio" not in lowmap:
                raise ValueError("Missing height_width_ratio column for pout detection")
            
            # 讀取數據
            timestamps = pd.to_numeric(df[lowmap["time_seconds"]], errors="coerce").to_numpy()
            values = pd.to_numeric(df[lowmap["height_width_ratio"]], errors="coerce").to_numpy()
        
            m = np.isfinite(timestamps) & np.isfinite(values)
            t, r = t[m], r[m]

            # 低通
            r_filt = lowpass_filter(r, fs=fs, cutoff=cutoff, order=order)

            # 基線扣除
            win = int(4.0 * fs)
            baseline = moving_average(r_filt, win)
            r_detrend = r_filt - baseline

            # 零交叉 (回傳 index)
            zc_all, zc_up, zc_down = zero_crossings(
                r_detrend, t,
                deadband=0.005*np.std(r_detrend),
                min_interval=int(0.5*fs)
            )

            # 建立 segments 結果 (全段都保留)
            segments = []
            positive_segments = []  # 只收 >0 的段落
            for i, (s, e) in enumerate(zip(zc_all[:-1], zc_all[1:])):
                st, ed = t[s], t[e]
                dur = round(float(ed - st), 3)
                seg = {
                    "index": i,
                    "start_time": round(float(st), 3),
                    "end_time": round(float(ed), 3),
                    "duration": dur
                }
                segments.append(seg)

                # 判斷起點是不是在 >0 區域
                if r_detrend[s] >= 0:
                    positive_segments.append(seg)

                # breakpoints → 全部段落的結束時間
                breakpoints = [seg["end_time"] for seg in segments]

                # API response
                return {
                    "action_count": len(positive_segments),  # 只算正區段
                    "total_action_time": sum(seg["duration"] for seg in positive_segments),
                    "breakpoints": breakpoints,              # 全部段落
                    "segments": segments,                    # 全部段落
                    "debug": {
                        "fs_hz": fs,
                        "cutoff": cutoff,
                        "order": order,
                        "zc_all": len(zc_all),
                        "zc_up": len(zc_up),
                        "zc_down": len(zc_down),
                    }
                }


        except Exception as e:
            print(f"动作分析错误 [{header}]: {str(e)}")
            return JSONResponse({
                "message": "ERROR when parsing CSV",
                "motion": header,
                "error": str(e),
                # **({"pout_count": 0, "total_hold_time": 0.0} 
                #    if motion_type == "poutLip" 
                #    else {"close_count": 0, "total_close_time": 0.0} 
                #    if motion_type == "closeLip"
                #    else {}),
                "breakpoints": [],
                "segments": []
            }, status_code=500)


    except Exception as e:
        print(f"ERROR API : {str(e)}")
        return JSONResponse({
            "message": "ROBUST API ERROR",
            "error": str(e)
        }, status_code=500)