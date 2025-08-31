# api/ingest.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import io, csv, math
import numpy as np
import pandas as pd
import ruptures as rpt

app = FastAPI()

TARGET_LEN_SEC = 3.0   # 預期每段約 3s
MIN_LEN_SEC    = 3.0   # 最短段長 3s（避免過碎）

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
        motion = "poutLip"

        reader = csv.DictReader(io.StringIO("\n".join(lines)))
        df = pd.DataFrame(reader)
        if "time_seconds" not in df.columns or "height_width_ratio" not in df.columns:
            return JSONResponse({"message": "bad schema", "motion": motion}, status_code=400)

        t = pd.to_numeric(df["time_seconds"], errors="coerce").to_numpy()
        y = pd.to_numeric(df["height_width_ratio"], errors="coerce").to_numpy()
        m = np.isfinite(t) & np.isfinite(y)
        t, y = t[m], y[m]
        n = len(t)
        if n < 2:
            return JSONResponse({
                "message": "APITEST OK", "motion": motion,
                "pout_count": 0, "total_hold_time": 0.0,
                "breakpoints": [], "segments": []
            })

        # 估取樣率
        dt = float(np.median(np.diff(t))) if n > 1 else 0.05
        fs = 1.0 / dt if dt > 0 else 20.0
        total_dur = float(t[-1] - t[0]) if n > 1 else 0.0

        # 預期段數 → 變化點數上限
        min_size   = max(2, int(round(MIN_LEN_SEC * fs)))
        exp_segs   = max(1, int(math.floor(total_dur / max(1e-9, TARGET_LEN_SEC))))
        k_max      = max(0, n // max(2, min_size) - 1)
        n_bkps     = max(0, min(k_max, exp_segs - 1))

        # ★ 安全護欄：資料太短或 n_bkps==0 時，不呼叫 ruptures，直接一段
        bkps = [n]
        if n_bkps > 0 and n >= min_size * 2:
            try:
                algo = rpt.Binseg(model="l2", min_size=min_size).fit(y)
                bkps = algo.predict(n_bkps=n_bkps)  # 斷點（含最後一個 n）
            except Exception:
                bkps = [n]  # 任何錯誤都回退為單段

        # 斷點時間（不含最後 n）
        bp_times = [float(t[min(max(b - 1, 0), n - 1)]) for b in bkps[:-1]]

        # 段落詳情
        segments, s = [], 0
        for i, e in enumerate(bkps):
            st = float(t[s])
            ed = float(t[e - 1]) if e - 1 < n else float(t[-1])
            segments.append({
                "index": i,
                "start_time": round(st, 3),
                "end_time": round(ed, 3),
                "duration": round(ed - st, 3)
            })
            s = e

        # 嘟嘴次數：先扣影片頭一段，再 /2 進位
        pout_count = int(math.ceil(max(0, len(segments) - 1) / 2.0))

        # 維持時間總和：扣頭後，取奇數段（1,3,5…）累加
        total_hold = sum(
            seg["duration"] for idx, seg in enumerate(segments[1:], start=1)
            if idx % 2 == 1
        )

        result = {
            "message": "APITEST OK",
            "motion": motion,  # ← 你要的欄位
            "pout_count": pout_count,
            "total_hold_time": round(float(total_hold), 3),
            "breakpoints": [round(x, 2) for x in bp_times],
            "segments": segments
        }
        print("APITEST result", result)
        return JSONResponse(result, status_code=200)

    # 非嘟嘴
    return JSONResponse({
        "message": "APITEST OK",
        "motion": "unknown",
        "reason": "missing height_width_ratio in header",
        "receivedCount": max(0, len(lines) - 1)
    }, status_code=200)
