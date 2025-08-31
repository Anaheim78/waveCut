# api/ingest.py
from fastapi import FastAPI, Request
import io, csv, math
import numpy as np
import pandas as pd
import ruptures as rpt

app = FastAPI()

TARGET_LEN_SEC = 3.0
MIN_LEN_SEC    = 3.0

@app.post("/")
async def ingest(request: Request):
    data = await request.json()
    lines = data.get("lines") or []
    if not lines or not isinstance(lines, list):
        return {"message": "no lines", "receivedCount": 0}

    header = lines[0]
    cols_lower = [c.strip().lower() for c in header.split(",")]

    if "height_width_ratio" in cols_lower:
        motion = "poutLip"

        reader = csv.DictReader(io.StringIO("\n".join(lines)))
        df = pd.DataFrame(reader)

        if "time_seconds" not in df.columns or "height_width_ratio" not in df.columns:
            return {"message": "bad schema", "motion": motion}

        t = pd.to_numeric(df["time_seconds"], errors="coerce").to_numpy()
        y = pd.to_numeric(df["height_width_ratio"], errors="coerce").to_numpy()
        m = np.isfinite(t) & np.isfinite(y)
        t, y = t[m], y[m]
        n = len(t)
        if n < 2:
            return {"message": "not enough data", "motion": motion,
                    "pout_count": 0, "total_hold_time": 0.0, "breakpoints": []}

        dt = float(np.median(np.diff(t))) if n > 1 else 0.05
        fs = 1.0/dt if dt > 0 else 20.0
        total_dur = float(t[-1]-t[0]) if n > 1 else 0.0

        min_size = max(2, int(round(MIN_LEN_SEC * fs)))
        exp_segments = max(1, int(math.floor(total_dur / max(1e-9, TARGET_LEN_SEC))))
        k_max = max(0, n // max(2, min_size) - 1)
        n_bkps = max(0, min(k_max, exp_segments - 1))

        algo = rpt.Binseg(model="l2", min_size=min_size).fit(y)
        bkps = algo.predict(n_bkps=n_bkps)

        bp_times = [float(t[min(max(b-1,0), n-1)]) for b in bkps[:-1]]

        segments, s = [], 0
        for i, e in enumerate(bkps):
            st = float(t[s]); ed = float(t[e-1]) if e-1 < n else float(t[-1])
            segments.append({
                "index": i,
                "start_time": round(st, 3),
                "end_time": round(ed, 3),
                "duration": round(ed - st, 3)
            })
            s = e

        pout_count = int(math.ceil(max(0, len(segments)-1)/2.0))
        total_hold = sum(seg["duration"] for idx, seg in enumerate(segments[1:], start=1) if idx % 2 == 1)

        return {
            "message": "APITEST OK",
            "motion": motion,
            "pout_count": pout_count,
            "total_hold_time": round(float(total_hold), 3),
            "breakpoints": [round(x, 2) for x in bp_times],
            "segments": segments
        }

    return {
        "message": "APITEST OK",
        "motion": "unknown",
        "reason": "missing height_width_ratio in header",
        "receivedCount": max(0, len(lines)-1)
    }
