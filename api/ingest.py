# api/ingest.py
from flask import Flask, request, jsonify
import io, csv, math
import numpy as np
import pandas as pd
import ruptures as rpt

app = Flask(__name__)

TARGET_LEN_SEC = 3.0   # 預期每段約 3s
MIN_LEN_SEC    = 3.0   # 最短段長 3s（避免過碎）

@app.route("/", methods=["POST"])
def ingest():
    data = request.get_json(silent=True) or {}
    lines = data.get("lines") or []
    if not lines or not isinstance(lines, list):
        return jsonify({"message": "no lines", "receivedCount": 0}), 400

    header = lines[0]
    cols = [c.strip() for c in header.split(",")]
    cols_lower = [c.lower() for c in cols]

    # 嘟嘴模式判斷：看 header 是否有 height_width_ratio
    if "height_width_ratio" in cols_lower:
        motion = "poutLip"

        # 讀成 DataFrame
        reader = csv.DictReader(io.StringIO("\n".join(lines)))
        df = pd.DataFrame(reader)

        # 基本欄位
        if "time_seconds" not in df.columns or "height_width_ratio" not in df.columns:
            return jsonify({"message": "bad schema", "motion": motion}), 400

        # 轉數值 & 去 NaN
        t = pd.to_numeric(df["time_seconds"], errors="coerce").to_numpy()
        y = pd.to_numeric(df["height_width_ratio"], errors="coerce").to_numpy()
        m = np.isfinite(t) & np.isfinite(y)
        t, y = t[m], y[m]
        n = len(t)
        if n < 2:
            return jsonify({"message": "not enough data", "motion": motion,
                            "pout_count": 0, "total_hold_time": 0.0, "breakpoints": []})

        # 估取樣率
        dt = float(np.median(np.diff(t))) if n > 1 else 0.05
        fs = 1.0 / dt if dt > 0 else 20.0
        total_dur = float(t[-1] - t[0]) if n > 1 else 0.0

        # 根據預期段長，自動決定變化點數
        min_size = max(2, int(round(MIN_LEN_SEC * fs)))
        exp_segments = max(1, int(math.floor(total_dur / max(1e-9, TARGET_LEN_SEC))))
        k_max = max(0, n // max(2, min_size) - 1)
        n_bkps = max(0, min(k_max, exp_segments - 1))

        # 一維分段（對 height_width_ratio）
        algo = rpt.Binseg(model="l2", min_size=min_size).fit(y)
        bkps = algo.predict(n_bkps=n_bkps)  # 斷點（含最後一個 n）

        # 斷點時間（不含最後的 n）
        bp_times = [float(t[min(max(b - 1, 0), n - 1)]) for b in bkps[:-1]]

        # 段落詳情
        segments = []
        s = 0
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

        # 嘟嘴次數：先扣掉影片頭一段，再 /2 進位
        seg_cnt = len(segments)
        pout_count = int(math.ceil(max(0, seg_cnt - 1) / 2.0))

        # 維持總時間：扣頭後，取奇數段（1,3,5...）累加
        total_hold = 0.0
        for idx, seg in enumerate(segments[1:], start=1):
            if idx % 2 == 1:  # 維持段
                total_hold += seg["duration"]

        result = {
            "message": "APITEST OK",
            "motion": motion,                          # ★ 新增：動作類型
            "pout_count": pout_count,
            "total_hold_time": round(float(total_hold), 3),
            "breakpoints": [round(x, 2) for x in bp_times],
            "segments": segments
        }

        print(f"📥 Received from Android: header={header}")
        print(f"➡️ result: {result}")
        return jsonify(result), 200

    # 非嘟嘴資料
    return jsonify({
        "message": "APITEST OK",
        "motion": "unknown",
        "reason": "missing height_width_ratio in header",
        "receivedCount": max(0, len(lines) - 1)
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
