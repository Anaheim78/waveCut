# api/count_puff_cheek.py
from fastapi import Request
import io, csv
import numpy as np
import pandas as pd

# 兩組臉頰點
LEFT_CHEEK_IDXS  = [117,118,101,36,203,212,214,192,147,123,98,97,164,0,37,39,40,186]
RIGHT_CHEEK_IDXS = [164,0,267,269,270,410,423,327,326,432,434,416,376,352,346,347,330,266]

def _row_points3d(row, idxs):
    pts = []
    for k in idxs:
        px = float(row[f"point{k}_x"])
        py = float(row[f"point{k}_y"])
        pz = float(row[f"point{k}_z"])
        pts.append([px, py, pz])
    return np.asarray(pts, dtype=float)  # (N, 3)

async def run(req: Request, mode="PUFF"):
    """
    只解析 payload，檢查第一列 row 組出 P_L / P_R，印出 shape，並回傳 shape。
    """
    body = await req.json()
    lines = body.get("lines") or []
    if not lines or not isinstance(lines, list):
        print("[shapecheck] no lines")
        return {"status": "ERROR", "error": "no lines"}

    # 轉 DataFrame
    df = pd.DataFrame(csv.DictReader(io.StringIO("\n".join(lines))))
    print(f"[shapecheck] rows={len(df)}, cols={len(df.columns)}")

    # 檢查必要欄位
    need = [f"point{k}_{ax}" for k in LEFT_CHEEK_IDXS+RIGHT_CHEEK_IDXS for ax in ("x","y","z")]
    need = ["time_seconds", "state"] + need + ["img_w", "img_h"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        print(f"[shapecheck] missing columns: {missing[:6]} ... total {len(missing)}")
        return {"status": "ERROR", "error": f"missing columns ({len(missing)})", "missing_preview": missing[:10]}

    if len(df) == 0:
        return {"status": "ERROR", "error": "no data rows"}

    # 只拿第一列做 shape 檢查
    row0 = df.iloc[0]
    P_L = _row_points3d(row0, LEFT_CHEEK_IDXS)   # (18, 3)
    P_R = _row_points3d(row0, RIGHT_CHEEK_IDXS)  # (18, 3)

    print(f"[shapecheck] P_L shape: {P_L.shape}")
    print(f"[shapecheck] P_R shape: {P_R.shape}")
    # 若想快速肉眼確認，印一點點數值：
    print(f"[shapecheck] P_L sample first 2: {P_L[:2].tolist()}")
    print(f"[shapecheck] P_R sample first 2: {P_R[:2].tolist()}")

    return {
        "status": "OK",
        "mode": mode,
        "rows": len(df),
        "cols": len(df.columns),
        "left_shape": list(P_L.shape),
        "right_shape": list(P_R.shape),
    }
