from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json

# 只綁 POUT_LIPS → count_pout_lips（兩檔可跑）
from . import count_pout_lips

app = FastAPI()

ANALYZERS = {
    "POUT_LIPS": count_pout_lips,   # 之後要擴充再加其他模組
}

@app.post("/")
async def ingest(req: Request):
    try:
        # 讀原始 body（會被 Starlette 快取，下游還能再次 await req.json()）
        raw = await req.body()
        body = json.loads(raw)

        training_type = body.get("trainingType")
        if not training_type or training_type not in ANALYZERS:
            return JSONResponse(
                {"message": "unknown trainingType", "trainingType": training_type},
                status_code=400
            )

        analyzer = ANALYZERS[training_type]
        # 把「同一個 Request」原封不動丟給子程式處理
        result = await analyzer.run(req)

        return JSONResponse({"trainingType": training_type, "result": result}, status_code=200)

    except Exception as e:
        return JSONResponse({"message": "INGEST ERROR", "error": str(e)}, status_code=500)
