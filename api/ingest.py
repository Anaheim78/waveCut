# api/ingest.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json

# 既有模組
from . import count_pout_lips
from . import count_sip_lips

# 新增臉頰
from . import count_puff_cheek  

app = FastAPI()

ANALYZERS = {
    "POUT_LIPS":   count_pout_lips,
    "SIP_LIPS":    count_sip_lips,
    "PUFF_CHEEK":  count_puff_cheek,   # 只做 shape 驗證
    "REDUCE_CHEEK": count_puff_cheek,  # 共用同一支，但進去 run() 要分 mode
}

@app.post("/")
async def ingest(req: Request):
    try:
        raw = await req.body()
        body = json.loads(raw)

        training_type = body.get("trainingType")
        if not training_type or training_type not in ANALYZERS:
            return JSONResponse(
                {"message": "unknown trainingType", "trainingType": training_type},
                status_code=400
            )

        analyzer = ANALYZERS[training_type]
        result = await analyzer.run(req, mode=training_type)  # ★ 加 mode 參數區分 PUFF / REDUCE

        return JSONResponse({"trainingType": training_type, "result": result}, status_code=200)

    except Exception as e:
        return JSONResponse({"message": "INGEST ERROR", "error": str(e)}, status_code=500)

@app.get("/health")
async def health():
    return {"status": "healthy"}
