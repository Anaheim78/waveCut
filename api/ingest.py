# api/ingest.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json

from . import count_pout_lips
from . import count_sip_lips
from . import count_puff_cheek  # 臉頰

app = FastAPI()

ANALYZERS = {
    "POUT_LIPS":   count_pout_lips,
    "SIP_LIPS":    count_sip_lips,
    "PUFF_CHEEK":  count_puff_cheek,   # 不用 mode，run(req) 即可
    "REDUCE_CHEEK": count_puff_cheek,  # 之後如果要縮臉再在 puff_cheek 內部判斷
}

@app.post("/")
async def ingest(req: Request):
    try:
        raw = await req.body()
        body = json.loads(raw) if raw else {}
        training_type = body.get("trainingType")

        if not training_type or training_type not in ANALYZERS:
            return JSONResponse(
                {"message": "unknown trainingType", "trainingType": training_type},
                status_code=400
            )

        analyzer = ANALYZERS[training_type]
        # 統一呼叫：每個子模組的 run 只收 req
        result = await analyzer.run(req)

        return JSONResponse({"trainingType": training_type, "result": result}, status_code=200)

    except Exception as e:
        return JSONResponse({"message": "INGEST ERROR", "error": str(e)}, status_code=500)

@app.get("/health")
async def health():
    return {"status": "healthy"}
