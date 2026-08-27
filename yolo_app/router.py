import io
import time

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

router = APIRouter(prefix="/yolo", tags=["yolo"])


@router.post("/predict", summary="이미지 객체 탐지")
async def predict(request: Request, file: UploadFile = File(...)):
    """
    이미지 파일을 업로드받아 YOLO 모델로 객체 탐지를 수행한다.

    - **file**: 업로드할 이미지 파일 (image/* 타입만 허용)
    - 응답에는 탐지된 객체별 클래스, 신뢰도(confidence), 바운딩 박스 좌표가 포함된다.
    """
    predictor = request.app.state.yolo_predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Image file required")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    detections = predictor.predict(image)

    payload = {
        "filename": file.filename,
        "image_size": {"width": image.width, "height": image.height},
        "num_detections": len(detections),
        "detections": detections,
        "timestamp": time.time(),
    }

    return JSONResponse(payload)
