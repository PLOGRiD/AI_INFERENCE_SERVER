from fastapi import APIRouter, HTTPException, Request

from nir_app.schemas import (
    PredictRequest,
    PredictResponse,
)

router = APIRouter(prefix="/nir", tags=["nir"])


@router.post("/predict", response_model=PredictResponse, summary="NIR+YOLO 융합 유리/투명플라스틱 분류")
def predict(req: PredictRequest, request: Request):
    """
    AS7265x 센서 18채널 NIR 값과 YOLO 검출 결과를 융합하여 유리/투명 플라스틱을 분류한다.

    - **nir**: 파장 오름차순 18채널 반사율 값 (행 합 1로 정규화)
    - **yolo_scores**: YOLO 26개 클래스 신뢰도 벡터 (yolo_class 대신 사용 가능)
    - **yolo_class** / **yolo_conf**: YOLO 검출 클래스명과 신뢰도 (yolo_scores 대신 사용 가능)
    """
    try:
        result = request.app.state.nir_predictor.predict(
            nir=req.nir,
            yolo_scores=req.yolo_scores,
            yolo_class=req.yolo_class,
            yolo_conf=req.yolo_conf,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return PredictResponse(**result)
