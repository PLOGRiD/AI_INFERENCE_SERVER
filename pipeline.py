import io
import json

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image

router = APIRouter(tags=["pipeline"])

GLASS_CLASSES = {"기타술병", "맥주병", "박카스병", "소주병", "음료수병", "주방용기"}
PLASTIC_CLASSES = {"일회용음료수잔", "페트병"}
MATERIAL_CHECK_CLASSES = GLASS_CLASSES | PLASTIC_CLASSES


@router.post("/predict", summary="이미지+NIR 통합 파이프라인 추론")
async def predict(
    request: Request,
    file: UploadFile = File(..., description="탐지할 이미지 파일"),
    nir: str = Form(..., description="AS7265x 18채널 반사율 값. JSON 배열(\"[0.04,0.05,...]\") 또는 콤마 구분 문자열(\"0.04,0.05,...\") 둘 다 허용"),
):
    """
    이미지로 YOLO 객체 탐지를 수행하고, 최고 신뢰도 검출 클래스가 유리/투명플라스틱
    판별이 필요한 다음 8개 병·용기 클래스 중 하나이면 NIR 데이터로 재질을 재검증하여
    최종 결과를 확정한다.

    - 기타술병
    - 맥주병
    - 박카스병
    - 소주병
    - 음료수병
    - 주방용기
    - 일회용음료수잔
    - 페트병

    ---

    - 위 8개 클래스가 아니면 NIR 검증 없이 YOLO 결과를 그대로 최종 결과로 사용한다.
    - 8개 클래스 중 하나이면 NIR 결과(유리/투명 플라스틱)와 대조한다.
      - 기대 재질과 일치하면 YOLO 클래스를 최종 결과로 확정한다.
      - 불일치하면 NIR을 신뢰하여 최종 결과를 재질명(유리/투명 플라스틱)으로 대체한다.
        (예: YOLO가 "소주병"으로 검출했으나 NIR이 "투명 플라스틱"이면 최종 결과는 "투명 플라스틱")
    """
    nir_raw = nir.strip()
    try:
        nir_values = json.loads(nir_raw)
    except json.JSONDecodeError:
        try:
            nir_values = [float(v) for v in nir_raw.strip("[]").split(",")]
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"nir은 JSON 배열 또는 콤마로 구분된 숫자 문자열이어야 함 (받은 값: {nir_raw[:80]!r})",
            )

    yolo_predictor = request.app.state.yolo_predictor
    nir_predictor = request.app.state.nir_predictor
    if yolo_predictor is None or nir_predictor is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않음")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Image file required")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    detections = yolo_predictor.predict(image)

    if not detections:
        raise HTTPException(status_code=422, detail="탐지된 객체가 없음")

    top_detection = max(detections, key=lambda d: d["confidence"])
    yolo_class = top_detection["class_name"]

    nir_result = None
    material_verified = False
    label_overridden = False
    final_label = yolo_class

    if yolo_class in MATERIAL_CHECK_CLASSES:
        material_verified = True
        try:
            nir_result = nir_predictor.predict(
                nir=nir_values, yolo_class=yolo_class, yolo_conf=top_detection["confidence"]
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        nir_label = nir_result["label"]
        expected_glass = yolo_class in GLASS_CLASSES and nir_label == "유리"
        expected_plastic = yolo_class in PLASTIC_CLASSES and nir_label == "투명 플라스틱"

        if not (expected_glass or expected_plastic):
            final_label = nir_label
            label_overridden = True

    return {
        "final_label": final_label,
        "material_verified": material_verified,
        "label_overridden": label_overridden,
    }
