import json

from PIL import Image

GLASS_CLASSES = {"기타술병", "맥주병", "박카스병", "소주병", "음료수병", "주방용기"}
PLASTIC_CLASSES = {"일회용음료수잔", "페트병"}
MATERIAL_CHECK_CLASSES = GLASS_CLASSES | PLASTIC_CLASSES

NIR_CHANNELS = 18


class PipelineError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def parse_nir_values(raw: str) -> list[float]:
    """JSON 배열 또는 콤마로 구분된 문자열을 18채널 float 리스트로 변환한다."""
    raw = raw.strip()
    try:
        values = json.loads(raw)
    except json.JSONDecodeError:
        try:
            values = [float(v) for v in raw.strip("[]").split(",")]
        except ValueError:
            raise PipelineError(f"nir은 JSON 배열 또는 콤마로 구분된 숫자 문자열이어야 함 (받은 값: {raw[:80]!r})")

    if len(values) != NIR_CHANNELS:
        raise PipelineError(f"nir은 {NIR_CHANNELS}채널이어야 함 (받은 개수: {len(values)})")

    return values


def classify_material(image: Image.Image, nir_values: list[float], yolo_predictor, nir_predictor) -> dict:
    """
    이미지로 YOLO 객체 탐지를 수행하고, 최고 신뢰도 검출 클래스가 유리/투명플라스틱
    판별이 필요한 8개 병·용기 클래스 중 하나이면 NIR 데이터로 재질을 재검증하여
    최종 결과를 확정한다.
    
    HTTP 엔드포인트와 Redis Streams 컨슈머가 로직을 공유한다.
    """
    detections = yolo_predictor.predict(image)
    if not detections:
        raise PipelineError("탐지된 객체가 없음", status_code=422)

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
            raise PipelineError(str(e))

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
        "yolo_class": yolo_class,
        "yolo_confidence": top_detection["confidence"],
        "nir_result": nir_result,
    }
