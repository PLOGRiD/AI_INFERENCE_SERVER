from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

N_NIR_CHANNELS = 18
N_YOLO_CLASSES = 26


class PredictRequest(BaseModel):
    nir: List[float] = Field(
        ...,
        description="AS7265x 센서 18채널 반사율 값 (파장 오름차순, 행 합 1로 정규화)",
        min_length=N_NIR_CHANNELS,
        max_length=N_NIR_CHANNELS,
    )
    yolo_scores: Optional[List[float]] = Field(
        None, description="YOLO 26개 클래스 신뢰도 벡터 (yolo_class 대신 사용 가능)"
    )
    yolo_class: Optional[str] = Field(None, description="YOLO 검출 클래스명 (yolo_scores 대신 사용 가능)")
    yolo_conf: Optional[float] = Field(None, description="YOLO 검출 신뢰도 (yolo_class와 함께 사용)")

    @model_validator(mode="after")
    def check_yolo_input(self):
        if self.yolo_scores is None and self.yolo_class is None:
            raise ValueError("yolo_scores 또는 yolo_class 중 하나는 필요함")
        if self.yolo_scores is not None and len(self.yolo_scores) != N_YOLO_CLASSES:
            raise ValueError(f"yolo_scores는 {N_YOLO_CLASSES}차원이어야 함")
        return self


class PredictResponse(BaseModel):
    label: str
    confidence: float
    prob: Dict[str, float]
    z_rgb: Dict[str, float]
    z_n: List[float]
    gate: float
    yolo_class: Optional[str] = None
    yolo_in_scope: bool
