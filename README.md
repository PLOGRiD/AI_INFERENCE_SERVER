# PLOGRID AI Inference API ♻️

이미지(YOLO 객체 탐지)와 AS7265x NIR 분광 센서 데이터를 융합하여 재활용 폐기물 종류를 분류하는 FastAPI 기반 추론 서버입니다.

## 개요

일반 이미지 기반 객체 탐지만으로는 유리와 투명 플라스틱을 혼동하기 쉽습니다. 이 프로젝트는 YOLO 탐지 결과와 NIR(근적외선) 분광 센서 값을 함께 사용해 재질을 재검증함으로써 분류 정확도를 높입니다.

- **YOLO 모델**: [`do1ng/few_shot2`](https://huggingface.co/do1ng/few_shot2) (Hugging Face, `best.pt`) — 이미지에서 쓰레기 객체를 탐지
- **NIR-YOLO 융합 모델**: [`yeajongcheol/nir-yolo-fusion`](https://huggingface.co/yeajongcheol/nir-yolo-fusion) (Hugging Face, `fusion_model.pt`) — YOLO 클래스 정보와 18채널 NIR 반사율 값을 함께 인코딩해 유리/투명 플라스틱을 2진 분류

## 아키텍처

```
클라이언트
   │
   ├─ POST /predict           ── 이미지 + NIR 통합 파이프라인
   ├─ POST /yolo/predict      ── YOLO 객체 탐지만 단독 호출
   ├─ POST /nir/predict       ── NIR+YOLO 융합 분류만 단독 호출
   └─ GET  /health            ── 서버 및 모델 로드 상태 확인
```

### 통합 파이프라인 (`POST /predict`) 동작 방식

1. 업로드된 이미지에서 YOLO로 객체를 탐지하고, 신뢰도가 가장 높은 탐지 하나를 선택
2. 탐지된 클래스가 재질 확인이 필요한 8개 클래스(아래 표)에 해당하면 NIR 센서 값으로 재질을 재검증
   - 기대 재질과 일치하면 YOLO 클래스를 최종 결과로 확정
   - 불일치하면 NIR 결과(유리/투명 플라스틱)를 신뢰하여 최종 결과를 대체
3. 8개 클래스에 해당하지 않으면 NIR 검증 없이 YOLO 결과를 그대로 사용

| 구분 | 클래스 |
|---|---|
| 유리 계열 | 기타술병, 맥주병, 박카스병, 소주병, 음료수병, 주방용기 |
| 투명 플라스틱 계열 | 일회용음료수잔, 페트병 |

## 설치 및 실행

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

uvicorn main:app --reload
```

### Docker로 실행

```bash
docker compose up --build
```

`docker-compose.yml`에서 모델 캐시(`./models`)를 컨테이너에 볼륨 마운트해두므로, 컨테이너를 새로 띄워도 모델을 다시 다운로드하지 않습니다.

### 환경 변수

| 변수 | 설명 | 기본값 |
|---|---|---|
| `NIR_MODEL_DIR` | NIR 융합 모델(`fusion_model.pt`) 다운로드/캐시 디렉터리 | `<프로젝트 루트>/models` |
| `YOLO_MODEL_DIR` | YOLO 탐지 모델(`best.pt`) 다운로드/캐시 디렉터리 | `<프로젝트 루트>/models` |

## API 요약

### `POST /predict` — 이미지+NIR 통합 파이프라인
- **입력**: `file` (이미지 파일, multipart), `nir` (18채널 반사율 값, JSON 배열 또는 콤마 구분 문자열)
- **출력**: `final_label`, `material_verified`, `label_overridden`

### `POST /yolo/predict` — YOLO 객체 탐지
- **입력**: `file` (이미지 파일)
- **출력**: 탐지된 객체별 `class_id`, `class_name`, `confidence`, `bbox` 리스트

### `POST /nir/predict` — NIR+YOLO 융합 분류
- **입력**: `nir`(18채널 반사율), `yolo_scores`(26클래스 신뢰도 벡터) 또는 `yolo_class`+`yolo_conf` 중 하나
- **출력**: `label`, `confidence`, `prob`, `z_rgb`, `z_n`, `gate`, `yolo_class`, `yolo_in_scope`

### `GET /health` — 서버 상태 확인
- **출력**: `status`, `yolo_loaded`, `nir_loaded`
