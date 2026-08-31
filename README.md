# PLOGRID AI Inference Worker ♻️

이미지(YOLO 객체 탐지)와 AS7265x NIR 분광 센서 데이터를 융합하여 재활용 폐기물 종류를 분류하는 Redis Streams 기반 이벤트 드리븐 워커입니다.

## 개요

일반 이미지 기반 객체 탐지만으로는 유리와 투명 플라스틱을 혼동하기 쉽습니다. 이 프로젝트는 YOLO 탐지 결과와 NIR(근적외선) 분광 센서 값을 함께 사용해 재질을 재검증함으로써 분류 정확도를 높입니다.

- **YOLO 모델**: [`do1ng/few_shot2`](https://huggingface.co/do1ng/few_shot2) (Hugging Face, `best.pt`) — 이미지에서 쓰레기 객체를 탐지
- **NIR-YOLO 융합 모델**: [`yeajongcheol/nir-yolo-fusion`](https://huggingface.co/yeajongcheol/nir-yolo-fusion) (Hugging Face, `fusion_model.pt`) — YOLO 클래스 정보와 18채널 NIR 반사율 값을 함께 인코딩해 유리/투명 플라스틱을 2진 분류

## 아키텍처

```
프로듀서(백엔드)
   │  XADD trash-analysis-requests
   ▼
Redis Streams: trash-analysis-requests
   │  XREADGROUP (consumer group: plogrid-ai-workers)
   ▼
PLOGRID AI Worker  ── YOLO 탐지 + NIR 재검증
   │  XADD
   ▼
Redis Streams: trash-analysis-results
   │  XREADGROUP
   ▼
백엔드  ── 결과를 DB에 저장 후 SSE로 클라이언트에 전송
```

HTTP 엔드포인트는 없습니다. 백엔드가 `trash-analysis-requests` 스트림에 요청 메시지를 넣으면, 이 워커가 컨슈머 그룹으로 소비해서 분류를 수행하고 `trash-analysis-results` 스트림에 결과를 발행합니다. 백엔드는 이 결과 스트림을 다시 읽어 DB에 저장하고, SSE로 클라이언트에 전달합니다.

### 요청 스트림 (`trash-analysis-requests`) 필드

| 필드 | 설명 |
|---|---|
| `imageUrl` | 분석할 이미지의 URL |
| `spectralValues` | AS7265x 18채널 반사율 값 (JSON 배열 또는 콤마 구분 문자열) |
| `ploggingId` | 요청 식별자 |
| `latitude`, `longitude` | 위치 정보 |

### 결과 스트림 (`trash-analysis-results`) 필드

| 필드 | 설명 |
|---|---|
| `status` | `success` 또는 `error` |
| `finalLabel` | (성공 시) 최종 분류 라벨 |
| `materialVerified` | (성공 시) NIR 재질 재검증 수행 여부 |
| `labelOverridden` | (성공 시) NIR 결과로 YOLO 라벨이 대체됐는지 여부 |
| `errorMessage` | (실패 시) 오류 메시지 |
| `ploggingId`, `imageUrl`, `latitude`, `longitude` | 요청 메시지에서 그대로 전달되는 필드 |

### 분류 동작 방식

1. `imageUrl`로 이미지를 다운로드해 YOLO로 객체를 탐지하고, 신뢰도가 가장 높은 탐지 하나를 선택
2. 탐지된 클래스가 재질 확인이 필요한 8개 클래스(아래 표)에 해당하면 NIR 센서 값으로 재질을 재검증
   - 기대 재질과 일치하면 YOLO 클래스를 최종 결과로 확정
   - 불일치하면 NIR 결과(유리/투명 플라스틱)를 신뢰하여 최종 결과를 대체
3. 8개 클래스에 해당하지 않으면 NIR 검증 없이 YOLO 결과를 그대로 사용

| 구분 | 클래스 |
|---|---|
| 유리 계열 | 기타술병, 맥주병, 박카스병, 소주병, 음료수병, 주방용기 |
| 투명 플라스틱 계열 | 일회용음료수잔, 페트병 |