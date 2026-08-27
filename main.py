import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from nir_app.inference import NIRPredictor
from nir_app.router import router as nir_router
from pipeline import router as pipeline_router
from yolo_app.predictor import YOLOPredictor
from yolo_app.router import router as yolo_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("plogrid-ai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading YOLO model...")
    app.state.yolo_predictor = YOLOPredictor()

    logger.info("Loading NIR model...")
    app.state.nir_predictor = NIRPredictor()

    logger.info("All models loaded.")
    yield

    app.state.yolo_predictor = None
    app.state.nir_predictor = None


app = FastAPI(title="PLOGRID AI Inference API", version="1.0.0", lifespan=lifespan)

app.include_router(pipeline_router)
app.include_router(yolo_router)
app.include_router(nir_router)

@app.get("/health", summary="전체 서버 상태 확인")
async def health():
    """서버 및 YOLO/NIR 두 모델이 모두 정상적으로 로드되었는지 한 번에 확인한다."""
    return {
        "status": "ok",
        "yolo_loaded": app.state.yolo_predictor is not None,
        "nir_loaded": app.state.nir_predictor is not None,
    }
