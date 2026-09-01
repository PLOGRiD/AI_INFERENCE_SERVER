import asyncio
import io
import logging
import os

import httpx
from PIL import Image
from redis import asyncio as aioredis

from pipeline import PipelineError, classify_material, parse_nir_values
from stream_app.producer import ResultProducer

logger = logging.getLogger("plogrid-ai.stream")

REDIS_URL = os.getenv("REDIS_URL")
REQUEST_STREAM = "trash-analysis-requests"
CONSUMER_GROUP = "plogrid-ai-workers"
CONSUMER_NAME = os.getenv("REDIS_CONSUMER_NAME")

BLOCK_MS = 2000
BATCH_SIZE = 1
RECONNECT_DELAY_S = 5

class StreamConsumer:
    """trash-analysis-requests 스트림을 소비해 YOLO+NIR 분류를 수행하는 컨슈머."""

    def __init__(self, yolo_predictor, nir_predictor):
        self.yolo_predictor = yolo_predictor
        self.nir_predictor = nir_predictor
        self.redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        self.http = httpx.AsyncClient(timeout=15.0)
        self.producer = ResultProducer()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="redis-stream-consumer")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.http.aclose()
        await self.redis.aclose()
        await self.producer.close()

    async def _run(self) -> None:
        await self._ensure_group()
        await self._recover_pending()
        logger.info(
            "Redis Streams consumer started (stream=%s group=%s consumer=%s)",
            REQUEST_STREAM, CONSUMER_GROUP, CONSUMER_NAME,
        )
        while True:
            try:
                response = await self.redis.xreadgroup(
                    groupname=CONSUMER_GROUP,
                    consumername=CONSUMER_NAME,
                    streams={REQUEST_STREAM: ">"},
                    count=BATCH_SIZE,
                    block=BLOCK_MS,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if "NOGROUP" in str(e):
                    logger.warning("스트림/컨슈머 그룹이 사라짐, 다시 생성 시도")
                    await self._ensure_group()
                else:
                    logger.exception("Redis Streams 읽기 실패, %d초 후 재시도", RECONNECT_DELAY_S)
                    await asyncio.sleep(RECONNECT_DELAY_S)
                continue

            if not response:
                continue

            for _stream_name, messages in response:
                for record_id, fields in messages:
                    await self._handle(record_id, fields)

    async def _ensure_group(self) -> None:
        while True:
            try:
                await self.redis.xgroup_create(REQUEST_STREAM, CONSUMER_GROUP, id="0", mkstream=True)
                return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if "BUSYGROUP" in str(e):
                    return
                logger.exception("Redis 연결/컨슈머 그룹 생성 실패, %d초 후 재시도", RECONNECT_DELAY_S)
                await asyncio.sleep(RECONNECT_DELAY_S)

    async def _recover_pending(self) -> None:
        """이전 실행이 크래시로 죽어서 ACK 못 하고 남긴, 이 컨슈머 이름 앞으로 걸린 메시지를 이어서 처리한다."""
        while True:
            response = await self.redis.xreadgroup(
                groupname=CONSUMER_GROUP,
                consumername=CONSUMER_NAME,
                streams={REQUEST_STREAM: "0"},
                count=BATCH_SIZE,
            )
            if not response:
                return
            _stream_name, messages = response[0]
            if not messages:
                return
            for record_id, fields in messages:
                await self._handle(record_id, fields)

    async def _handle(self, record_id: str, fields: dict) -> None:
        try:
            result = await self._process(fields)
            await self.producer.publish_success(fields, result)
        except Exception as e:
            logger.exception(
                "메시지 처리 실패 (recordId=%s ploggingId=%s)", record_id, fields.get("ploggingId")
            )
            await self.producer.publish_error(fields, str(e))
        finally:
            await self.redis.xack(REQUEST_STREAM, CONSUMER_GROUP, record_id)

    async def _process(self, fields: dict) -> dict:
        nir_values = parse_nir_values(fields["spectralValues"])

        resp = await self.http.get(fields["imageUrl"])
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")

        try:
            return classify_material(image, nir_values, self.yolo_predictor, self.nir_predictor)
        except PipelineError as e:
            raise RuntimeError(str(e))
