import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from nir_app.model import FusionHead, SpectralEncoder

REPO_ID = "yeajongcheol/nir-yolo-fusion"
MODEL_FILENAME = "fusion_model.pt"

CACHE_DIR = Path(os.getenv("NIR_MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))

CSV_ORDER = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "R", "S", "T", "U", "V", "W"]


class NIRPredictor:
    def __init__(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        ckpt_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME, local_dir=str(CACHE_DIR))
        ck = torch.load(ckpt_path, weights_only=False, map_location="cpu")

        self.enc = SpectralEncoder(18, (64, 32), 2, 0.3)
        self.enc.load_state_dict(ck["encoder"])
        self.enc.eval()

        self.head = FusionHead(4, (16,), 2, 0.2)
        self.head.load_state_dict(ck["head"])
        self.head.eval()

        self.gate = float(torch.sigmoid(ck["gate_logit"]))
        self.mu = ck["nir_mu"]
        self.sd = ck["nir_sd"]
        self.channels = list(ck["channels"])
        self.glass_ids = list(ck["glass_ids"])
        self.pet_ids = list(ck["pet_ids"])
        self.names = dict(ck["names_ko"])
        self.labels = list(ck["labels"])
        self._name2id = {v: k for k, v in self.names.items()}

    def to_zrgb(self, yolo_scores=None, yolo_class=None, yolo_conf=None):
        """26클래스 점수 또는 단일 검출 -> z_rgb (2,) = [유리, 투명 플라스틱]"""
        s = np.zeros(26, dtype=np.float32)
        if yolo_scores is not None:
            s = np.asarray(yolo_scores, dtype=np.float32).ravel()
            if s.size != 26:
                raise ValueError(f"yolo_scores 는 26차원이어야 함 (받은 값 {s.size})")
        elif yolo_class is not None:
            cid = self._name2id[yolo_class] if isinstance(yolo_class, str) else int(yolo_class)
            if cid not in self.names:
                raise ValueError(f"알 수 없는 클래스: {yolo_class}")
            s[cid] = 1.0 if yolo_conf is None else float(yolo_conf)
        else:
            raise ValueError("yolo_scores 또는 yolo_class 중 하나는 필요함")

        g = s[self.glass_ids].max() if self.glass_ids else 0.0
        p = s[self.pet_ids].max() if self.pet_ids else 0.0
        return np.array([g, p], dtype=np.float32), s

    def predict(
        self,
        nir: List[float],
        yolo_scores: Optional[List[float]] = None,
        yolo_class: Optional[Union[str, int]] = None,
        yolo_conf: Optional[float] = None,
        nir_order: str = "wavelength",
    ) -> dict:
        z_rgb, s26 = self.to_zrgb(yolo_scores, yolo_class, yolo_conf)

        x = np.asarray(nir, dtype=float).ravel()
        if x.size != 18:
            raise ValueError(f"nir 은 18채널이어야 함 (받은 값 {x.size})")
        if nir_order == "csv":
            x = np.array([x[CSV_ORDER.index(c)] for c in self.channels])

        xs = (x - self.mu) / self.sd

        with torch.no_grad():
            zn = self.enc(torch.tensor(xs.astype(np.float32))[None])
            f = torch.cat([torch.tensor(z_rgb)[None], zn * self.gate], 1)
            prob = torch.softmax(self.head(f), 1).numpy()[0]
            zn = zn.numpy()[0]

        i = int(prob.argmax())
        det = int(s26.argmax()) if s26.max() > 0 else None

        return {
            "label": self.labels[i],
            "confidence": float(prob[i]),
            "prob": {self.labels[0]: float(prob[0]), self.labels[1]: float(prob[1])},
            "z_rgb": {self.labels[0]: float(z_rgb[0]), self.labels[1]: float(z_rgb[1])},
            "z_n": zn.tolist(),
            "gate": self.gate,
            "yolo_class": self.names.get(det) if det is not None else None,
            "yolo_in_scope": bool(z_rgb.max() > 0),
        }
