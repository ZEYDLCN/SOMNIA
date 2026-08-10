"""
Eğitilmiş Temporal Transformer'ı ONNX'e export eder — web formunda
("Bugünkü tahmin" özelliği) gerçek model çıkarımı yapabilmek için.

Neden ONNX: tam PyTorch (torch) kütüphanesi yüzlerce MB'tır ve
PythonAnywhere/Koyeb gibi ücretsiz platformların disk kotasını
zorlayabilir. ONNX Runtime çok daha hafiftir (~50-60MB) ve sadece
çıkarım (inference) için yeterlidir — eğitim burada yapılmıyor zaten.

Kullanım:
    python -m src.models.export_onnx
    -> src/webapp/model/temporal_transformer.onnx
    -> src/webapp/model/normalizer.json

Bu iki dosya KÜÇÜKTÜR (model ~100KB, normalizer birkaç KB) ve BİLEREK
repoya commit edilir (checkpoints/ klasörünün aksine) — web formunun
deploy edildiği ortamda modeli yeniden eğitmeye gerek kalmasın diye.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.data.preprocessing import build_pipeline
from src.models.transformer import TemporalSleepTransformer, TransformerConfig

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = ROOT / "checkpoints" / "temporal_transformer.pt"
OUT_DIR = ROOT / "src" / "webapp" / "model"
ONNX_PATH = OUT_DIR / "temporal_transformer.onnx"
NORMALIZER_PATH = OUT_DIR / "normalizer.json"


class _InferenceWrapper(torch.nn.Module):
    """ONNX export'u basitleştirmek için: sadece tahmini (attention
    dict'i değil) döndüren ince bir sarmalayıcı."""

    def __init__(self, model: TemporalSleepTransformer):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, return_attn=False)


def main() -> None:
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    cfg = TransformerConfig(**ckpt["config"])
    model = TemporalSleepTransformer(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    wrapper = _InferenceWrapper(model)
    wrapper.eval()
    dummy = torch.randn(1, cfg.window_size, cfg.n_features)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        str(ONNX_PATH),
        input_names=["window"],
        output_names=["sleep_quality_pred"],
        dynamic_axes={"window": {0: "batch"}, "sleep_quality_pred": {0: "batch"}},
        opset_version=17,
    )
    print(f"Kaydedildi: {ONNX_PATH}")

    # Normalizer: aynı ölçekleme (mean/std) + özellik sırası, pipeline
    # eğitiminde kullanılanla BİREBİR aynı olmalı.
    splits = build_pipeline(window_size=cfg.window_size)
    normalizer = {
        "window_size": cfg.window_size,
        "n_features": cfg.n_features,
        "feature_names": splits.train.feature_names,
        "mean": splits.feature_means.tolist(),
        "std": splits.feature_stds.tolist(),
    }
    NORMALIZER_PATH.write_text(json.dumps(normalizer, indent=2, ensure_ascii=False))
    print(f"Kaydedildi: {NORMALIZER_PATH}")

    # Doğrulama: ONNX çıktısı, orijinal PyTorch modeliyle eşleşmeli.
    import onnxruntime as ort

    session = ort.InferenceSession(str(ONNX_PATH))
    x_np = dummy.numpy().astype(np.float32)
    onnx_out = session.run(None, {"window": x_np})[0]
    with torch.no_grad():
        torch_out = wrapper(dummy).numpy()
    max_diff = float(np.abs(onnx_out - torch_out).max())
    print(f"Doğrulama: ONNX vs PyTorch max fark = {max_diff:.6f} (0'a yakın olmalı)")
    assert max_diff < 1e-4, "ONNX export sırasında sayısal sapma çok büyük!"


if __name__ == "__main__":
    main()
