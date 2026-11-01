"""Build the local speech-embedding model used by backend/embed.py (one-off, ~100 MB).

Two ways, both ending in models/<name>.onnx (+ a .json with its settings):

  python -m scripts.build_speech_model                 # export: needs torch + transformers
                                                        # (pip install -r requirements-train.txt)
  python -m scripts.build_speech_model --download      # prebuilt ONNX, needs only the app's packages

--export keeps only the first --layer transformer layers of WavLM Base+ (default 6). Middle layers
describe speech sounds best; the last layers drift towards what the model was pre-trained to
predict. Cutting the model there also halves its run time. --download fetches a prebuilt export
of the full 12-layer model (Xenova/wavlm-base-plus on Hugging Face).

Any wav2vec2-family model can be exported with --model, for example an Indian-language one.
Afterwards set SSL_MODEL_PATH in .env if the file name differs from the default.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
PREBUILT = "https://huggingface.co/Xenova/wavlm-base-plus/resolve/main/onnx/model_quantized.onnx"


def export(model_id: str, layer: int, out: Path) -> None:
    try:
        import torch
        from transformers import AutoConfig, AutoFeatureExtractor, AutoModel
    except ImportError:
        sys.exit("Exporting needs torch and transformers: pip install -r requirements-train.txt "
                 "(or use --download)")
    from onnxruntime.quantization import QuantType, quantize_dynamic

    cfg = AutoConfig.from_pretrained(model_id)
    cfg.num_hidden_layers = layer                     # drop everything above the chosen layer
    model = AutoModel.from_pretrained(model_id, config=cfg).eval()
    normalize = bool(getattr(AutoFeatureExtractor.from_pretrained(model_id), "do_normalize", False))

    class Wrapped(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_values):
            return self.m(input_values).last_hidden_state

    fp32 = out.with_name(out.stem + ".fp32.onnx")
    dummy = torch.zeros(1, 16000)
    torch.onnx.export(Wrapped(model), (dummy,), str(fp32), input_names=["input_values"],
                      output_names=["last_hidden_state"], opset_version=17, dynamo=False,
                      dynamic_axes={"input_values": {1: "samples"}, "last_hidden_state": {1: "frames"}})
    quantize_dynamic(str(fp32), str(out), weight_type=QuantType.QInt8)
    fp32.unlink(missing_ok=True)
    write_meta(out, model_id, layer, normalize, "export")


def download(out: Path) -> None:
    tmp = out.with_suffix(".part")
    with httpx.stream("GET", PREBUILT, follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    tmp.replace(out)
    # WavLM Base+ is pre-trained on raw (un-normalised) audio; this export keeps all 12 layers.
    write_meta(out, "microsoft/wavlm-base-plus", 12, False, PREBUILT)


def write_meta(out: Path, model_id: str, layer: int, normalize: bool, source: str) -> None:
    out.with_suffix(".json").write_text(json.dumps(
        {"model": model_id, "layer": layer, "normalize_input": normalize, "source": source,
         "frame_ms": 20, "quantized": "int8"}, indent=2), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true", help="fetch the prebuilt 12-layer model instead")
    ap.add_argument("--model", default="microsoft/wavlm-base-plus")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    MODELS.mkdir(exist_ok=True)
    if a.download:
        download(a.out or MODELS / "wavlm-base-plus-l12.int8.onnx")
    else:
        name = a.model.split("/")[-1]
        export(a.model, a.layer, a.out or MODELS / f"{name}-l{a.layer}.int8.onnx")


if __name__ == "__main__":
    main()
