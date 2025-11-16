from __future__ import annotations
import argparse, json, os, sys, random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

# We rely on HuggingFace transformers locally cached (offline use).
from transformers import AutoConfig, AutoModel, AutoTokenizer


LABELS = ["PRT_B", "PRT_I", "PRT_E", "BND", "O"]  # fixed order
NUM_LABELS = len(LABELS)


# ---- Small wrapper with a linear head (no CRF in export; CRF was training-time only) ----

class MiniLMTagger(nn.Module):
    def __init__(self, hf_model_path: str, hidden_size: int, dropout_p: float = 0.0):
        super().__init__()
        cfg = AutoConfig.from_pretrained(hf_model_path, local_files_only=True)
        cfg.hidden_dropout_prob = 0.0
        cfg.attention_probs_dropout_prob = 0.0
        self.encoder = AutoModel.from_pretrained(hf_model_path, config=cfg, local_files_only=True)
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, NUM_LABELS, bias=True)

        # Initialize head deterministically to small values (keeps logits non-zero, stable)
        torch.manual_seed(13)
        nn.init.xavier_uniform_(self.classifier.weight, gain=0.8)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # input_ids: [B,T], attention_mask: [B,T]
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state  # [B,T,H]
        h = self.dropout(h)
        logits = self.classifier(h)  # [B,T,5]
        return logits


# ---- Export helpers ----

def _infer_hidden_size(hf_model_path: str) -> int:
    cfg = AutoConfig.from_pretrained(hf_model_path, local_files_only=True)
    if hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    # fallbacks for odd configs
    for k in ["dim", "d_model", "hidden"]:
        if hasattr(cfg, k):
            return int(getattr(cfg, k))
    raise RuntimeError("Cannot infer hidden size from config.")


@dataclass
class ExportPaths:
    onnx_out: str
    quant_out: str

def _paths(variant: str, out_dir: str) -> ExportPaths:
    os.makedirs(out_dir, exist_ok=True)
    if variant.upper() == "L3":
        onnx = os.path.join(out_dir, "splitter_miniL3.onnx")
        qout = os.path.join(out_dir, "splitter_miniL3.int8.onnx")
    elif variant.upper() == "L6":
        onnx = os.path.join(out_dir, "splitter_miniL6.onnx")
        qout = os.path.join(out_dir, "splitter_miniL6.int8.onnx")
    else:
        raise ValueError("variant must be L3 or L6")
    return ExportPaths(onnx, qout)

def _set_all_seeds(seed: int = 13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def export_onnx(
    hf_model_path: str,
    variant: str,
    out_dir: str,
    max_seq_len: int = 128,
    opset: int = 13,
 ) -> Tuple[ExportPaths, int]:
    _set_all_seeds(13)
    hidden = _infer_hidden_size(hf_model_path)
    # CL-2: print hidden size for diagnostics
    print(f"[INFO] Hidden size = {hidden}  (variant={variant})")
    model = MiniLMTagger(hf_model_path, hidden_size=hidden, dropout_p=0.0)
    model.eval()

    # If a trained head checkpoint exists, try to load its weights into the classifier.
    # The training script writes a dict of parameter-name -> tensor (e.g. 'fc.weight','fc.bias').
    ckpt_path = os.path.join("artifacts","models","splitter_head.pt")
    if os.path.exists(ckpt_path):
        try:
            sd = torch.load(ckpt_path, map_location="cpu")
            # If the saved object is a mapping of raw tensors -> use that
            if isinstance(sd, dict):
                # Normalise keys: training head used 'fc.weight' and 'fc.bias'
                new_sd = {}
                for k, v in sd.items():
                    if k.startswith("fc."):
                        new_k = k.replace("fc.", "classifier.")
                    else:
                        new_k = k
                    new_sd[new_k] = v
                try:
                    model.classifier.load_state_dict({
                        k.split("classifier.",1)[1]: v for k,v in new_sd.items() if k.startswith("classifier.")
                    })
                    print(f"[INFO] Loaded head weights from {ckpt_path}")
                except Exception:
                    # fallback - attempt to set weight/bias directly
                    w = new_sd.get("classifier.weight")
                    b = new_sd.get("classifier.bias")
                    if w is not None:
                        with torch.no_grad():
                            model.classifier.weight.copy_(w)
                    if b is not None:
                        with torch.no_grad():
                            model.classifier.bias.copy_(b)
                    print(f"[WARN] Partially loaded head weights from {ckpt_path}")
        except Exception as e:
            print(f"[WARN] Failed to load head checkpoint {ckpt_path}: {e}")

    # Dummy inputs (deterministic)
    B, T = 1, max_seq_len
    input_ids = torch.ones((B, T), dtype=torch.long)
    attention_mask = torch.ones((B, T), dtype=torch.long)

    p = _paths(variant, out_dir)

    # Dynamic axes so we can run variable length
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "logits": {0: "batch", 1: "seq"},
    }

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        p.onnx_out,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=opset,
    )
    return p, hidden


def quantize_dynamic(in_path: str, out_path: str):
    # Quantize linear weights to INT8 for CPU speed
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        model_input=in_path,
        model_output=out_path,
        weight_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["L3", "L6"], required=True)
    # Point this to a local MiniLM directory (offline) or a HF id if you have cache populated
    ap.add_argument("--hf_model_path", required=True, help="Local dir or HF model id (cached offline).")
    ap.add_argument("--out_dir", default="artifacts/models", help="Where to write ONNX files.")
    ap.add_argument("--max_seq_len", type=int, default=128)
    ap.add_argument("--opset", type=int, default=13, help="ONNX opset version to use for export")
    ap.add_argument("--no_quant", action="store_true")
    args = ap.parse_args()

    # CL-1: sanity-check local model dir is complete. Accept either a
    # full fast tokenizer snapshot (tokenizer.json) OR the older layout
    # with tokenizer_config.json + vocab.txt.
    model_root = args.hf_model_path
    has_config = os.path.exists(os.path.join(model_root, "config.json"))
    has_tokenizer_json = os.path.exists(os.path.join(model_root, "tokenizer.json"))
    has_tokenizer_legacy = os.path.exists(os.path.join(model_root, "tokenizer_config.json")) and os.path.exists(os.path.join(model_root, "vocab.txt"))
    if not has_config or not (has_tokenizer_json or has_tokenizer_legacy):
        raise FileNotFoundError(
            f"--hf_model_path must contain config.json and either tokenizer.json OR (tokenizer_config.json + vocab.txt). Checked: {model_root}"
        )
    print(f"[OK] Using local HF model at: {args.hf_model_path}")

    # If exporting the L6 variant, scaled_dot_product_attention requires opset >= 14.
    if args.variant.upper() == "L6" and args.opset < 14:
        print(f"[WARN] variant=L6 requires opset>=14 for scaled_dot_product_attention; overriding opset {args.opset} -> 14")
        args.opset = 14

    paths, hidden = export_onnx(
        hf_model_path=args.hf_model_path,
        variant=args.variant,
        out_dir=args.out_dir,
        max_seq_len=args.max_seq_len,
        opset=args.opset,
    )
    print(f"Exported: {paths.onnx_out}")

    if not args.no_quant:
        quantize_dynamic(paths.onnx_out, paths.quant_out)
        print(f"Quantized: {paths.quant_out}")

    # Emit a small manifest for reproducibility
    man = {
        "variant": args.variant,
        "hf_model_path": args.hf_model_path,
        "onnx_out": paths.onnx_out,
        "quantized_out": None if args.no_quant else paths.quant_out,
        "max_seq_len": args.max_seq_len,
        "labels": LABELS,
        "seed": 13,
        "hidden_size": int(hidden),
    }
    mpath = os.path.join(args.out_dir, f"splitter_{args.variant.lower()}_export.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    print(f"Wrote manifest: {mpath}")


if __name__ == "__main__":
    main()
