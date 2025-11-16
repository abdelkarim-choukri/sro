Exporter usage

This repository includes `scripts/export_splitter_onnx.py` to export a MiniLM encoder + small linear head to ONNX and optionally quantize to INT8.

Important notes

- Use the HF *snapshot* directory as `--hf_model_path` when working with a local HuggingFace cache. For example:

  models_cache/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/<hash>/

  The exporter requires a `config.json` and either a `tokenizer.json` (fast tokenizer snapshot) or the legacy `tokenizer_config.json` + `vocab.txt` layout.

- For the L6 variant (MiniLM-L6), the model uses scaled-dot-product attention which requires ONNX opset >= 14. The exporter accepts `--opset` and will auto-upgrade to 14 for `variant=L6` if a lower opset is specified.

- Example export command:

  ```powershell
  python -m scripts.export_splitter_onnx --variant L6 \
    --hf_model_path <path-to-snapshot-dir> \
    --out_dir artifacts/models --opset 14
  ```

- For CI: install the pinned versions in `requirements-dev.txt` (onnx>=1.15, onnxruntime>=1.17) so opset 14 and quantization are available.

Contact

If you still hit missing files, double-check you are passing the snapshot folder (not the parent repository folder) as `--hf_model_path`.