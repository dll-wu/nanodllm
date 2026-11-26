# Portable model weights

A native trainer checkpoint retains optimizer, RNG and resume state. Use only your own or explicitly trusted trainer checkpoints. To share model weights independently of those records, export an inference file. Supply the known training-checkpoint SHA-256. For a Hugging Face tokenizer, also supply its directory and retain the run's `data_identity.v1.json` or `data_identity.v2.json` next to the input checkpoint, or pass its location with `--data-identity`.

```bash
checkpoint=runs/tiny-block/model_latest.pt
source_sha=$(python -c 'import sys; from nanodllm.weights import file_sha256; print(file_sha256(sys.argv[1]))' "$checkpoint")
nanodllm-export-weights --trusted-checkpoint "$checkpoint" --source-sha256 "$source_sha" --output runs/tiny-block/model.weights.pt
nanodllm-generate --checkpoint runs/tiny-block/model.weights.pt --prompt "The lamp" --capacity 16 --seed 7
```

For the built-in byte tokenizer omit `--tokenizer-dir`. An export includes the named tensors, resolved forward architecture and scale multipliers, RoPE base, objective and clean-stream conventions, tokenizer content identity, parent file digest and completed training counters. It omits corpus and host paths, optimizer and RNG state. The exporter prints the new file's SHA-256. Publish that digest with the file if you distribute it. The native and exported file digests differ.

Read the export with `nanodllm.weights.load_weights(path, expected_checkpoint_sha256=...)`. For an HF model pass `tokenizer_path=...`. The loader uses restricted PyTorch tensor deserialization, verifies tokenizer identity and strictly matches state-dict keys. The returned object exposes the model, tokenizer, config, checkpoint receipt and diffusion continuation adapter. The export can also initialize a compatible recipe using `--parent FILE.weights.pt --parent-sha256 DIGEST`. A child starts a new optimizer and schedule. To resume an interrupted optimizer trajectory, use its original native training checkpoint.
