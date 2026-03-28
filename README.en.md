# Qwen3-VL-Embedding on llama.cpp

Chinese version: [README.md](README.md)

This repository converts `Qwen3-VL-Embedding` models to GGUF for `llama.cpp` and uses a locally patched `llama.cpp` to reproduce the official Python / Hugging Face embedding behavior.

Current takeaways:

- For the validated `text`, `image`, and `image + text` paths, this patched `llama.cpp` is closer to the official behavior than clean upstream.
- The recommended deployment setup is `F16 + mmproj-f16`.
- A fixed regression script is included to compare the Python reference implementation against `llama-vl-embedding`.
- Regression supports two modes: `strict` for numeric closeness and `retrieval` for ranking consistency.
- The one-shot script supports split precision for the main model and `mmproj`, optional `Q8_0` quantization for the main model, full precision sweeps, and writing detailed results to a file.
- The default fixture only has 5 samples, so `retrieval` mode should be treated as a smoke test / sanity check, not as a substitute for a large-scale retrieval benchmark.

The repo only tracks code, scripts, and documentation. Model weights, GGUF files, logs, and local experiment artifacts are intentionally ignored.

## 0. Clone

This repository uses submodules for:

- `llama.cpp`
- `Qwen3-VL-Embedding`

Recommended first clone:

```bash
git clone --recursive <this-repo-url>
```

If you already cloned it:

```bash
git submodule update --init --recursive
```

## 1. Layout

All relative paths below are repo-root-relative:

- `./llama.cpp`: patched `llama.cpp` submodule
- `./Qwen3-VL-Embedding`: official Python reference submodule
- `./models/Qwen3-VL-Embedding-2B`: local Hugging Face model directory
- `./scripts/check_qwen3_vl_embedding_regression.py`: Python vs `llama-vl-embedding` regression script
- `./scripts/data/qwen3_vl_embedding_regression_inputs.json`: fixed regression fixture
- `./scripts/convert_and_regress_qwen3_vl_embedding.sh`: one-shot build / convert / regress entrypoint

## 2. Quick Start

Activate your Python environment first:

```bash
conda activate base
cd /path/to/qwen3vl-embedding
```

Install conversion dependencies once:

```bash
python -m pip install -r ./llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
```

GPU regression:

```bash
./scripts/convert_and_regress_qwen3_vl_embedding.sh \
  --model-dir ./models/Qwen3-VL-Embedding-2B \
  --python-device cuda \
  --cuda-visible-devices 0
```

If you care more about retrieval ranking than strict numeric matching:

```bash
./scripts/convert_and_regress_qwen3_vl_embedding.sh \
  --model-dir ./models/Qwen3-VL-Embedding-8B \
  --python-device cuda \
  --cuda-visible-devices 0 \
  --regression-mode retrieval
```

To test `INT8/Q8_0` for the main model while keeping `mmproj-f16`:

```bash
./scripts/convert_and_regress_qwen3_vl_embedding.sh \
  --model-dir ./models/Qwen3-VL-Embedding-8B \
  --model-outtype f16 \
  --mmproj-outtype f16 \
  --model-quant-type int8 \
  --python-device cuda \
  --python-torch-dtype float16 \
  --cuda-visible-devices 0 \
  --regression-mode retrieval
```

To split precision, for example `F32 + mmproj-f16`:

```bash
./scripts/convert_and_regress_qwen3_vl_embedding.sh \
  --model-dir ./models/Qwen3-VL-Embedding-8B \
  --model-outtype f32 \
  --mmproj-outtype f16 \
  --python-device cuda \
  --python-torch-dtype float32 \
  --cuda-visible-devices 0
```

To sweep the common `llama.cpp` precision variants:

```bash
./scripts/convert_and_regress_qwen3_vl_embedding.sh \
  --model-dir ./models/Qwen3-VL-Embedding-8B \
  --python-device cuda \
  --python-torch-dtype float16 \
  --cuda-visible-devices 0 \
  --regression-mode retrieval \
  --results-file ./logs/qwen3-vl-embedding-8b-sweep.txt \
  --run-all-precisions
```

The sweep currently runs:

- `f32 + mmproj-f32`
- `bf16 + mmproj-bf16`
- `f16 + mmproj-f16`
- `Q8_0(main) + mmproj-f16`

CPU-only:

```bash
./scripts/convert_and_regress_qwen3_vl_embedding.sh \
  --model-dir ./models/Qwen3-VL-Embedding-2B \
  --python-device cpu \
  --llama-ngl 0
```

By default, the script will:

1. build `llama-vl-embedding`
2. generate `Qwen3-VL-Embedding-2B-f16.gguf`
3. generate `mmproj-Qwen3-VL-Embedding-2B-f16.gguf`
4. run regression on the fixed fixture

If outputs already exist, conversion is skipped. Use `--force-convert` to regenerate them.

If you pass `--results-file`, detailed regression output and the final `suite_summary` are written to that file, while the terminal keeps only a compact summary.

## 3. One-Shot Script Arguments

```bash
./scripts/convert_and_regress_qwen3_vl_embedding.sh --help
```

Common arguments:

- `--model-dir PATH`: HF model directory
- `--outtype f16|bf16|f32`: shorthand precision for both main model and `mmproj`
- `--model-outtype f16|bf16|f32`: precision for the main model GGUF
- `--mmproj-outtype f16|bf16|f32`: precision for the `mmproj` GGUF
- `--model-quant-type none|int8|Q8_0`: optional post-quantization for the main model; `int8` maps to `Q8_0`
- `--python-device auto|cpu|cuda`: device selection for the Python reference
- `--python-torch-dtype auto|float32|float16|bfloat16`: `torch_dtype` override for the Python reference
- `--cuda-visible-devices VALUE`: explicit GPU selection
- `--llama-ngl VALUE`: value passed to `llama-vl-embedding` as `-ngl`
- `--llama-no-mmproj-offload`: disable `mmproj` GPU offload
- `--regression-mode strict|retrieval`: regression policy
- `--run-all-precisions`: run the built-in precision sweep
- `--results-file PATH`: append detailed regression logs and sweep summaries to a file
- `--cuda-build auto|on|off`: whether to configure `GGML_CUDA`
- `--force-convert`: reconvert even if GGUF outputs already exist
- `--rebuild`: force a rebuild of `llama-vl-embedding`
- `--skip-build`: skip compilation
- `--skip-regression`: convert only, do not run regression
- `--install-convert-deps`: run `pip install -r ...` inside the script

Notes:

- GPU mode affects both runtime and build configuration.
- With `--cuda-build auto`, the script enables `-DGGML_CUDA=ON` if `--python-device cuda` or `--llama-ngl != 0`.
- Use `--cuda-build off` to force a CPU-only build.
- If an existing `build/` directory was configured with a different `GGML_CUDA` mode, the script reconfigures it automatically.
- `--outtype` is only a shorthand. If you also pass `--model-outtype` or `--mmproj-outtype`, the split settings take precedence.
- Quantization only applies to the main model. Keeping `mmproj` in `f16` or `f32` is still recommended.
- `strict` mode is meant for high-precision numeric regression.
- `retrieval` mode focuses on `pairwise_pearson`, `Spearman`, `Recall@K`, and `MRR`, and is more suitable when ranking is stable but elementwise differences still exist.
- Passing `--results-file` keeps the terminal quieter and makes long sweeps easier to review.
- For precision sweeps, fix `--python-torch-dtype` instead of using `auto`. Otherwise the Python reference may switch between `bf16`, `float16`, and `float32` depending on the machine, which makes the comparison harder to interpret.
- If you want to mimic A100-style default deployment behavior, `--python-torch-dtype bfloat16` or `auto` is fine. If you want apples-to-apples numeric comparisons, use `float32` for `f32`, and `float16` for `f16 / Q8_0`.
- A passing `retrieval` result only means that ranking structure looks stable on the current small fixture. It does not mean elementwise outputs are aligned, and it does not guarantee equivalence on a large retrieval dataset.

## 4. Manual Conversion

If you do not want to use the one-shot script, you can run the steps manually.

### 4.1 Convert the main model

```bash
cd ./llama.cpp

python ./convert_hf_to_gguf.py \
  ../models/Qwen3-VL-Embedding-2B \
  --outfile ../Qwen3-VL-Embedding-2B-f16.gguf \
  --outtype f16
```

For `bf16`:

```bash
python ./convert_hf_to_gguf.py \
  ../models/Qwen3-VL-Embedding-2B \
  --outfile ../Qwen3-VL-Embedding-2B-bf16.gguf \
  --outtype bf16
```

### 4.2 Convert `mmproj`

```bash
python ./convert_hf_to_gguf.py \
  ../models/Qwen3-VL-Embedding-2B \
  --outfile ../mmproj-Qwen3-VL-Embedding-2B-f16.gguf \
  --outtype f16 \
  --mmproj
```

For `bf16`:

```bash
python ./convert_hf_to_gguf.py \
  ../models/Qwen3-VL-Embedding-2B \
  --outfile ../mmproj-Qwen3-VL-Embedding-2B-bf16.gguf \
  --outtype bf16 \
  --mmproj
```

### 4.3 Build

```bash
cmake -S ./llama.cpp -B ./llama.cpp/build
cmake --build ./llama.cpp/build --target llama-vl-embedding -j
```

### 4.4 Run

Text-only:

```bash
./llama.cpp/build/bin/llama-vl-embedding \
  -m ./Qwen3-VL-Embedding-2B-f16.gguf \
  --inputs '[{"text":"hello world"}]' \
  --pooling last \
  --embd-normalize 2 \
  --embd-output-format array \
  -c 4096
```

Multimodal:

```bash
./llama.cpp/build/bin/llama-vl-embedding \
  -m ./Qwen3-VL-Embedding-2B-f16.gguf \
  --mmproj ./mmproj-Qwen3-VL-Embedding-2B-f16.gguf \
  --inputs '[{"text":"A dog on the beach","image":"./Qwen3-VL-Embedding/data/examples/0.jpeg"}]' \
  --pooling last \
  --embd-normalize 2 \
  --embd-output-format array \
  -c 4096 \
  -ngl auto
```

## 5. Regression Script

Regression script:

- [`scripts/check_qwen3_vl_embedding_regression.py`](scripts/check_qwen3_vl_embedding_regression.py)

Default fixture:

- [`scripts/data/qwen3_vl_embedding_regression_inputs.json`](scripts/data/qwen3_vl_embedding_regression_inputs.json)

Recommended run:

```bash
python ./scripts/check_qwen3_vl_embedding_regression.py \
  --gguf-model ./Qwen3-VL-Embedding-2B-f16.gguf \
  --mmproj ./mmproj-Qwen3-VL-Embedding-2B-f16.gguf \
  --python-device cuda \
  --cuda-visible-devices 0 \
  --llama-ngl auto
```

For a fairer 8B comparison on A100, explicitly fix the Python reference dtype:

```bash
python ./scripts/check_qwen3_vl_embedding_regression.py \
  --hf-model-dir ./models/Qwen3-VL-Embedding-8B \
  --gguf-model ./Qwen3-VL-Embedding-8B-f32.gguf \
  --mmproj ./mmproj-Qwen3-VL-Embedding-8B-f32.gguf \
  --python-device cuda \
  --python-torch-dtype float32 \
  --llama-ngl auto \
  --cuda-visible-devices 0
```

If you care more about ranking consistency:

```bash
python ./scripts/check_qwen3_vl_embedding_regression.py \
  --hf-model-dir ./models/Qwen3-VL-Embedding-8B \
  --gguf-model ./Qwen3-VL-Embedding-8B-f16.gguf \
  --mmproj ./mmproj-Qwen3-VL-Embedding-8B-f16.gguf \
  --python-device cuda \
  --python-torch-dtype float16 \
  --llama-ngl auto \
  --cuda-visible-devices 0 \
  --regression-mode retrieval
```

Stable CPU-only numeric regression:

```bash
python ./scripts/check_qwen3_vl_embedding_regression.py \
  --python-device cpu \
  --llama-ngl 0
```

The script reports:

- per-sample cosine similarity
- mean / max absolute difference
- max absolute difference of the pairwise similarity matrix
- retrieval consistency metrics: `pairwise_pearson`, `nn_top1_acc`, `topK_overlap`
- retrieval metrics: `spearman_mean`, `spearman_min`, `Recall@1`, `Recall@K`, `MRR`
- Python and `llama-vl-embedding` load / run / total timings

Interpretation:

- `strict = OK`: numeric behavior is also reasonably close
- `retrieval = OK`: ranking consistency is good on the current fixture, but you should still validate on a larger evaluation set
- the default fixture only has 5 samples, so `Recall@1 = 1.0` and `MRR = 1.0` should be treated as smoke-test signals, not as final evidence

## 6. Ablation: `vl-embedding` Frontend Alone Is Not Enough

To show that the later Qwen3-VL alignment commits are actually necessary, you can
compare against the frontend-only commit `e60684b`.

The following commands:

1. create a temporary worktree at `e60684b`
2. build `llama-vl-embedding` with `-j4`
3. run the same `2B + f32 + python float32 + strict` regression

```bash
git -C ./llama.cpp worktree add /tmp/llama.cpp-vl-only-test e60684b

cmake -S /tmp/llama.cpp-vl-only-test -B /tmp/llama.cpp-vl-only-test/build -DGGML_CUDA=ON
CMAKE_BUILD_PARALLEL_LEVEL=4 cmake --build /tmp/llama.cpp-vl-only-test/build --target llama-vl-embedding

python ./scripts/check_qwen3_vl_embedding_regression.py \
  --repo-root . \
  --hf-model-dir ./models/Qwen3-VL-Embedding-2B \
  --gguf-model ./Qwen3-VL-Embedding-2B-f32.gguf \
  --mmproj ./mmproj-Qwen3-VL-Embedding-2B-f32.gguf \
  --llama-bin /tmp/llama.cpp-vl-only-test/build/bin/llama-vl-embedding \
  --python-device cuda \
  --python-torch-dtype float32 \
  --regression-mode strict \
  --llama-ngl auto \
  --cuda-visible-devices 0
```

Observed result:

- text remains essentially unchanged:
  - `text_query cosine = 0.999999285`
  - `text_doc cosine = 0.999999642`
  - `text_concat cosine = 0.999999940`
- image behavior degrades significantly:
  - `image_only cosine = 0.969332635`
  - `image_text cosine = 0.988328397`
  - `pairwise_similarity max_abs_diff = 0.047971517`
  - `nn_top1_acc = 0.8`
  - `recall_at_1 = 0.8`
  - `regression_check = FAILED`

Compared with the current patched version:

- the same regression now returns `regression_check = OK`
- current version:
  - `image_only cosine = 0.999910116`
  - `image_text cosine = 0.999916017`
  - `pairwise_similarity max_abs_diff = 0.000650585`

This ablation shows:

- `e60684b` mainly solves the CLI / frontend entry point
- the `vl-embedding` frontend alone is not enough to align Qwen3-VL image behavior
- the later `d37dfa2`, `646165f`, and the resize follow-up are what bring the multimodal path close to HF

## 7. Why This Patched Version Is Closer to the Official Implementation

There are two different meanings of "closer":

- behavior is closer: outputs on the same inputs are closer to official Python / HF
- code is closer: the implementation structure looks more like HF source code

This repo mainly improves behavioral closeness.

Key points:

- absolute position interpolation now uses `qwen3vl_fast_pos_embed_interpolate()` instead of the generic `resize_position_embeddings()`
- visual `RoPE` and `LayerNorm` are handled more explicitly inside the `Qwen3VL` path
- `QWEN3VL` image resize now follows a more Pillow-like bicubic path
- `vl-embedding` input format follows the official Python array-style inputs

Patch embedding itself was not rewritten aggressively because upstream's stable patch-conv path is already basically equivalent for single-image inputs.

## 8. Verified Alignment

The current version has been validated against the official Python / HF reference on:

- `text`
- `image`
- `image + text`

Typical results on the fixed fixture:

- `text_query` cosine around `0.999999`
- `text_doc` cosine around `0.999999`
- `image_only` cosine around `0.99991`
- `image_text` cosine around `0.99991`

That means this version is behaviorally closer than clean upstream on the validated paths.

Limitations:

- this does not mean bit-identical outputs
- validation is currently strongest for `text`, `image`, and `image + text`
- the video path has not been validated to the same depth

## 8. GPU Vision Crash Fix

The `Qwen3-VL` vision CUDA offload path previously crashed with `signal 11`. The root cause was not a generic CUDA graph issue. Host code was directly reading `position_embeddings` that had already been offloaded to a GPU backend.

The fix was:

- copy `position_embeddings` back with `ggml_backend_tensor_get()`
- then run `fast_pos_embed_interpolate` on the host side

That crash is fixed in the current version, and `mmproj` can be offloaded to GPU again. If you still see an immediate image-side crash on an older build, rebuild first.

## 9. Quantization

Build the quantization tool first:

```bash
cmake --build ./llama.cpp/build --target llama-quantize -j
```

Then quantize the main model, for example `Q8_0`:

```bash
./llama.cpp/build/bin/llama-quantize \
  ./Qwen3-VL-Embedding-2B-f32.gguf \
  ./Qwen3-VL-Embedding-2B-Q8_0.gguf \
  Q8_0
```

Recommendations for embedding use cases:

- replace PyTorch in production: prefer `F16 + mmproj-f16`
- keep a highest-precision baseline: keep one `F32`
- maximize speed: try `Q8_0 + mmproj-f16`, but expect accuracy loss

One-shot equivalent:

```bash
./scripts/convert_and_regress_qwen3_vl_embedding.sh \
  --model-dir ./models/Qwen3-VL-Embedding-2B \
  --model-outtype f16 \
  --mmproj-outtype f16 \
  --model-quant-type int8 \
  --python-device cuda \
  --python-torch-dtype float16 \
  --cuda-visible-devices 0 \
  --regression-mode retrieval
```

## 10. Comparison with the PyTorch Reference

The table below comes from the same 5-sample fixture on the same `RTX 2080 Ti`:

| Config | Python reference | llama.cpp | Alignment |
|---|---:|---:|---|
| `F32 + mmproj-f32` | load `10.283s`, run `0.963s`, total `11.246s` | load `16.159s`, prompt eval `2.160s`, total `17.515s`, wall `19.601s` | regression passed |
| `F32 + mmproj-f16` | load `9.143s`, run `1.848s`, total `10.991s` | load `13.592s`, prompt eval `1.982s`, total `14.766s`, wall `17.093s` | regression passed |
| `F16 + mmproj-f16` | load `9.546s`, run `0.926s`, total `10.472s` | load `7.300s`, prompt eval `0.377s`, total `7.566s`, wall `8.472s` | regression passed |
| `Q8_0 + mmproj-f16` | load `5.639s`, run `0.962s`, total `6.601s` | load `3.907s`, prompt eval `0.326s`, total `4.177s`, wall `5.016s` | regression failed |

Notes:

- the HF reference weights are themselves `bfloat16`
- so the fairest runtime comparison is usually `PyTorch bf16` vs `llama.cpp F16`
- `F32` is more useful as a numeric baseline than as the fastest runtime option on this GPU

Current recommendation:

- for production replacement of PyTorch embedding: prefer `F16 + mmproj-f16`
- for a high-precision baseline: keep `F32`
- for maximum speed: consider `Q8_0`
- if `Q8_0` passes `retrieval` but fails `strict`, treat it as a speed-oriented option, not as the default precision choice

## 11. Relevant Code

Core files:

- [`llama.cpp/tools/mtmd/clip.cpp`](llama.cpp/tools/mtmd/clip.cpp)
- [`llama.cpp/tools/mtmd/models/qwen3vl.cpp`](llama.cpp/tools/mtmd/models/qwen3vl.cpp)
- [`llama.cpp/examples/embedding/vl-embedding.cpp`](llama.cpp/examples/embedding/vl-embedding.cpp)

Helper scripts:

- [`scripts/check_qwen3_vl_embedding_regression.py`](scripts/check_qwen3_vl_embedding_regression.py)
- [`scripts/convert_and_regress_qwen3_vl_embedding.sh`](scripts/convert_and_regress_qwen3_vl_embedding.sh)
