# Qwen3-VL-Embedding on llama.cpp

This repository converts `Qwen3-VL-Embedding` models into GGUF files usable by `llama.cpp`, and uses a locally patched `llama.cpp` to reproduce the official Python / Hugging Face embedding behavior.

Current conclusions:

- For the already validated `text`, `image`, and `image + text` inputs, the current locally patched `llama.cpp` is closer to the official behavior than clean upstream.
- The recommended deployment configuration is `F16 + mmproj-f16`.
- A fixed regression script is included, so the Python reference implementation and `llama-vl-embedding` can be compared directly.
- Regression supports two decision modes: `strict` looks at numeric closeness, `retrieval` looks at retrieval ranking consistency.
- The one-shot script supports separate precision for the main model and `mmproj`, `Q8_0` for the main model, a full-precision sweep, and writing detailed results to a file.
- The current default fixture only has 5 samples, so `retrieval` mode is better treated as a smoke test / sanity check and should not directly replace large-scale retrieval evaluation.

The repository itself only tracks code, scripts, and documentation; model weights, GGUF files, logs, and local experiment artifacts are not included in version control.

## 0. Clone

This repository uses submodules to track dependency repositories:

- `llama.cpp`
- `Qwen3-VL-Embedding`

For the first checkout, it is recommended to use:

```bash
git clone --recursive <this-repo-url>
```

If you have already cloned it, run:

```bash
git submodule update --init --recursive
```

## 1. Directory Layout

All relative paths below are relative to the repository root:

- `./llama.cpp`: the locally patched `llama.cpp` submodule
- `./Qwen3-VL-Embedding`: the official Python reference implementation submodule
- `./models/Qwen3-VL-Embedding-2B`: local Hugging Face model directory
- `./scripts/check_qwen3_vl_embedding_regression.py`: Python vs `llama-vl-embedding` regression script
- `./scripts/data/qwen3_vl_embedding_regression_inputs.json`: fixed regression fixture
- `./scripts/convert_and_regress_qwen3_vl_embedding.sh`: one-shot build, convert, and regression entrypoint

## 2. Quick Start

First activate the Python environment you have already prepared:

```bash
conda activate base
cd /path/to/qwen3vl-embedding
```

Install the conversion dependencies once:

```bash
python -m pip install -r ./llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
```

If your machine has CUDA and you want to run GPU regression directly:

```bash
./scripts/convert_and_regress_qwen3_vl_embedding.sh \
  --model-dir ./models/Qwen3-VL-Embedding-2B \
  --python-device cuda \
  --cuda-visible-devices 0
```

If you care more about retrieval ranking than elementwise numeric agreement, you can switch to `retrieval` mode:

```bash
./scripts/convert_and_regress_qwen3_vl_embedding.sh \
  --model-dir ./models/Qwen3-VL-Embedding-8B \
  --python-device cuda \
  --cuda-visible-devices 0 \
  --regression-mode retrieval
```

If you want to test an `INT8/Q8_0` main model directly while keeping `mmproj-f16`:

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

If you want to set the main model and `mmproj` precision separately, for example `F32 + mmproj-f16`:

```bash
./scripts/convert_and_regress_qwen3_vl_embedding.sh \
  --model-dir ./models/Qwen3-VL-Embedding-8B \
  --model-outtype f32 \
  --mmproj-outtype f16 \
  --python-device cuda \
  --python-torch-dtype float32 \
  --cuda-visible-devices 0
```

If you want to run through all the common `llama.cpp` precision variants in one go:

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

This sweep runs, in order:

- `f32 + mmproj-f32`
- `bf16 + mmproj-bf16`
- `f16 + mmproj-f16`
- `Q8_0(main) + mmproj-f16`

If you only want to use the CPU:

```bash
./scripts/convert_and_regress_qwen3_vl_embedding.sh \
  --model-dir ./models/Qwen3-VL-Embedding-2B \
  --python-device cpu \
  --llama-ngl 0
```

By default, this script will:

1. build `llama-vl-embedding`
2. generate `Qwen3-VL-Embedding-2B-f16.gguf`
3. generate `mmproj-Qwen3-VL-Embedding-2B-f16.gguf`
4. run the Python vs `llama-vl-embedding` regression on the fixed fixture

If the output files already exist, it skips the repeated conversion; if you want to force reconversion, pass `--force-convert`.

If you pass `--results-file`, the detailed regression output and `suite_summary` are written to the file, while the terminal keeps only a compact summary for each variant.

## 3. One-Shot Script Arguments

```bash
./scripts/convert_and_regress_qwen3_vl_embedding.sh --help
```

Common arguments:

- `--model-dir PATH`: HF model directory
- `--outtype f16|bf16|f32`: shorthand precision for both the main model and `mmproj`, defaults to `f16`
- `--model-outtype f16|bf16|f32`: precision for the main model GGUF only
- `--mmproj-outtype f16|bf16|f32`: precision for the `mmproj` GGUF only
- `--model-quant-type none|int8|Q8_0`: optional post-quantization for the main model; `int8` maps to `Q8_0`
- `--python-device auto|cpu|cuda`: device selection for the Python reference implementation
- `--python-torch-dtype auto|float32|float16|bfloat16`: `torch_dtype` for the Python reference implementation
- `--cuda-visible-devices VALUE`: explicitly select the GPU
- `--llama-ngl VALUE`: `-ngl` passed to `llama-vl-embedding`
- `--llama-no-mmproj-offload`: disable `mmproj` GPU offload
- `--regression-mode strict|retrieval`: regression decision mode, defaults to `strict`
- `--run-all-precisions`: run `f32`, `bf16`, `f16`, and `Q8_0(main)+mmproj-f16` in sequence
- `--results-file PATH`: append detailed regression output and the sweep summary to a file
- `--cuda-build auto|on|off`: whether to explicitly set `GGML_CUDA` during the build
- `--force-convert`: reconvert even if the target GGUF already exists
- `--rebuild`: force a rebuild of `llama-vl-embedding`
- `--skip-build`: skip compilation
- `--skip-regression`: convert only, do not run regression
- `--install-convert-deps`: run `pip install -r ...` once inside the script

Notes:

- The "GPU mode" of the current script affects not only runtime, but also the build.
- With `--cuda-build auto`, if `--python-device cuda` is set or `--llama-ngl` is not `0`, the script passes `-DGGML_CUDA=ON` at configure time.
- If you want to force a CPU build, you can explicitly pass `--cuda-build off`.
- If the `GGML_CUDA` configuration in an existing `build/` directory does not match the current mode, the script reconfigures and rebuilds automatically instead of blindly reusing the old binaries.
- `--outtype` is a shorthand; if you also pass `--model-outtype` or `--mmproj-outtype`, the separately specified arguments take precedence.
- Quantization is applied only to the main model; keeping `mmproj` at `f16` or `f32` is still recommended.
- `strict` mode keeps the original numeric thresholds and is suitable for precise regression on already well-aligned cases such as 2B.
- `retrieval` mode mainly looks at `pairwise_pearson`, `Spearman`, `Recall@K`, and `MRR`, and is more suitable for cases like 8B where ranking is stable but small elementwise residuals still remain.
- If `--results-file` is passed, the terminal keeps only a compact summary and the full regression output is written to the file, which is convenient for long sweeps.
- When doing a "precision sweep", it is recommended to pin `--python-torch-dtype` instead of using `auto`. Otherwise the Python reference may vary between `bf16/float16/float32` on different machines, making it hard to tell whether a difference comes from `llama.cpp` or from the reference implementation itself.
- If your goal is to mimic the default deployment behavior on an A100, either `--python-torch-dtype bfloat16` or `auto` will do; if your goal is an apples-to-apples numeric comparison, it is recommended to explicitly use `float32` against `f32`, and `float16` against `f16/Q8_0`.
- A passing `retrieval` result only means the ranking structure on the current small sample has not obviously broken; it does not mean elementwise numerics are aligned, nor does it guarantee equivalence on a large-scale retrieval dataset.

## 4. Manual Conversion

If you do not want to use the one-shot script, you can also do it manually.

### 4.1 Convert the main model

```bash
cd ./llama.cpp

python ./convert_hf_to_gguf.py \
  ../models/Qwen3-VL-Embedding-2B \
  --outfile ../Qwen3-VL-Embedding-2B-f16.gguf \
  --outtype f16
```

If you want to generate `bf16`:

```bash
python ./convert_hf_to_gguf.py \
  ../models/Qwen3-VL-Embedding-2B \
  --outfile ../Qwen3-VL-Embedding-2B-bf16.gguf \
  --outtype bf16
```

### 4.2 Convert mmproj

```bash
python ./convert_hf_to_gguf.py \
  ../models/Qwen3-VL-Embedding-2B \
  --outfile ../mmproj-Qwen3-VL-Embedding-2B-f16.gguf \
  --outtype f16 \
  --mmproj
```

The `bf16` version:

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

Text only:

```bash
./llama.cpp/build/bin/llama-vl-embedding \
  -m ./Qwen3-VL-Embedding-2B-f16.gguf \
  --inputs '[{"text":"hello world"}]' \
  --pooling last \
  --embd-normalize 2 \
  --embd-output-format array \
  -c 4096
```

Image + text:

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

The fixed regression script is located at:

- [`scripts/check_qwen3_vl_embedding_regression.py`](scripts/check_qwen3_vl_embedding_regression.py)

The default fixed fixture is located at:

- [`scripts/data/qwen3_vl_embedding_regression_inputs.json`](scripts/data/qwen3_vl_embedding_regression_inputs.json)

Run the recommended configuration directly:

```bash
python ./scripts/check_qwen3_vl_embedding_regression.py \
  --gguf-model ./Qwen3-VL-Embedding-2B-f16.gguf \
  --mmproj ./mmproj-Qwen3-VL-Embedding-2B-f16.gguf \
  --python-device cuda \
  --cuda-visible-devices 0 \
  --llama-ngl auto
```

If you want a fairer 8B comparison on an A100, it is recommended to explicitly pin the Python reference dtype:

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

If you care more about retrieval ranking consistency, you can switch directly to `retrieval` mode:

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

If you want stable CPU numeric regression:

```bash
python ./scripts/check_qwen3_vl_embedding_regression.py \
  --python-device cpu \
  --llama-ngl 0
```

It reports:

- cosine similarity for each sample
- mean / max absolute diff
- max difference of the pairwise similarity matrix
- retrieval consistency: `pairwise_pearson`, `nn_top1_acc`, `topK_overlap`
- retrieval metrics: `spearman_mean`, `spearman_min`, `Recall@1`, `Recall@K`, `MRR`
- load / run / total timing for Python and `llama-vl-embedding`

It is recommended to interpret the results as follows:

- `strict = OK`: the numerics are also reasonably close, suitable for high-precision regression.
- `retrieval = OK`: ranking consistency on the current samples is good enough, but a final judgment should still combine a larger evaluation set.
- The current fixed fixture only has 5 samples, so the explanatory power of `Recall@1/MRR = 1.0` is limited; it is better treated as a daily smoke test.

## 6. Ablation: The `vl-embedding` Frontend Alone Is Not Enough

To verify whether the later `Qwen3-VL` adaptation commits are actually "necessary", you can compare directly against a version containing only the frontend commit `e60684b`.

The following set of commands will:

1. check out `e60684b` in a temporary worktree
2. build `llama-vl-embedding` with `-j4`
3. run regression on the same `2B + f32 + python float32 + strict` samples

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

Actual results:

- the three text samples are essentially unchanged:
  - `text_query cosine = 0.999999285`
  - `text_doc cosine = 0.999999642`
  - `text_concat cosine = 0.999999940`
- the image path degrades significantly:
  - `image_only cosine = 0.969332635`
  - `image_text cosine = 0.988328397`
  - `pairwise_similarity max_abs_diff = 0.047971517`
  - `nn_top1_acc = 0.8`
  - `recall_at_1 = 0.8`
  - `regression_check = FAILED`

Compared with the current version:

- the same regression returns `regression_check = OK` on the current version
- current version:
  - `image_only cosine = 0.999910116`
  - `image_text cosine = 0.999916017`
  - `pairwise_similarity max_abs_diff = 0.000650585`

This ablation shows:

- `e60684b` mainly solves the CLI / frontend entry point
- the `vl-embedding` frontend alone cannot align the Qwen3-VL image path with the official implementation
- what actually brings the multimodal results closer to HF is the later `d37dfa2`, `646165f`, and the subsequent resize follow-up fix

## 7. Why the Current Patched Version Is Closer to the Official Implementation

Two different meanings of "close" are distinguished here:

- behaviorally close: for the same inputs, the outputs are closer to official Python / HF
- code-wise close: the code is organized more like the HF source

The current set of local patches mainly improves "behavioral closeness".

Key points:

- absolute position embedding interpolation now uses `qwen3vl_fast_pos_embed_interpolate()` instead of the generic `resize_position_embeddings()`
- visual `RoPE` and `LayerNorm` are handled more explicitly in the `Qwen3VL` branch
- `QWEN3VL` image resize follows a bicubic path that is closer to Pillow
- the `vl-embedding` input format now matches the array-style input of the official Python implementation

Conversely, the patch embedding backbone was not forcibly rewritten in full, because the stable patch conv path in upstream is already basically equivalent for single-image inputs.

## 8. Verified Alignment Results

In the current version, `vl-embedding` has been aligned with the official Python / HF reference implementation on these inputs:

- `text`
- `image`
- `image + text`

Typical results on the current fixed regression fixture:

- `text_query` cosine about `0.999999`
- `text_doc` cosine about `0.999999`
- `image_only` cosine about `0.99991`
- `image_text` cosine about `0.99991`

This shows that on the already validated paths, the current version is closer to the official behavior than clean upstream.

The boundaries should also be clear:

- this does not mean bit-identical to the official implementation
- the current focus of validation is `text`, `image`, and `image + text`
- the video path has not yet been validated at the same level of completeness

## 8. GPU Vision Crash Fix

Previously, the vision CUDA offload path of `Qwen3-VL` produced `signal 11`. The root cause was not a generic CUDA graph issue, but that host-side code directly read `position_embeddings` that had already been offloaded to the GPU backend.

The fix is:

- first pull `position_embeddings` back to the host with `ggml_backend_tensor_get()`
- then run `fast_pos_embed_interpolate` on the host

The current version has already fixed this problem, and `mmproj` can go through GPU offload normally; if you still see an immediate crash on images with an old build, rebuild first.

## 9. Quantization

Build the quantization tool first:

```bash
cmake --build ./llama.cpp/build --target llama-quantize -j
```

Then quantize the main model, for example to `Q8_0`:

```bash
./llama.cpp/build/bin/llama-quantize \
  ./Qwen3-VL-Embedding-2B-f32.gguf \
  ./Qwen3-VL-Embedding-2B-Q8_0.gguf \
  Q8_0
```

Recommendations for embedding use:

- formally replacing PyTorch: prefer `F16 + mmproj-f16`
- keeping the highest numeric baseline: keep a copy of `F32`
- pursuing maximum speed: you can try `Q8_0 + mmproj-f16`, but you must accept accuracy degradation

If you use the one-shot script, the equivalent form is:

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

## 10. Comparison with the PyTorch Reference Implementation

The following results come from the same fixed 5 samples on the same `RTX 2080 Ti`:

| Config | Python reference | llama.cpp | Alignment |
|---|---:|---:|---|
| `F32 + mmproj-f32` | load `10.283s`, run `0.963s`, total `11.246s` | load `16.159s`, prompt eval `2.160s`, total `17.515s`, wall `19.601s` | regression passed |
| `F32 + mmproj-f16` | load `9.143s`, run `1.848s`, total `10.991s` | load `13.592s`, prompt eval `1.982s`, total `14.766s`, wall `17.093s` | regression passed |
| `F16 + mmproj-f16` | load `9.546s`, run `0.926s`, total `10.472s` | load `7.300s`, prompt eval `0.377s`, total `7.566s`, wall `8.472s` | regression passed |
| `Q8_0 + mmproj-f16` | load `5.639s`, run `0.962s`, total `6.601s` | load `3.907s`, prompt eval `0.326s`, total `4.177s`, wall `5.016s` | regression failed |

Notes:

- the HF weights of the Python reference model are themselves `bfloat16`
- so the fairest runtime comparison is actually `PyTorch bf16` vs `llama.cpp F16`
- `F32` is better used as a numeric baseline rather than the best speed configuration on this card

Current recommendations:

- to formally replace PyTorch for running embeddings: prefer `F16 + mmproj-f16`
- to keep a high-precision baseline: also keep a copy of `F32`
- for maximum speed: consider `Q8_0` further
- if `Q8_0` only passes `retrieval` and fails `strict`, it is better treated as a speed-oriented option rather than the default precision configuration.

## 11. Related Code

Core files:

- [`llama.cpp/tools/mtmd/clip.cpp`](llama.cpp/tools/mtmd/clip.cpp)
- [`llama.cpp/tools/mtmd/models/qwen3vl.cpp`](llama.cpp/tools/mtmd/models/qwen3vl.cpp)
- [`llama.cpp/examples/embedding/vl-embedding.cpp`](llama.cpp/examples/embedding/vl-embedding.cpp)

Helper scripts:

- [`scripts/check_qwen3_vl_embedding_regression.py`](scripts/check_qwen3_vl_embedding_regression.py)
- [`scripts/convert_and_regress_qwen3_vl_embedding.sh`](scripts/convert_and_regress_qwen3_vl_embedding.sh)
