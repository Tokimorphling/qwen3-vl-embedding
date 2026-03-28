# llama.cpp Qwen3-VL Commit Notes

This document summarizes the larger `llama.cpp` commits used in this repo for
Qwen3-VL / Qwen3-VL-Embedding support. It exists to make later review, cherry-pick,
and history cleanup easier.

## Commit Stack

The original large commit:

- `22bab3c` `Add Qwen3-VL embedding frontend`

was split into the following smaller commits on `llama.cpp` `master`:

1. `e60684b` `examples: add llama-vl-embedding frontend`
2. `d37dfa2` `mtmd: align Qwen3-VL preprocessing and projector inputs`
3. `646165f` `mtmd: align Qwen3-VL vision graph with HF`
4. `098d588` `Fix Qwen3-VL resize defaults`

The final tree after the split is identical to the previously working tree.

## e60684b

Title:
- `examples: add llama-vl-embedding frontend`

Files:
- `common/common.h`
- `common/arg.cpp`
- `examples/embedding/vl-embedding.cpp`
- `tools/mtmd/CMakeLists.txt`

Purpose:
- Add a standalone `llama-vl-embedding` CLI.
- Keep the original `embedding.cpp` path untouched.
- Accept Python-like structured JSON inputs for text, image, and mixed cases.

Main responsibilities:
- CLI entry point for Qwen3-VL-Embedding style usage.
- Argument parsing for structured inputs.
- Reuse `mtmd` for multimodal tokenization / image handling.

Why this commit is separate:
- It is user-facing CLI work.
- It does not change the Qwen3-VL vision math by itself.
- It is the easiest part to review independently.

## d37dfa2

Title:
- `mtmd: align Qwen3-VL preprocessing and projector inputs`

Files:
- `tools/mtmd/clip.cpp`

Purpose:
- Align image preprocessing and host-side auxiliary inputs with the HF / Transformers
  Qwen3-VL path.

Main changes:
- Add `qwen3vl_fast_pos_embed_interpolate(...)` for Qwen3-VL learned position embedding interpolation.
- Add `qwen3vl_build_vision_rope_tables(...)` for host-side Qwen3-VL vision RoPE tables.
- Feed `learned_pos_embd`, `rope_cos`, `rope_sin`, `ln_one`, and `ln_eps` into the Qwen3-VL graph.
- Add image preprocessing helpers and Qwen-like smart-resize handling.
- Add debug hooks such as `MTMD_DEBUG_TENSOR` and `MTMD_DEBUG_PREPROC`.

Important note about bicubic:
- `resize_bicubic_pillow(...)` intentionally uses `a = -0.5`.
- This matches Pillow's actual bicubic implementation in `Resample.c`.
- The common review claim that Pillow bicubic must use `a = -0.75` is not correct for this code path.

Why this commit is separate:
- This is the main preprocessing / tensor-plumbing change.
- It is large, but conceptually focused on the `clip.cpp` side only.

## 646165f

Title:
- `mtmd: align Qwen3-VL vision graph with HF`

Files:
- `tools/mtmd/models/qwen3vl.cpp`

Purpose:
- Align the Qwen3-VL vision graph itself with the HF forward path.

Main changes:
- Make `learned_pos_embd`, `rope_cos`, `rope_sin`, `ln_one`, and `ln_eps` explicit graph inputs.
- Replace generic internal handling with Qwen3-VL-specific vision graph wiring.
- Use explicit LayerNorm logic matching PyTorch LayerNorm behavior.
- Use explicit Qwen3-VL vision RoPE application.
- Keep vision FFN in simple `fc1 -> act -> fc2` style by passing `nullptr` for gate tensors.

Important note about FFN:
- The use of `nullptr` gate tensors here is intentional.
- `build_ffn(...)` falls back to the non-gated path when `gate == nullptr`.
- That matches Qwen3-VL vision MLP structure better than a gated SwiGLU path.

Open review point:
- HF vision config uses `hidden_act = "gelu_pytorch_tanh"` for Qwen3-VL vision.
- Current GGUF conversion path reduces that to the generic `vision_use_gelu(true)` signal.
- This has not shown up as the dominant alignment issue so far, but it is still a reasonable place for future precision review.

Why this commit is separate:
- It changes the graph math and tensor layout in `qwen3vl.cpp`.
- It should be reviewed separately from preprocessing and CLI code.

## 098d588

Title:
- `Fix Qwen3-VL resize defaults`

Files:
- `tools/mtmd/clip.cpp`

Purpose:
- Restore the correct default resize behavior for Qwen3-VL after rebasing onto newer upstream.

Main changes:
- Keep Qwen3-VL on `RESIZE_ALGO_BICUBIC_PILLOW`.
- Keep `image_resize_pad = false`.

Why this commit exists separately:
- It is a follow-up behavior fix after the main Qwen3-VL support landed.
- It is small and easy to reason about on its own.

## Validation Notes

### `qwen3vl_fast_pos_embed_interpolate(...)` is necessary

We ran a control experiment:

- Keep `vl-embedding` and the current vision graph.
- Only remove the merge-order reordering from `qwen3vl_fast_pos_embed_interpolate(...)`.

Result:
- Text samples stayed effectively unchanged.
- Image regression degraded sharply.
- Example `2B + f32 + python float32 + strict regression`:
  - current implementation:
    - `image_only cosine = 0.999910116`
    - `image_text cosine = 0.999916017`
    - `regression_check = OK`
  - row-major control experiment:
    - `image_only cosine = 0.936925173`
    - `image_text cosine = 0.950305402`
    - `regression_check = FAILED`

Conclusion:
- The merge-order permutation is not optional.
- It is needed to match the actual Qwen3-VL token ordering.

### `RESIZE_ALGO_BICUBIC_PILLOW` is measurably better than plain `RESIZE_ALGO_BICUBIC`

We ran another control experiment:

- Keep `qwen3vl_fast_pos_embed_interpolate(...)`.
- Only replace the Qwen3-VL default resize path with plain `RESIZE_ALGO_BICUBIC`.

Result:
- Text samples stayed unchanged.
- Image regression became noticeably worse.
- Example `2B + f32 + python float32 + strict regression`:
  - `BICUBIC_PILLOW`:
    - `image_only cosine = 0.999910116`
    - `image_text cosine = 0.999916017`
    - `pairwise_similarity max_abs_diff = 0.000650585`
    - `regression_check = OK`
  - `BICUBIC`:
    - `image_only cosine = 0.999664187`
    - `image_text cosine = 0.999657214`
    - `pairwise_similarity max_abs_diff = 0.007593751`
    - `regression_check = FAILED`

Conclusion:
- Plain bicubic is usable, but it is not as close to HF / Pillow behavior.
- `RESIZE_ALGO_BICUBIC_PILLOW` should remain the default for Qwen3-VL.

## Review Guidance

If these commits need to be reviewed again, the recommended order is:

1. `e60684b` for CLI / API shape
2. `d37dfa2` for preprocessing and host-side auxiliary tensors
3. `646165f` for graph math
4. `098d588` for the follow-up resize default fix

This order keeps user-facing changes, preprocessing changes, graph changes,
and follow-up fixes clearly separated.
