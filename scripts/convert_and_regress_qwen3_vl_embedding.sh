#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/convert_and_regress_qwen3_vl_embedding.sh [options]

Options:
  --model-dir PATH              Hugging Face model directory.
  --output-dir PATH             Directory used for generated GGUF files.
  --outtype f16|bf16|f32        Shorthand output type for both main model and mmproj. Default: f16
  --model-outtype f16|bf16|f32  Output type for the main model GGUF.
  --mmproj-outtype f16|bf16|f32 Output type for the mmproj GGUF.
  --model-quant-type none|int8|Q8_0
                                Optional post-quantization for the main model. Default: none
  --python-device auto|cpu|cuda
                               Device for the Python reference. Default: auto
  --python-torch-dtype auto|float32|float16|bfloat16
                               Optional torch_dtype override for the Python reference. Default: auto
  --cuda-visible-devices VALUE Override CUDA_VISIBLE_DEVICES for regression.
  --llama-ngl VALUE            Value passed to llama-vl-embedding as -ngl. Default: auto
  --llama-no-mmproj-offload    Pass --no-mmproj-offload to regression.
  --regression-mode strict|retrieval
                               Regression policy. Default: strict
  --run-all-precisions         Run a built-in sweep: f32/f32, bf16/bf16, f16/f16, Q8_0(main)+f16(mmproj)
  --results-file PATH          Append detailed regression logs and suite summary to this file.
  --cuda-build auto|on|off     Whether to configure llama.cpp with GGML_CUDA. Default: auto
  --python-bin PATH            Python executable. Default: python
  --cmake-bin PATH             CMake executable. Default: cmake
  --skip-build                 Skip building llama-vl-embedding.
  --skip-regression            Skip the regression step.
  --rebuild                    Force rebuilding llama-vl-embedding.
  --force-convert              Re-run conversion even if GGUF outputs already exist.
  --install-convert-deps       Run pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
  -h, --help                   Show this help message.
EOF
}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
script_path=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
model_dir="$repo_root/models/Qwen3-VL-Embedding-2B"
output_dir="$repo_root"
outtype="f16"
model_outtype=""
mmproj_outtype=""
model_quant_type="none"
python_device="auto"
python_torch_dtype="auto"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES-}"
llama_ngl="auto"
llama_no_mmproj_offload=0
regression_mode="strict"
run_all_precisions=0
results_file=""
python_bin="${PYTHON:-python}"
cmake_bin="${CMAKE:-cmake}"
skip_build=0
skip_regression=0
rebuild=0
force_convert=0
install_convert_deps=0
cuda_build="auto"

while (($# > 0)); do
  case "$1" in
    --model-dir)
      model_dir="$2"
      shift 2
      ;;
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    --outtype)
      outtype="$2"
      shift 2
      ;;
    --model-outtype)
      model_outtype="$2"
      shift 2
      ;;
    --mmproj-outtype)
      mmproj_outtype="$2"
      shift 2
      ;;
    --model-quant-type)
      model_quant_type="$2"
      shift 2
      ;;
    --python-device)
      python_device="$2"
      shift 2
      ;;
    --python-torch-dtype)
      python_torch_dtype="$2"
      shift 2
      ;;
    --cuda-visible-devices)
      cuda_visible_devices="$2"
      shift 2
      ;;
    --llama-ngl)
      llama_ngl="$2"
      shift 2
      ;;
    --llama-no-mmproj-offload)
      llama_no_mmproj_offload=1
      shift
      ;;
    --regression-mode)
      regression_mode="$2"
      shift 2
      ;;
    --run-all-precisions)
      run_all_precisions=1
      shift
      ;;
    --results-file)
      results_file="$2"
      shift 2
      ;;
    --python-bin)
      python_bin="$2"
      shift 2
      ;;
    --cuda-build)
      cuda_build="$2"
      shift 2
      ;;
    --cmake-bin)
      cmake_bin="$2"
      shift 2
      ;;
    --skip-build)
      skip_build=1
      shift
      ;;
    --skip-regression)
      skip_regression=1
      shift
      ;;
    --rebuild)
      rebuild=1
      shift
      ;;
    --force-convert)
      force_convert=1
      shift
      ;;
    --install-convert-deps)
      install_convert_deps=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "$outtype" in
  f16|bf16|f32) ;;
  *)
    echo "--outtype must be f16, bf16, or f32, got: $outtype" >&2
    exit 1
    ;;
esac

if [[ -z "$model_outtype" ]]; then
  model_outtype="$outtype"
fi

if [[ -z "$mmproj_outtype" ]]; then
  mmproj_outtype="$outtype"
fi

case "$model_outtype" in
  f16|bf16|f32) ;;
  *)
    echo "--model-outtype must be f16, bf16, or f32, got: $model_outtype" >&2
    exit 1
    ;;
esac

case "$mmproj_outtype" in
  f16|bf16|f32) ;;
  *)
    echo "--mmproj-outtype must be f16, bf16, or f32, got: $mmproj_outtype" >&2
    exit 1
    ;;
esac

case "$model_quant_type" in
  none) ;;
  int8|q8_0|Q8_0)
    model_quant_type="Q8_0"
    ;;
  *)
    echo "--model-quant-type must be none, int8, or Q8_0, got: $model_quant_type" >&2
    exit 1
    ;;
esac

case "$python_device" in
  auto|cpu|cuda) ;;
  *)
    echo "--python-device must be auto, cpu, or cuda, got: $python_device" >&2
    exit 1
    ;;
esac

case "$python_torch_dtype" in
  auto|float32|float16|bfloat16) ;;
  *)
    echo "--python-torch-dtype must be auto, float32, float16, or bfloat16, got: $python_torch_dtype" >&2
    exit 1
    ;;
esac

case "$regression_mode" in
  strict|retrieval) ;;
  *)
    echo "--regression-mode must be strict or retrieval, got: $regression_mode" >&2
    exit 1
    ;;
esac

case "$cuda_build" in
  auto|on|off) ;;
  *)
    echo "--cuda-build must be auto, on, or off, got: $cuda_build" >&2
    exit 1
    ;;
esac

model_dir=$(cd "$(dirname "$model_dir")" && pwd)/$(basename "$model_dir")
mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd)
if [[ -n "$results_file" ]]; then
  mkdir -p "$(dirname "$results_file")"
  results_file=$(cd "$(dirname "$results_file")" && pwd)/$(basename "$results_file")
  if [[ "${QWEN3_VL_RESULTS_APPEND-0}" != "1" ]]; then
    : > "$results_file"
  fi
fi
llama_dir="$repo_root/llama.cpp"
build_dir="$llama_dir/build"
llama_bin="$build_dir/bin/llama-vl-embedding"
quantize_bin="$build_dir/bin/llama-quantize"
model_name=$(basename "$model_dir")
base_model_gguf="$output_dir/${model_name}-${model_outtype}.gguf"
if [[ "$model_quant_type" == "none" ]]; then
  gguf_model="$base_model_gguf"
else
  gguf_model="$output_dir/${model_name}-${model_outtype}-${model_quant_type}.gguf"
fi
mmproj="$output_dir/mmproj-${model_name}-${mmproj_outtype}.gguf"

if [[ ! -d "$model_dir" ]]; then
  echo "model directory not found: $model_dir" >&2
  exit 1
fi

if [[ $install_convert_deps -eq 1 ]]; then
  "$python_bin" -m pip install -r "$llama_dir/requirements/requirements-convert_hf_to_gguf.txt"
fi

effective_cuda_build="$cuda_build"
if [[ "$effective_cuda_build" == "auto" ]]; then
  if [[ "$python_device" == "cuda" || "$llama_ngl" != "0" ]]; then
    effective_cuda_build="on"
  else
    effective_cuda_build="off"
  fi
fi

needs_reconfigure=0
if [[ -f "$build_dir/CMakeCache.txt" ]]; then
  current_cuda=$(sed -n 's/^GGML_CUDA:BOOL=//p' "$build_dir/CMakeCache.txt" | tail -n 1)
  if [[ "$effective_cuda_build" == "on" && "$current_cuda" != "1" && "$current_cuda" != "ON" ]]; then
    needs_reconfigure=1
  fi
  if [[ "$effective_cuda_build" == "off" && "$current_cuda" != "0" && "$current_cuda" != "OFF" ]]; then
    needs_reconfigure=1
  fi
else
  needs_reconfigure=1
fi

needs_quantize_bin=0
if [[ ( "$model_quant_type" != "none" || $run_all_precisions -eq 1 ) && ! -x "$quantize_bin" ]]; then
  needs_quantize_bin=1
fi

log_note() {
  echo "$*"
  if [[ -n "$results_file" ]]; then
    echo "$*" >> "$results_file"
  fi
}

if [[ $skip_build -eq 0 ]]; then
  if [[ $rebuild -eq 1 || ! -x "$llama_bin" || $needs_reconfigure -eq 1 || $needs_quantize_bin -eq 1 ]]; then
    cmake_config_cmd=("$cmake_bin" -S "$llama_dir" -B "$build_dir")

    if [[ "$effective_cuda_build" == "on" ]]; then
      cmake_config_cmd+=(-DGGML_CUDA=ON)
    elif [[ "$effective_cuda_build" == "off" ]]; then
      cmake_config_cmd+=(-DGGML_CUDA=OFF)
    fi

    log_note "[build] ${cmake_config_cmd[*]}"
    "${cmake_config_cmd[@]}"
    build_targets=(llama-vl-embedding)
    if [[ "$model_quant_type" != "none" || $run_all_precisions -eq 1 ]]; then
      build_targets+=(llama-quantize)
    fi
    "$cmake_bin" --build "$build_dir" --target "${build_targets[@]}" -j
  else
    log_note "[build] reuse existing binary: $llama_bin"
  fi
fi

convert_if_needed() {
  local target=$1
  shift
  if [[ $force_convert -eq 1 || ! -f "$target" ]]; then
    log_note "[convert] generating $(basename "$target")"
    "$python_bin" "$llama_dir/convert_hf_to_gguf.py" "$model_dir" --outfile "$target" "$@"
  else
    log_note "[convert] reuse existing $(basename "$target")"
  fi
}

quantize_if_needed() {
  local source=$1
  local target=$2
  local quant_type=$3

  if [[ $force_convert -eq 1 || ! -f "$target" ]]; then
    if [[ ! -x "$quantize_bin" ]]; then
      echo "quantize binary not found: $quantize_bin" >&2
      exit 1
    fi
    log_note "[quantize] generating $(basename "$target") from $(basename "$source") as $quant_type"
    "$quantize_bin" "$source" "$target" "$quant_type"
  else
    log_note "[quantize] reuse existing $(basename "$target")"
  fi
}

append_results_header() {
  local title=$1
  if [[ -z "$results_file" ]]; then
    return
  fi

  {
    echo
    echo "================================================================"
    echo "$title"
    echo "timestamp: $(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "repo_root: $repo_root"
    echo "model_dir: $model_dir"
    echo "python_device: $python_device"
    echo "python_torch_dtype: $python_torch_dtype"
    echo "llama_ngl: $llama_ngl"
    echo "llama_no_mmproj_offload: $llama_no_mmproj_offload"
    echo "regression_mode: $regression_mode"
    echo "cuda_visible_devices: ${cuda_visible_devices-}"
    echo "================================================================"
  } >> "$results_file"
}

run_regression_logged() {
  local label=$1
  shift
  local -a cmd=("$@")
  local tmp
  tmp=$(mktemp)

  append_results_header "$label"
  if [[ -n "$results_file" ]]; then
    {
      echo "[command] ${cmd[*]}"
    } >> "$results_file"
  fi

  set +e
  "${cmd[@]}" >"$tmp" 2>&1
  local status=$?
  set -e

  if [[ -n "$results_file" ]]; then
    cat "$tmp" >> "$results_file"
  fi

  if [[ -n "$results_file" ]]; then
    local python_line llama_line pairwise_line retrieval_line metrics_line result_line
    python_line=$(grep -m1 '^python_reference:' "$tmp" || true)
    llama_line=$(grep -m1 '^llama_vl_embedding:' "$tmp" || true)
    pairwise_line=$(grep -m1 '^pairwise_similarity:' "$tmp" || true)
    retrieval_line=$(grep -m1 '^retrieval_consistency:' "$tmp" || true)
    metrics_line=$(grep -m1 '^retrieval_metrics:' "$tmp" || true)
    result_line=$(grep -m1 '^regression_check:' "$tmp" || true)

    echo "[$label] ${result_line:-exit_code=$status}"
    [[ -n "$python_line" ]] && echo "[$label] $python_line"
    [[ -n "$llama_line" ]] && echo "[$label] $llama_line"
    [[ -n "$pairwise_line" ]] && echo "[$label] $pairwise_line"
    [[ -n "$retrieval_line" ]] && echo "[$label] $retrieval_line"
    [[ -n "$metrics_line" ]] && echo "[$label] $metrics_line"
    echo "[$label] detailed log: $results_file"
  else
    cat "$tmp"
  fi

  rm -f "$tmp"
  return $status
}

if [[ $run_all_precisions -eq 1 ]]; then
  suite_failed=0
  suite_summary=()
  variants=(
    "f32 f32 none"
    "bf16 bf16 none"
    "f16 f16 none"
    "f16 f16 Q8_0"
  )

  for variant in "${variants[@]}"; do
    read -r suite_model_outtype suite_mmproj_outtype suite_model_quant_type <<< "$variant"

    suite_base_model_gguf="$output_dir/${model_name}-${suite_model_outtype}.gguf"
    if [[ "$suite_model_quant_type" == "none" ]]; then
      suite_gguf_model="$suite_base_model_gguf"
    else
      suite_gguf_model="$output_dir/${model_name}-${suite_model_outtype}-${suite_model_quant_type}.gguf"
    fi
    suite_mmproj="$output_dir/mmproj-${model_name}-${suite_mmproj_outtype}.gguf"
    suite_label="model_outtype=$suite_model_outtype mmproj_outtype=$suite_mmproj_outtype model_quant_type=$suite_model_quant_type"

    log_note ""
    log_note "================================================================"
    log_note "[suite] $suite_label"
    log_note "================================================================"

    convert_if_needed "$suite_base_model_gguf" --outtype "$suite_model_outtype"
    if [[ "$suite_model_quant_type" != "none" ]]; then
      quantize_if_needed "$suite_base_model_gguf" "$suite_gguf_model" "$suite_model_quant_type"
    fi
    convert_if_needed "$suite_mmproj" --outtype "$suite_mmproj_outtype" --mmproj

    if [[ $skip_regression -eq 1 ]]; then
      log_note "[suite] regression skipped"
      suite_summary+=("$suite_label status=SKIPPED")
      continue
    fi

    regression_cmd=(
      "$python_bin"
      "$repo_root/scripts/check_qwen3_vl_embedding_regression.py"
      --repo-root "$repo_root"
      --hf-model-dir "$model_dir"
      --gguf-model "$suite_gguf_model"
      --mmproj "$suite_mmproj"
      --llama-bin "$llama_bin"
      --python-device "$python_device"
      --python-torch-dtype "$python_torch_dtype"
      --regression-mode "$regression_mode"
      --llama-ngl "$llama_ngl"
    )

    if [[ -n "$cuda_visible_devices" ]]; then
      regression_cmd+=(--cuda-visible-devices "$cuda_visible_devices")
    fi
    if [[ $llama_no_mmproj_offload -eq 1 ]]; then
      regression_cmd+=(--llama-no-mmproj-offload)
    fi

    log_note "[regression] ${regression_cmd[*]}"
    if run_regression_logged "$suite_label" "${regression_cmd[@]}"; then
      log_note "[suite] completed"
      suite_summary+=("$suite_label status=OK")
    else
      log_note "[suite] failed: $suite_label"
      suite_failed=1
      suite_summary+=("$suite_label status=FAILED")
    fi
  done

  if [[ -n "$results_file" ]]; then
    {
      echo
      echo "================================================================"
      echo "suite_summary"
      printf '%s\n' "${suite_summary[@]}"
    } >> "$results_file"
    log_note "[suite] summary file: $results_file"
  fi

  exit $suite_failed
fi

convert_if_needed "$base_model_gguf" --outtype "$model_outtype"
if [[ "$model_quant_type" != "none" ]]; then
  quantize_if_needed "$base_model_gguf" "$gguf_model" "$model_quant_type"
fi
convert_if_needed "$mmproj" --outtype "$mmproj_outtype" --mmproj

if [[ $skip_regression -eq 1 ]]; then
  log_note "[done] conversion finished"
  log_note "  gguf_model: $gguf_model"
  log_note "  mmproj:     $mmproj"
  exit 0
fi

regression_cmd=(
  "$python_bin"
  "$repo_root/scripts/check_qwen3_vl_embedding_regression.py"
  --repo-root "$repo_root"
  --hf-model-dir "$model_dir"
  --gguf-model "$gguf_model"
  --mmproj "$mmproj"
  --llama-bin "$llama_bin"
  --python-device "$python_device"
  --python-torch-dtype "$python_torch_dtype"
  --regression-mode "$regression_mode"
  --llama-ngl "$llama_ngl"
)

if [[ -n "$cuda_visible_devices" ]]; then
  regression_cmd+=(--cuda-visible-devices "$cuda_visible_devices")
fi

if [[ $llama_no_mmproj_offload -eq 1 ]]; then
  regression_cmd+=(--llama-no-mmproj-offload)
fi

log_note "[regression] ${regression_cmd[*]}"
run_label="model_outtype=$model_outtype mmproj_outtype=$mmproj_outtype model_quant_type=$model_quant_type"
run_regression_logged "$run_label" "${regression_cmd[@]}"
