#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/convert_and_regress_qwen3_vl_embedding.sh [options]

Options:
  --model-dir PATH              Hugging Face model directory.
  --output-dir PATH             Directory used for generated GGUF files.
  --outtype f16|bf16|f32       Output type for both main model and mmproj. Default: f16
  --python-device auto|cpu|cuda
                               Device for the Python reference. Default: auto
  --cuda-visible-devices VALUE Override CUDA_VISIBLE_DEVICES for regression.
  --llama-ngl VALUE            Value passed to llama-vl-embedding as -ngl. Default: auto
  --llama-no-mmproj-offload    Pass --no-mmproj-offload to regression.
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
model_dir="$repo_root/models/Qwen3-VL-Embedding-2B"
output_dir="$repo_root"
outtype="f16"
python_device="auto"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES-}"
llama_ngl="auto"
llama_no_mmproj_offload=0
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
    --python-device)
      python_device="$2"
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

case "$python_device" in
  auto|cpu|cuda) ;;
  *)
    echo "--python-device must be auto, cpu, or cuda, got: $python_device" >&2
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
llama_dir="$repo_root/llama.cpp"
build_dir="$llama_dir/build"
llama_bin="$build_dir/bin/llama-vl-embedding"
model_name=$(basename "$model_dir")
gguf_model="$output_dir/${model_name}-${outtype}.gguf"
mmproj="$output_dir/mmproj-${model_name}-${outtype}.gguf"

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

if [[ $skip_build -eq 0 ]]; then
  if [[ $rebuild -eq 1 || ! -x "$llama_bin" || $needs_reconfigure -eq 1 ]]; then
    cmake_config_cmd=("$cmake_bin" -S "$llama_dir" -B "$build_dir")

    if [[ "$effective_cuda_build" == "on" ]]; then
      cmake_config_cmd+=(-DGGML_CUDA=ON)
    elif [[ "$effective_cuda_build" == "off" ]]; then
      cmake_config_cmd+=(-DGGML_CUDA=OFF)
    fi

    echo "[build] ${cmake_config_cmd[*]}"
    "${cmake_config_cmd[@]}"
    "$cmake_bin" --build "$build_dir" --target llama-vl-embedding -j
  else
    echo "[build] reuse existing binary: $llama_bin"
  fi
fi

convert_if_needed() {
  local target=$1
  shift
  if [[ $force_convert -eq 1 || ! -f "$target" ]]; then
    echo "[convert] generating $(basename "$target")"
    "$python_bin" "$llama_dir/convert_hf_to_gguf.py" "$model_dir" --outfile "$target" "$@"
  else
    echo "[convert] reuse existing $(basename "$target")"
  fi
}

convert_if_needed "$gguf_model" --outtype "$outtype"
convert_if_needed "$mmproj" --outtype "$outtype" --mmproj

if [[ $skip_regression -eq 1 ]]; then
  echo "[done] conversion finished"
  echo "  gguf_model: $gguf_model"
  echo "  mmproj:     $mmproj"
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
  --llama-ngl "$llama_ngl"
)

if [[ -n "$cuda_visible_devices" ]]; then
  regression_cmd+=(--cuda-visible-devices "$cuda_visible_devices")
fi

if [[ $llama_no_mmproj_offload -eq 1 ]]; then
  regression_cmd+=(--llama-no-mmproj-offload)
fi

echo "[regression] ${regression_cmd[*]}"
"${regression_cmd[@]}"
