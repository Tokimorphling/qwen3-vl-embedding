#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="qwen3vl-eval"
PYTHON_VERSION="3.12"
CUDA_VARIANT="cu128"
VLMEVAL_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/VLMEvalKit"
USE_CONDA=0
INSTALL_VLMEVAL=1
PYTHON_BIN=""

usage() {
    cat <<'EOF'
Usage:
  setup_eval_env.sh [options]

Options:
  --env-name NAME         Conda env name when --conda is used. Default: qwen3vl-eval
  --python-version VER    Python version for conda env. Default: 3.12
  --cuda VARIANT          PyTorch wheel variant: cu128|cu126|cu118|cpu. Default: cu128
  --python-bin PATH       Use a specific Python interpreter. Default: current shell python
  --vlmeval-dir PATH      VLMEvalKit checkout dir. Default: $XDG_CACHE_HOME/VLMEvalKit
  --conda                 Create/activate a conda env before installing
  --no-vlmeval            Skip cloning/installing VLMEvalKit
  --no-conda              Alias for the default behavior; use current shell python
  -h, --help              Show this help

Examples:
  ./scripts/setup_eval_env.sh
  ./scripts/setup_eval_env.sh --cuda cu126
  ./scripts/setup_eval_env.sh --cuda cpu
  ./scripts/setup_eval_env.sh --conda --env-name qwen3vl-eval
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --python-version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --cuda)
            CUDA_VARIANT="$2"
            shift 2
            ;;
        --python-bin)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --vlmeval-dir)
            VLMEVAL_DIR="$2"
            shift 2
            ;;
        --conda)
            USE_CONDA=1
            shift
            ;;
        --no-vlmeval)
            INSTALL_VLMEVAL=0
            shift
            ;;
        --no-conda)
            USE_CONDA=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Missing command: $1" >&2
        exit 1
    }
}

case "${CUDA_VARIANT}" in
    cu128)
        TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
        ;;
    cu126)
        TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
        ;;
    cu118)
        TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
        ;;
    cpu)
        TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
        ;;
    *)
        echo "--cuda must be one of: cu128, cu126, cu118, cpu" >&2
        exit 2
        ;;
esac

VLMEVAL_DIR="${VLMEVAL_DIR/#\~/$HOME}"

if [[ -n "${PYTHON_BIN}" ]]; then
    USE_CONDA=0
fi

if [[ "${USE_CONDA}" -eq 1 ]]; then
    require_cmd conda
    eval "$(conda shell.bash hook)"

    if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
        echo "[conda] creating env ${ENV_NAME} (python=${PYTHON_VERSION})"
        conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
    fi

    echo "[conda] activating ${ENV_NAME}"
    conda activate "${ENV_NAME}"
    PYTHON_BIN="$(command -v python)"
else
    if [[ -z "${PYTHON_BIN}" ]]; then
        PYTHON_BIN="$(command -v python)"
    fi
fi

require_cmd git
"${PYTHON_BIN}" -m pip install --upgrade pip wheel

echo "[pip] installing setuptools with pkg_resources support"
"${PYTHON_BIN}" -m pip install --upgrade --ignore-installed "setuptools>=75,<82"

"${PYTHON_BIN}" - <<'PY'
import pkg_resources
import sys

print(f"[check] pkg_resources: {pkg_resources.__file__}")
if "/usr/lib/python3/dist-packages/pkg_resources" in pkg_resources.__file__:
    raise SystemExit(
        "Old system pkg_resources is still active. "
        "Use a virtualenv/conda env, or reinstall setuptools into the active interpreter."
    )
PY

echo "[pip] installing core build deps"
"${PYTHON_BIN}" -m pip install cmake ninja packaging tabulate

echo "[pip] installing PyTorch (${CUDA_VARIANT})"
"${PYTHON_BIN}" -m pip install \
    --index-url "${TORCH_INDEX_URL}" \
    "torch==2.8.*" \
    "torchvision>=0.23.0,<0.24.0" \
    torchaudio

echo "[pip] installing llama.cpp conversion deps"
"${PYTHON_BIN}" -m pip install -r "${ROOT_DIR}/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt"

echo "[pip] installing Qwen3-VL-Embedding reference package"
"${PYTHON_BIN}" -m pip install -e "${ROOT_DIR}/Qwen3-VL-Embedding"

if [[ "${INSTALL_VLMEVAL}" -eq 1 ]]; then
    if [[ ! -f "${VLMEVAL_DIR}/run.py" ]]; then
        echo "[setup] cloning VLMEvalKit into ${VLMEVAL_DIR}"
        rm -rf "${VLMEVAL_DIR}"
        git clone --depth 1 https://github.com/open-compass/VLMEvalKit "${VLMEVAL_DIR}"
    fi
    echo "[pip] installing VLMEvalKit"
    "${PYTHON_BIN}" -m pip install -e "${VLMEVAL_DIR}"
fi

ENV_DESC="current python"
if [[ "${USE_CONDA}" -eq 1 ]]; then
    ENV_DESC="${ENV_NAME}"
fi

cat <<EOF

Done.
python: ${PYTHON_BIN}
env   : ${ENV_DESC}
cuda  : ${CUDA_VARIANT}

Next:
  1. Build both llama-server binaries:
     ${ROOT_DIR}/scripts/build_both_llama_servers.sh
  2. Run VLMEval compare:
     ${ROOT_DIR}/scripts/run_vlmeval_llama_server_compare.sh --preset doc
EOF
