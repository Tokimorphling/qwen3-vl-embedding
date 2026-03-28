#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MOD_LLAMA_DIR="${ROOT_DIR}/llama.cpp"
CLEAN_LLAMA_DIR="${ROOT_DIR}/llama.cpp-clean"
VLMEVAL_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/VLMEvalKit"
WORK_DIR="${ROOT_DIR}/logs/vlmeval_compare_$(date +%Y%m%d_%H%M%S)"

HF_REPO="Qwen/Qwen3-VL-4B-Instruct-GGUF:Q4_K_M"
MODEL_FILE=""
MMPROJ_FILE=""
MODEL_ALIAS="qwen3-vl-local"
DATASETS="MME,MMStar,HallusionBench,RealWorldQA"
PRESET=""
MOD_PORT=18081
CLEAN_PORT=18082
API_KEY="sk-local-vlmeval"
CTX_SIZE=8192
GPU_LAYERS="auto"
API_NPROC=4
INSTALL_VLMEVAL=1
BUILD_IF_MISSING=1
KEEP_SERVERS=0

usage() {
    cat <<'EOF'
Usage:
  run_vlmeval_llama_server_compare.sh [options]

Options:
  --mod-llama-dir PATH       Modified llama.cpp dir. Default: ./llama.cpp
  --clean-llama-dir PATH     Clean/original llama.cpp dir. Default: ./llama.cpp-clean
  --vlmeval-dir PATH         VLMEvalKit checkout dir. Default: $XDG_CACHE_HOME/VLMEvalKit
  --work-dir PATH            Output dir for VLMEvalKit and summary files
  --hf-repo REPO[:QUANT]     HF GGUF repo for both servers
  --model-file PATH          Local GGUF model file for both servers
  --mmproj-file PATH         Local mmproj GGUF file for both servers
  --preset NAME              Dataset preset: quick|doc|general|all
  --datasets CSV             Comma-separated datasets. Default: MME,MMStar,HallusionBench,RealWorldQA
  --ctx-size N               llama-server ctx size. Default: 8192
  --gpu-layers N|auto|all    llama-server --gpu-layers. Default: auto
  --mod-port PORT            Modified server port. Default: 18081
  --clean-port PORT          Clean server port. Default: 18082
  --api-key KEY              Shared local API key. Default: sk-local-vlmeval
  --api-nproc N              VLMEvalKit API thread count. Default: 4
  --model-alias NAME         OpenAI model id exposed by both servers. Default: qwen3-vl-local
  --no-install-vlmeval       Do not clone/install VLMEvalKit automatically
  --no-build                 Do not build llama-server if binary is missing
  --keep-servers             Keep the two llama-server processes alive after evaluation
  -h, --help                 Show this help

Notes:
  - quick   : fast exact-match smoke test
  - doc     : more sensitive to OCR / document / perception changes
  - general : stronger broad VLM comparison
  - all     : union of all supported datasets in this wrapper
  - If --model-file is provided, the script starts both servers with -m/--mmproj and ignores --hf-repo.
  - If both --preset and --datasets are provided, --datasets wins.
  - Default datasets are chosen to work with exact matching, so no judge API is required.
  - The script starts two local OpenAI-compatible llama-server instances, runs VLMEvalKit once
    with both models, then writes a markdown summary into WORK_DIR/summary.md.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mod-llama-dir)
            MOD_LLAMA_DIR="$2"
            shift 2
            ;;
        --clean-llama-dir)
            CLEAN_LLAMA_DIR="$2"
            shift 2
            ;;
        --vlmeval-dir)
            VLMEVAL_DIR="$2"
            shift 2
            ;;
        --work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        --hf-repo)
            HF_REPO="$2"
            shift 2
            ;;
        --model-file)
            MODEL_FILE="$2"
            shift 2
            ;;
        --mmproj-file)
            MMPROJ_FILE="$2"
            shift 2
            ;;
        --preset)
            PRESET="$2"
            shift 2
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --ctx-size)
            CTX_SIZE="$2"
            shift 2
            ;;
        --gpu-layers)
            GPU_LAYERS="$2"
            shift 2
            ;;
        --mod-port)
            MOD_PORT="$2"
            shift 2
            ;;
        --clean-port)
            CLEAN_PORT="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --api-nproc)
            API_NPROC="$2"
            shift 2
            ;;
        --model-alias)
            MODEL_ALIAS="$2"
            shift 2
            ;;
        --no-install-vlmeval)
            INSTALL_VLMEVAL=0
            shift
            ;;
        --no-build)
            BUILD_IF_MISSING=0
            shift
            ;;
        --keep-servers)
            KEEP_SERVERS=1
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

MOD_LLAMA_DIR="$(cd "${MOD_LLAMA_DIR}" && pwd)"
CLEAN_LLAMA_DIR="$(cd "${CLEAN_LLAMA_DIR}" && pwd)"
VLMEVAL_DIR="${VLMEVAL_DIR/#\~/$HOME}"
WORK_DIR="${WORK_DIR/#\~/$HOME}"
MODEL_FILE="${MODEL_FILE/#\~/$HOME}"
MMPROJ_FILE="${MMPROJ_FILE/#\~/$HOME}"
WORK_DIR="$(python - <<'PY' "$WORK_DIR"
import os, sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
)"
if [[ -n "${MODEL_FILE}" ]]; then
    MODEL_FILE="$(python - <<'PY' "$MODEL_FILE"
import os, sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
)"
fi
if [[ -n "${MMPROJ_FILE}" ]]; then
    MMPROJ_FILE="$(python - <<'PY' "$MMPROJ_FILE"
import os, sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
)"
fi

if [[ -n "${MODEL_FILE}" && ! -f "${MODEL_FILE}" ]]; then
    echo "Local model file not found: ${MODEL_FILE}" >&2
    exit 1
fi
if [[ -n "${MMPROJ_FILE}" && ! -f "${MMPROJ_FILE}" ]]; then
    echo "Local mmproj file not found: ${MMPROJ_FILE}" >&2
    exit 1
fi
if [[ -z "${MODEL_FILE}" && -n "${MMPROJ_FILE}" ]]; then
    echo "--mmproj-file requires --model-file" >&2
    exit 2
fi

if [[ -n "${PRESET}" ]]; then
    case "${PRESET}" in
        quick)
            DATASETS="MME,MMStar,HallusionBench,RealWorldQA"
            ;;
        doc)
            DATASETS="DocVQA_VAL,MME-RealWorld-Lite,RealWorldQA,MMBench_DEV_EN"
            ;;
        general)
            DATASETS="MMStar,RealWorldQA,MMBench_DEV_EN,MMMU_DEV_VAL"
            ;;
        all)
            DATASETS="MME,HallusionBench,MMStar,RealWorldQA,MMBench_DEV_EN,MME-RealWorld-Lite,MMMU_DEV_VAL,DocVQA_VAL"
            ;;
        *)
            echo "Unknown preset: ${PRESET}" >&2
            exit 2
            ;;
    esac
fi

MOD_BIN="${MOD_LLAMA_DIR}/build/bin/llama-server"
CLEAN_BIN="${CLEAN_LLAMA_DIR}/build/bin/llama-server"
CONFIG_JSON="${WORK_DIR}/vlmeval_config.json"
SUMMARY_MD="${WORK_DIR}/summary.md"
MOD_LOG="${WORK_DIR}/llama_server_mod.log"
CLEAN_LOG="${WORK_DIR}/llama_server_clean.log"

MOD_PID=""
CLEAN_PID=""

cleanup() {
    if [[ "${KEEP_SERVERS}" -eq 0 ]]; then
        [[ -n "${MOD_PID}" ]] && kill "${MOD_PID}" >/dev/null 2>&1 || true
        [[ -n "${CLEAN_PID}" ]] && kill "${CLEAN_PID}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Missing command: $1" >&2
        exit 1
    }
}

ensure_llama_server() {
    local llama_dir="$1"
    local bin_path="$2"

    if [[ -x "${bin_path}" ]]; then
        return
    fi
    if [[ "${BUILD_IF_MISSING}" -ne 1 ]]; then
        echo "Missing binary: ${bin_path}" >&2
        exit 1
    fi

    echo "[build] building llama-server in ${llama_dir}"
    if [[ ! -f "${llama_dir}/build/CMakeCache.txt" ]]; then
        cmake -S "${llama_dir}" -B "${llama_dir}/build" -DGGML_CUDA=ON
    fi
    cmake --build "${llama_dir}/build" --target llama-server -j4
}

ensure_vlmeval() {
    if [[ -f "${VLMEVAL_DIR}/run.py" ]]; then
        return
    fi
    if [[ "${INSTALL_VLMEVAL}" -ne 1 ]]; then
        echo "VLMEvalKit not found at ${VLMEVAL_DIR}" >&2
        exit 1
    fi

    echo "[setup] cloning VLMEvalKit into ${VLMEVAL_DIR}"
    rm -rf "${VLMEVAL_DIR}"
    git clone --depth 1 https://github.com/open-compass/VLMEvalKit "${VLMEVAL_DIR}"
}

install_vlmeval() {
    if [[ "${INSTALL_VLMEVAL}" -ne 1 ]]; then
        return
    fi
    echo "[setup] installing VLMEvalKit"
    python -m pip install -e "${VLMEVAL_DIR}"
}

wait_for_server() {
    local name="$1"
    local port="$2"
    local url="http://127.0.0.1:${port}/v1/models"

    for _ in $(seq 1 180); do
        if curl -fsS -H "Authorization: Bearer ${API_KEY}" "${url}" >/dev/null 2>&1; then
            echo "[server] ${name} ready at ${url}"
            return 0
        fi
        sleep 2
    done

    echo "[server] ${name} failed to start on port ${port}" >&2
    return 1
}

start_server() {
    local name="$1"
    local bin_path="$2"
    local port="$3"
    local log_path="$4"

    echo "[server] starting ${name} on :${port}"
    if [[ -n "${MODEL_FILE}" ]]; then
        if [[ -n "${MMPROJ_FILE}" ]]; then
            "${bin_path}" \
                -m "${MODEL_FILE}" \
                --mmproj "${MMPROJ_FILE}" \
                --host 127.0.0.1 \
                --port "${port}" \
                --alias "${MODEL_ALIAS}" \
                --api-key "${API_KEY}" \
                --ctx-size "${CTX_SIZE}" \
                --gpu-layers "${GPU_LAYERS}" \
                --jinja \
                --temp 0 \
                --no-webui \
                >"${log_path}" 2>&1 &
        else
            "${bin_path}" \
                -m "${MODEL_FILE}" \
                --host 127.0.0.1 \
                --port "${port}" \
                --alias "${MODEL_ALIAS}" \
                --api-key "${API_KEY}" \
                --ctx-size "${CTX_SIZE}" \
                --gpu-layers "${GPU_LAYERS}" \
                --jinja \
                --temp 0 \
                --no-webui \
                >"${log_path}" 2>&1 &
        fi
    else
        "${bin_path}" \
            --hf-repo "${HF_REPO}" \
            --host 127.0.0.1 \
            --port "${port}" \
            --alias "${MODEL_ALIAS}" \
            --api-key "${API_KEY}" \
            --ctx-size "${CTX_SIZE}" \
            --gpu-layers "${GPU_LAYERS}" \
            --jinja \
            --temp 0 \
            --no-webui \
            >"${log_path}" 2>&1 &
    fi

    local pid=$!
    if [[ "${name}" == "modified" ]]; then
        MOD_PID="${pid}"
    else
        CLEAN_PID="${pid}"
    fi
}

write_config() {
    python - <<'PY' "${CONFIG_JSON}" "${MODEL_ALIAS}" "${API_KEY}" "${MOD_PORT}" "${CLEAN_PORT}" "${DATASETS}"
import json
import sys

out_path, model_alias, api_key, mod_port, clean_port, datasets_csv = sys.argv[1:]
dataset_map = {
    "MME": {"class": "ImageYORNDataset", "dataset": "MME"},
    "HallusionBench": {"class": "ImageYORNDataset", "dataset": "HallusionBench"},
    "MMStar": {"class": "ImageMCQDataset", "dataset": "MMStar"},
    "RealWorldQA": {"class": "ImageMCQDataset", "dataset": "RealWorldQA"},
    "MMBench_DEV_EN": {"class": "ImageMCQDataset", "dataset": "MMBench_DEV_EN"},
    "MME-RealWorld-Lite": {"class": "MMERealWorld", "dataset": "MME-RealWorld-Lite"},
    "MMMU_DEV_VAL": {"class": "MMMUDataset", "dataset": "MMMU_DEV_VAL"},
    "DocVQA_VAL": {"class": "ImageVQADataset", "dataset": "DocVQA_VAL"},
}

selected = [x.strip() for x in datasets_csv.split(",") if x.strip()]
unknown = [x for x in selected if x not in dataset_map]
if unknown:
    raise SystemExit(f"Unsupported datasets in wrapper: {unknown}")

cfg = {
    "model": {
        "llama_mod": {
            "class": "GPT4V",
            "model": model_alias,
            "key": api_key,
            "api_base": f"http://127.0.0.1:{mod_port}/v1/chat/completions",
            "temperature": 0,
            "img_size": -1,
            "img_detail": "high",
            "retry": 5,
            "verbose": False,
            "max_tokens": 1024,
        },
        "llama_clean": {
            "class": "GPT4V",
            "model": model_alias,
            "key": api_key,
            "api_base": f"http://127.0.0.1:{clean_port}/v1/chat/completions",
            "temperature": 0,
            "img_size": -1,
            "img_detail": "high",
            "retry": 5,
            "verbose": False,
            "max_tokens": 1024,
        },
    },
    "data": {name: dataset_map[name] for name in selected},
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
print(out_path)
PY
}

require_cmd python
require_cmd cmake
require_cmd curl
require_cmd git

mkdir -p "${WORK_DIR}"
echo "[config] datasets=${DATASETS}"
ensure_llama_server "${MOD_LLAMA_DIR}" "${MOD_BIN}"
ensure_llama_server "${CLEAN_LLAMA_DIR}" "${CLEAN_BIN}"
ensure_vlmeval
install_vlmeval

write_config >/dev/null
start_server "modified" "${MOD_BIN}" "${MOD_PORT}" "${MOD_LOG}"
start_server "clean" "${CLEAN_BIN}" "${CLEAN_PORT}" "${CLEAN_LOG}"

wait_for_server "modified" "${MOD_PORT}"
wait_for_server "clean" "${CLEAN_PORT}"

echo "[eval] running VLMEvalKit"
(
    export OPENAI_API_KEY=""
    export OPENAI_API_BASE=""
    export LOCAL_LLM=""
    python "${VLMEVAL_DIR}/run.py" \
        --config "${CONFIG_JSON}" \
        --work-dir "${WORK_DIR}" \
        --mode all \
        --api-nproc "${API_NPROC}"
)

echo "[summary] writing ${SUMMARY_MD}"
python "${ROOT_DIR}/scripts/summarize_vlmeval_results.py" \
    --work-dir "${WORK_DIR}" \
    --output "${SUMMARY_MD}"

echo ""
echo "Done."
echo "work_dir: ${WORK_DIR}"
echo "summary : ${SUMMARY_MD}"
echo "config  : ${CONFIG_JSON}"
echo "logs    : ${MOD_LOG} ${CLEAN_LOG}"
