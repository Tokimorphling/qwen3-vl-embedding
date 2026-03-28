#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOD_DIR="${ROOT_DIR}/llama.cpp"
CLEAN_DIR="${ROOT_DIR}/llama.cpp-clean"
JOBS=4
CMAKE_ARGS=("-DGGML_CUDA=ON")
TARGET="llama-server"

usage() {
    cat <<'EOF'
Usage:
  build_both_llama_servers.sh [options]

Options:
  --mod-dir PATH        Modified llama.cpp directory. Default: ./llama.cpp
  --clean-dir PATH      Clean/original llama.cpp directory. Default: ./llama.cpp-clean
  --jobs N              Build parallelism. Default: 4
  --target NAME         CMake target to build. Default: llama-server
  --cpu-only            Configure without CUDA
  --cmake-arg ARG       Extra cmake configure arg. Can be repeated.
  -h, --help            Show this help

Examples:
  ./scripts/build_both_llama_servers.sh
  ./scripts/build_both_llama_servers.sh --jobs 8
  ./scripts/build_both_llama_servers.sh --target llama-mtmd-cli
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mod-dir)
            MOD_DIR="$2"
            shift 2
            ;;
        --clean-dir)
            CLEAN_DIR="$2"
            shift 2
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        --cpu-only)
            CMAKE_ARGS=()
            shift
            ;;
        --cmake-arg)
            CMAKE_ARGS+=("$2")
            shift 2
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

MOD_DIR="${MOD_DIR/#\~/$HOME}"
CLEAN_DIR="${CLEAN_DIR/#\~/$HOME}"
MOD_DIR="$(cd "${MOD_DIR}" && pwd)"
CLEAN_DIR="$(cd "${CLEAN_DIR}" && pwd)"

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Missing command: $1" >&2
        exit 1
    }
}

configure_and_build() {
    local src_dir="$1"
    local build_dir="${src_dir}/build"

    echo "[configure] ${src_dir}"
    cmake -S "${src_dir}" -B "${build_dir}" "${CMAKE_ARGS[@]}"

    echo "[build] ${src_dir} target=${TARGET} jobs=${JOBS}"
    cmake --build "${build_dir}" --target "${TARGET}" -j"${JOBS}"
}

require_cmd cmake

configure_and_build "${MOD_DIR}"
configure_and_build "${CLEAN_DIR}"

echo ""
echo "Built target '${TARGET}' in:"
echo "  ${MOD_DIR}/build"
echo "  ${CLEAN_DIR}/build"
