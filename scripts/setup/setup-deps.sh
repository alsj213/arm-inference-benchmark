#!/bin/bash
#
# 一键设置所有第三方依赖
# Usage: ./scripts/setup_deps.sh [--all|--minimal|--ncnn|--mnn|--tflite|--ort|--tvm]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
THIRD_PARTY="$PROJECT_ROOT/third_party"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --all         Setup all frameworks (default)"
    echo "  --minimal     Setup only MNN + ONNX Runtime"
    echo "  --mnn         Setup MNN only"
    echo "  --ort         Setup ONNX Runtime only"
    echo "  --tvm         Setup TVM only"
    echo "  --help        Show this help"
    echo ""
    exit 1
}

# Parse arguments
SETUP_ALL=true
SETUP_MNN=false
SETUP_ORT=false
SETUP_TVM=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all) SETUP_ALL=true; shift ;;
        --minimal) SETUP_ALL=false; SETUP_MNN=true; SETUP_ORT=true; shift ;;
        --mnn) SETUP_ALL=false; SETUP_MNN=true; shift ;;
        --ort) SETUP_ALL=false; SETUP_ORT=true; shift ;;
        --tvm) SETUP_ALL=false; SETUP_TVM=true; shift ;;
        --help) usage ;;
        *) error "Unknown option: $1" ;;
    esac
done

cd "$PROJECT_ROOT"

info "========================================"
info "   ARM Inference Benchmark - Setup Dependencies"
info "========================================"
echo ""

# Function to setup git submodule
setup_submodule() {
    local name="$1"
    local repo="$2"
    local path="$3"

    info "Setting up $name..."

    if [ -d "$path/.git" ]; then
        info "  $name already exists, updating..."
        cd "$path" && git pull origin master 2>/dev/null || git pull origin main 2>/dev/null
        cd "$PROJECT_ROOT"
    else
        info "  Adding $name as submodule..."
        git submodule add "$repo" "$path" 2>/dev/null || {
            warn "  Failed to add submodule, cloning directly..."
            rm -rf "$path"
            git clone "$repo" "$path"
        }
    fi

    info "  ✓ $name setup complete"
}

# Function to download ONNX Runtime
download_onnxruntime() {
    info "Setting up ONNX Runtime..."
    local ORT_VERSION="1.16.3"
    local ORT_DIR="$THIRD_PARTY/onnxruntime"
    local ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-android-${ORT_VERSION}.aar"

    mkdir -p "$ORT_DIR"
    cd "$ORT_DIR"

    if [ -f "lib-android/aarch64/libonnxruntime.so" ]; then
        info "  ONNX Runtime already exists"
        return
    fi

    info "  Downloading ONNX Runtime $ORT_VERSION..."
    curl -L -o onnxruntime-android.aar "$ORT_URL" 2>/dev/null || wget -q "$ORT_URL" -O onnxruntime-android.aar 2>/dev/null

    if [ ! -f "onnxruntime-android.aar" ]; then
        error "Failed to download ONNX Runtime. Please download manually from $ORT_URL"
    fi

    info "  Extracting..."
    mkdir -p temp
    cd temp
    unzip -q ../onnxruntime-android.aar

    # Extract headers
    cd ..
    mkdir -p include
    cp -r temp/headers/* include/ 2>/dev/null || true

    # Extract libraries
    mkdir -p lib-android/aarch64
    cp temp/jni/arm64-v8a/libonnxruntime.so lib-android/aarch64/

    # Cleanup
    rm -rf temp onnxruntime-android.aar

    info "  ✓ ONNX Runtime setup complete"
    cd "$PROJECT_ROOT"
}

# Function to download TFLite

echo ""
info "Project root: $PROJECT_ROOT"
info "Third party dir: $THIRD_PARTY"
echo ""

fi

# Setup MNN

# Setup ONNX Runtime
if [ "$SETUP_ALL" = true ] || [ "$SETUP_ORT" = true ]; then
    download_onnxruntime
fi

# Setup TVM (note: TVM is large, optional)
if [ "$SETUP_ALL" = true ] || [ "$SETUP_TVM" = true ]; then
    info "Setting up Apache TVM..."
    warn "  TVM is very large (~1GB) and takes time to clone"
    read -p "  Continue with TVM setup? (y/N) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        setup_submodule "TVM" "https://github.com/apache/tvm.git" "$THIRD_PARTY/tvm"
        info "  Note: You need to build TVM runtime manually. See third_party/README.md"
    else
        info "  Skipping TVM setup"
    fi
fi

echo ""
info "========================================"
info "   Setup complete!"
info "========================================"
echo ""
echo "Next steps:"
echo "  1. Build the project: ./scripts/build_android.sh"
echo "  2. Run benchmarks: ./scripts/run_benchmark_android.sh"
echo ""
