#!/bin/bash
# Build for Android arm64-v8a (Snapdragon 865)

set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_ROOT=$(dirname $SCRIPT_DIR)

# Parse arguments
BUILD_TYPE="Release"
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --debug    Build with debug symbols (default: Release)"
            echo "  --release  Build release version"
            echo "  --help     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$ANDROID_NDK" ]; then
    echo "ERROR: ANDROID_NDK environment variable is not set"
    echo "Please export ANDROID_NDK=/path/to/android-ndk-rxxxxx first"
    echo ""
    echo "Example:"
    echo "  export ANDROID_NDK=~/android-ndk-r25c"
    echo "  ./scripts/build_android.sh"
    exit 1
fi

if [ ! -d "$ANDROID_NDK" ]; then
    echo "ERROR: ANDROID_NDK directory does not exist: $ANDROID_NDK"
    exit 1
fi

echo "=== Building for Android arm64-v8a ==="
echo "ANDROID_NDK: $ANDROID_NDK"
echo "Project root: $PROJECT_ROOT"
echo "Build type: $BUILD_TYPE"

# Create build directory based on build type
if [ "$BUILD_TYPE" = "Debug" ]; then
    BUILD_DIR="$PROJECT_ROOT/build_android_debug"
else
    BUILD_DIR="$PROJECT_ROOT/build_android"
fi

mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-29 \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DBENCHMARK_NCNN=OFF \
    -DBENCHMARK_MNN=ON \
    -DBENCHMARK_TFLITE=OFF \
    -DBENCHMARK_TNN=OFF \
    -DBENCHMARK_ORT=ON \
    -DBENCHMARK_TVM=ON \
    -DBENCHMARK_LLAMACPP=ON
    -DBENCHMARK_MINDSPORE_LITE=OFF

# Build
make -j$(nproc)

echo "=== Build complete ==="
echo "Executable: $BUILD_DIR/src/benchmark_inference"
echo ""
echo "To run on device:"
echo "  adb push $PROJECT_ROOT/build_android/src/benchmark_inference /data/local/tmp/"
echo "  adb push models/ /data/local/tmp/benchmark-models/"
echo "  adb shell"
echo "  cd /data/local/tmp"
echo "  chmod +x ./benchmark_inference"
echo "  ./benchmark_inference --backend mnn --model mobilenetv2 --precision fp32 --threads 1 --warmup 10 --runs 100"
