#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# 构建 TVM Runtime + RPC Server (Android ARM64)
#
# 用于 MetaSchedule auto-tuning：TVM 在主机上搜索调度方案，
# 交叉编译后通过 RPC 在 Android 设备上实测延迟，反馈给代价模型。
#
# 用法:
#   ./tools/tvm/build_android_rpc.sh          # 构建
#   ./tools/tvm/build_android_rpc.sh --deploy # 构建 + 推送到设备
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")/.."
TVM_ROOT="$PROJECT_ROOT/third_party/tvm"
BUILD_DIR="$TVM_ROOT/build_android_rpc"
NDK_ROOT="${ANDROID_NDK:-/home/liu/android-ndk}"
NDK_TOOLCHAIN="$NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64"
NDK_SYSROOT="$NDK_TOOLCHAIN/sysroot"
TARGET_TRIPLE="aarch64-linux-android21"

# NDK 编译器
CC="$NDK_TOOLCHAIN/bin/${TARGET_TRIPLE}-clang"
CXX="$NDK_TOOLCHAIN/bin/${TARGET_TRIPLE}-clang++"

echo "=== 构建 TVM Android RPC Server ==="
echo "  NDK: $NDK_ROOT"
echo "  Build dir: $BUILD_DIR"
echo "  Target: $TARGET_TRIPLE"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# ── 1. 构建 libtvm_runtime.so (Android ARM64) ──
echo ""
echo "[1/3] 构建 libtvm_runtime.so (aarch64)..."
RUNTIME_SOURCES=(
    "$TVM_ROOT/src/runtime/builtin_fp16.cc"
    "$TVM_ROOT/src/runtime/c_runtime_api.cc"
    "$TVM_ROOT/src/runtime/const_loader_module.cc"
    "$TVM_ROOT/src/runtime/container_ffi.cc"
    "$TVM_ROOT/src/runtime/cpu_device_api.cc"
    "$TVM_ROOT/src/runtime/debug.cc"
    "$TVM_ROOT/src/runtime/disk_file_utils.cc"
    "$TVM_ROOT/src/runtime/dso_library.cc"
    "$TVM_ROOT/src/runtime/file_utils.cc"
    "$TVM_ROOT/src/runtime/library_module.cc"
    "$TVM_ROOT/src/runtime/logging.cc"
    "$TVM_ROOT/src/runtime/metadata.cc"
    "$TVM_ROOT/src/runtime/module.cc"
    "$TVM_ROOT/src/runtime/name_transforms.cc"
    "$TVM_ROOT/src/runtime/ndarray.cc"
    "$TVM_ROOT/src/runtime/nvtx.cc"
    "$TVM_ROOT/src/runtime/object_ffi.cc"
    "$TVM_ROOT/src/runtime/packed_func.cc"
    "$TVM_ROOT/src/runtime/profiling.cc"
    "$TVM_ROOT/src/runtime/registry.cc"
    "$TVM_ROOT/src/runtime/relax_vm/bytecode.cc"
    "$TVM_ROOT/src/runtime/relax_vm/executable.cc"
    "$TVM_ROOT/src/runtime/relax_vm/lm_support.cc"
    "$TVM_ROOT/src/runtime/relax_vm/ndarray_cache_support.cc"
    "$TVM_ROOT/src/runtime/relax_vm/vm.cc"
    "$TVM_ROOT/src/runtime/rpc/rpc_channel.cc"
    "$TVM_ROOT/src/runtime/rpc/rpc_endpoint.cc"
    "$TVM_ROOT/src/runtime/rpc/rpc_event_impl.cc"
    "$TVM_ROOT/src/runtime/rpc/rpc_local_session.cc"
    "$TVM_ROOT/src/runtime/rpc/rpc_module.cc"
    "$TVM_ROOT/src/runtime/rpc/rpc_server_env.cc"
    "$TVM_ROOT/src/runtime/rpc/rpc_session.cc"
    "$TVM_ROOT/src/runtime/rpc/rpc_socket_impl.cc"
    "$TVM_ROOT/src/runtime/source_utils.cc"
    "$TVM_ROOT/src/runtime/static_library.cc"
    "$TVM_ROOT/src/runtime/system_library.cc"
    "$TVM_ROOT/src/runtime/thread_pool.cc"
    "$TVM_ROOT/src/runtime/threading_backend.cc"
    "$TVM_ROOT/src/runtime/workspace_pool.cc"
    "$TVM_ROOT/src/runtime/minrpc/minrpc_logger.cc"
    "$TVM_ROOT/src/runtime/minrpc/rpc_reference.cc"
    "$TVM_ROOT/src/support/socket.cc"
    "$TVM_ROOT/src/support/utils.cc"
)

CFLAGS="-O2 -fPIC -std=c++17 -DNDEBUG"
CFLAGS="$CFLAGS -DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>"
CFLAGS="$CFLAGS -DTVM_LLVM_VERSION=150"
CFLAGS="$CFLAGS -I$TVM_ROOT/include"
CFLAGS="$CFLAGS -I$TVM_ROOT/3rdparty/tvm-ffi/include"
CFLAGS="$CFLAGS -I$TVM_ROOT/3rdparty/dlpack/include"
CFLAGS="$CFLAGS -I$TVM_ROOT/3rdparty/dmlc-core/include"
CFLAGS="$CFLAGS -I$TVM_ROOT/src"
CFLAGS="$CFLAGS -I$TVM_ROOT/3rdparty/compiler-rt"
# 用于 <builtin_fp16.h> 直接 include
CFLAGS="$CFLAGS -I$TVM_ROOT/include/tvm/runtime"

OBJ_DIR="$BUILD_DIR/runtime_objs"
mkdir -p "$OBJ_DIR"

OBJECTS=()
for src in "${RUNTIME_SOURCES[@]}"; do
    if [ -f "$src" ]; then
        obj="$OBJ_DIR/$(basename "$src" .cc).o"
        OBJECTS+=("$obj")
        [ -f "$obj" ] && continue  # 已编译则跳过
        $CXX $CFLAGS -c "$src" -o "$obj" &
    fi
done
wait

echo "  链接 libtvm_runtime.so ..."
$CXX -shared -o "$BUILD_DIR/libtvm_runtime.so" "${OBJECTS[@]}" -lpthread
file "$BUILD_DIR/libtvm_runtime.so"

# ── 2. 构建 tvm_rpc server ──
echo ""
echo "[2/3] 构建 tvm_rpc ..."
RPC_CFLAGS="$CFLAGS -I$TVM_ROOT/apps/cpp_rpc"

$CXX $RPC_CFLAGS \
    -c "$TVM_ROOT/apps/cpp_rpc/main.cc" \
    -o "$OBJ_DIR/tvm_rpc_main.o"

$CXX $RPC_CFLAGS \
    -c "$TVM_ROOT/apps/cpp_rpc/rpc_env.cc" \
    -o "$OBJ_DIR/rpc_env.o"

$CXX $RPC_CFLAGS \
    -c "$TVM_ROOT/apps/cpp_rpc/rpc_server.cc" \
    -o "$OBJ_DIR/rpc_server.o"

$CXX -pie -fPIE \
    "$OBJ_DIR/tvm_rpc_main.o" \
    "$OBJ_DIR/rpc_env.o" \
    "$OBJ_DIR/rpc_server.o" \
    -L"$BUILD_DIR" -ltvm_runtime -lpthread \
    -Wl,-rpath,/data/local/tmp \
    -o "$BUILD_DIR/tvm_rpc"

file "$BUILD_DIR/tvm_rpc"

# ── 3. 推送到设备 ──
CMD="${1:-}"
if [ "$CMD" = "--deploy" ]; then
    echo ""
    echo "[3/3] 推送到设备..."
    ADB="${ADB_PATH:-adb}"

    # 推送到设备
    $ADB push "$BUILD_DIR/libtvm_runtime.so" /data/local/tmp/
    $ADB push "$BUILD_DIR/tvm_rpc" /data/local/tmp/
    $ADB shell chmod +x /data/local/tmp/tvm_rpc

    # 验证
    echo "=== 设备端验证 ==="
    $ADB shell "file /data/local/tmp/tvm_rpc"
    $ADB shell "/data/local/tmp/tvm_rpc" 2>&1 | head -5 || true

    echo ""
    echo "=== 部署完成 ==="
    echo ""
    echo "启动 RPC 调优:"
    echo "  1. 在主机启动 tracker:"
    echo "     python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190"
    echo ""
    echo "  2. 在设备启动 RPC server (需转发端口):"
    echo "     adb reverse tcp:9190 tcp:9190"
    echo "     adb shell 'cd /data/local/tmp && ./tvm_rpc server --host=127.0.0.1 --port=9190 &'"
    echo ""
    echo "  3. 运行调优:"
    echo "     python tools/tvm/compile_all_models.py mobilenetv2 --tune"
else
    echo ""
    echo "=== 构建完成 ==="
    echo ""
    echo "产物:"
    echo "  $BUILD_DIR/libtvm_runtime.so"
    echo "  $BUILD_DIR/tvm_rpc"
    echo ""
    echo "部署到设备: $0 --deploy"
fi
