#!/bin/bash
# 综合基准测试脚本 - 自动化执行完整测试流程

set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_ROOT=$(dirname $SCRIPT_DIR)

# 使用 adb 从 Windows (WSL2)
if [ -x "/mnt/e/andorid/adb/adb.exe" ]; then
    adb() {
        /mnt/e/andorid/adb/adb.exe "$@"
    }
fi

# 创建结果目录
RESULTS_DIR="$PROJECT_ROOT/results/comprehensive_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "  综合基准测试脚本"
echo "=========================================="
echo "结果目录: $RESULTS_DIR"
echo ""

# 定义测试配置
BACKENDS=("mnn" "ort")
MODELS=("mobilenetv2" "resnet50")
PRECISIONS=("fp32")
THREADS=(1 2 4)
RUNS=100
WARMUP=10

# 记录开始时间
START_TIME=$(date +%s)

# 运行测试
for backend in "${BACKENDS[@]}"; do
    for model in "${MODELS[@]}"; do
        for precision in "${PRECISIONS[@]}"; do
            for threads in "${THREADS[@]}"; do
                echo "=========================================="
                echo "测试: $backend $model $precision $threads threads"
                echo "=========================================="

                # 运行基准测试
                "$SCRIPT_DIR/run_benchmark_android.sh" \
                    --backend "$backend" \
                    --model "$model" \
                    --precision "$precision" \
                    --threads "$threads" \
                    --runs "$RUNS" \
                    --warmup "$WARMUP" 2>&1 | tee "$RESULTS_DIR/${backend}_${model}_${precision}_${threads}threads.log"

                echo ""
            done
        done
    done
done

# 记录结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "=========================================="
echo "测试完成!"
echo "总耗时: $DURATION 秒"
echo "结果目录: $RESULTS_DIR"
echo "=========================================="

# 生成报告
echo "生成测试报告..."
"$SCRIPT_DIR/generate_report.py" "$RESULTS_DIR"

echo "完成!"
