#!/bin/bash
# Run full benchmark on all combinations

set -e

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_ROOT=$(dirname $SCRIPT_DIR)
BENCHMARK=$PROJECT_ROOT/build/benchmark_inference
RESULTS_DIR=$PROJECT_ROOT/results/sm8250

mkdir -p $RESULTS_DIR

echo "=== Starting Benchmark (SDM8250 - Snapdragon 865) ==="
date

# Run combinations
BACKENDS=("ncnn" "mnn" "tnn" "tflite" "qnn" "tvm")
MODELS=("mobilenetv2" "resnet50" "yolov8n" "bert")
PRECISIONS=("fp32" "fp16" "int8")
THREADS=(1 4)

RESULT_FILE=$RESULTS_DIR/results.csv
echo "backend,model,precision,threads,use_gpu,init_time_ms,p50_ms,p90_ms,p99_ms,mean_ms,throughput_fps,peak_mem_kb" > $RESULT_FILE

for threads in "${THREADS[@]}"; do
    for precision in "${PRECISIONS[@]}"; do
        for backend in "${BACKENDS[@]}"; do
            for model in "${MODELS[@]}"; do
                echo
                echo "-----------------------------------------------------"
                echo "Running: backend=$backend model=$model precision=$precision threads=$threads"
                echo

                OUTPUT=$($BENCHMARK \
                    --backend $backend \
                    --model $model \
                    --precision $precision \
                    --threads $threads \
                    --warmup 10 \
                    --runs 100)

                echo "$OUTPUT"

                # Parse output and append to CSV
                # This is a simple parser, adjust as needed
                INIT_TIME=$(echo "$OUTPUT" | grep "Init time" | awk '{print $3}')
                P50=$(echo "$OUTPUT" | grep "P50" | awk '{print $3}')
                P90=$(echo "$OUTPUT" | grep "P90" | awk '{print $3}')
                P99=$(echo "$OUTPUT" | grep "P99" | awk '{print $3}')
                MEAN=$(echo "$OUTPUT" | grep "Mean" | awk '{print $3}')
                FPS=$(echo "$OUTPUT" | grep "Throughput" | awk '{print $3}')
                MEM=$(echo "$OUTPUT" | grep "Peak mem" | awk '{print $3}')

                echo "$backend,$model,$precision,$threads,0,$INIT_TIME,$P50,$P90,$P99,$MEAN,$FPS,$MEM" >> $RESULT_FILE
            done
        done
    done
done

echo
echo "=== Benchmark Complete ==="
echo "Results saved to: $RESULT_FILE"
date
