// minimal_bert.cpp — 最小复现: 直接调 MNN C API, 对齐 benchmark.out doBench()
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/AutoTime.hpp>
#include <fstream>
#include <vector>
#include <cstdio>
#include <sys/time.h>

// Revert header — from MNN tools/cpp/
#include "revertMNNModel.hpp"

int main(int argc, char** argv) {
    const char* model_path = (argc > 1) ? argv[1] : "/data/local/tmp/benchmark/models/exported/mnn/bert.mnn";
    int loop = (argc > 2) ? atoi(argv[2]) : 10;
    int warmup = (argc > 3) ? atoi(argv[3]) : 10;

    // ── Revert preprocessing (exact copy of benchmark.out lines 121-131) ──
    auto revertor = std::unique_ptr<Revert>(new Revert(model_path));
    revertor->initialize(0.0f, 1);  // sparsity=0, sparseBlockOC=1 (defaults)
    auto modelBuffer = revertor->getBuffer();
    const auto bufferSize = revertor->getBufferSize();

    auto net = std::shared_ptr<MNN::Interpreter>(
        MNN::Interpreter::createFromBuffer(modelBuffer, bufferSize),
        MNN::Interpreter::destroy);
    revertor.reset();  // free Revert after net created

    // 2. Exact config matching benchmark.out
    net->setSessionMode(MNN::Interpreter::Session_Release);
    net->setSessionHint(MNN::Interpreter::HintMode::CPU_ENABLE_KLEIDIAI, 0);

    MNN::ScheduleConfig config;
    config.numThread = 4;
    config.type = MNN_FORWARD_CPU;

    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_Low;
    backendConfig.power = MNN::BackendConfig::Power_High;
    config.backendConfig = &backendConfig;

    // 3. Create session, cache tensors
    MNN::Session* session = net->createSession(config);
    MNN::Tensor* input = net->getSessionInput(session, NULL);
    MNN::Tensor* output = net->getSessionOutput(session, NULL);

    // 4. Create host tensors (exact match benchmark.out)
    std::shared_ptr<MNN::Tensor> givenTensor(
        MNN::Tensor::createHostTensorFromDevice(input, false));
    std::shared_ptr<MNN::Tensor> expectTensor(
        MNN::Tensor::createHostTensorFromDevice(output, false));

    const MNN::Backend* inBackend = net->getBackend(session, input);
    (void)inBackend;

    net->releaseModel();

    printf("Config: type=CPU threads=4 precision=Precision_Low(%d) power=Power_High(%d)\n",
           backendConfig.precision, backendConfig.power);
    printf("Model: %s\n", model_path);
    printf("Warmup=%d Runs=%d\n\n", warmup, loop);

    // 5. Warmup
    for (int i = 0; i < warmup; i++) {
        void* host = input->map(MNN::Tensor::MAP_TENSOR_WRITE, input->getDimensionType());
        input->unmap(MNN::Tensor::MAP_TENSOR_WRITE, input->getDimensionType(), host);
        net->runSession(session);
        host = output->map(MNN::Tensor::MAP_TENSOR_READ, output->getDimensionType());
        output->unmap(MNN::Tensor::MAP_TENSOR_READ, output->getDimensionType(), host);
    }

    // 6. Benchmark — exact copy of benchmark.out doBench() loop
    std::vector<float> costs;
    for (int i = 0; i < loop; i++) {
        MNN::Timer _t;  // Uses gettimeofday (same as benchmark.out)
        void* host = input->map(MNN::Tensor::MAP_TENSOR_WRITE, input->getDimensionType());
        input->unmap(MNN::Tensor::MAP_TENSOR_WRITE, input->getDimensionType(), host);
        net->runSession(session);
        host = output->map(MNN::Tensor::MAP_TENSOR_READ, output->getDimensionType());
        output->unmap(MNN::Tensor::MAP_TENSOR_READ, output->getDimensionType(), host);
        costs.push_back((float)_t.durationInUs() / 1000.0f);
    }

    // 7. Stats (same as displayStats)
    float min_val = costs[0], max_val = costs[0], sum = 0;
    for (auto v : costs) {
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
        sum += v;
    }
    float avg = sum / costs.size();

    // P50
    std::sort(costs.begin(), costs.end());
    float p50 = costs[costs.size() / 2];

    printf("Results (MNN::Timer, gettimeofday):\n");
    printf("  min=%.3f ms  max=%.3f ms  avg=%.3f ms  p50=%.3f ms\n", min_val, max_val, avg, p50);

    return 0;
}
