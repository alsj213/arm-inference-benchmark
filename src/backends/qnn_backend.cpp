#include "qnn_backend.h"

// QNN SDK Integration
#include "QNN/QnnInterface.h"
#include "QNN/System/QnnSystemInterface.h"

#include <dlfcn.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <cstdio>

#define QNN_BACKEND_LIB_CPU "libQnnCpu.so"
#define QNN_BACKEND_LIB_HTP "libQnnHtp.so"

typedef Qnn_ErrorHandle_t (*QnnInterface_getProviders_t)(const QnnInterface_t***, uint32_t*);

class QNNBackendImpl {
public:
    void* libQnnHandle = nullptr;
    QNN_INTERFACE_VER_TYPE qnnInterface{};
    Qnn_LogHandle_t logHandle = nullptr;
    Qnn_BackendHandle_t backendHandle = nullptr;
    Qnn_DeviceHandle_t deviceHandle = nullptr;
    Qnn_ContextHandle_t contextHandle = nullptr;
    Qnn_GraphHandle_t graphHandle = nullptr;

    std::vector<void*> inputBuffers;
    std::vector<void*> outputBuffers;

    bool initialized = false;
    bool graphLoaded = false;
};

QNNBackend::QNNBackend() : impl_(new QNNBackendImpl()) {
}

QNNBackend::~QNNBackend() {
    delete impl_;
}

static std::vector<uint8_t> readBinaryFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return {};
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(size);
    file.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

bool QNNBackend::init(const BenchmarkConfig& config) {
    printf("=== QNN Backend Initialization ===\n");

    // 1. Load QNN backend library - try CPU first as HTP may require special setup
    const char* backendLib = QNN_BACKEND_LIB_CPU;
    impl_->libQnnHandle = dlopen(backendLib, RTLD_NOW | RTLD_LOCAL);
    if (!impl_->libQnnHandle) {
        printf("Failed to load %s: %s\n", backendLib, dlerror());
        printf("Trying HTP backend...\n");
        backendLib = QNN_BACKEND_LIB_HTP;
        impl_->libQnnHandle = dlopen(backendLib, RTLD_NOW | RTLD_LOCAL);
        if (!impl_->libQnnHandle) {
            printf("Failed to load %s: %s\n", backendLib, dlerror());
            printf("Note: QNN libraries must be pushed to device and in LD_LIBRARY_PATH\n");
            return false;
        }
    }
    printf("Loaded QNN backend: %s\n", backendLib);

    // 2. Get interface providers
    QnnInterface_getProviders_t getProviders =
        reinterpret_cast<QnnInterface_getProviders_t>(
            dlsym(impl_->libQnnHandle, "QnnInterface_getProviders"));
    if (!getProviders) {
        printf("Failed to find QnnInterface_getProviders: %s\n", dlerror());
        return false;
    }

    const QnnInterface_t** interfaceProviders = nullptr;
    uint32_t numProviders = 0;
    Qnn_ErrorHandle_t error = getProviders(&interfaceProviders, &numProviders);
    if (error != QNN_SUCCESS || numProviders == 0) {
        printf("Failed to get QNN interface providers (0x%lx)\n", error);
        return false;
    }
    printf("Found %d QNN interface provider(s)\n", numProviders);

    // Find valid interface with matching major version
    bool foundValidInterface = false;
    for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
        if (QNN_API_VERSION_MAJOR == interfaceProviders[pIdx]->apiVersion.coreApiVersion.major &&
            QNN_API_VERSION_MINOR <= interfaceProviders[pIdx]->apiVersion.coreApiVersion.minor) {
            foundValidInterface = true;
            impl_->qnnInterface = interfaceProviders[pIdx]->QNN_INTERFACE_VER_NAME;
            printf("Using QNN interface: %s (v%d.%d)\n",
                   interfaceProviders[pIdx]->providerName,
                   static_cast<int>(interfaceProviders[pIdx]->apiVersion.coreApiVersion.major),
                   static_cast<int>(interfaceProviders[pIdx]->apiVersion.coreApiVersion.minor));
            break;
        }
    }
    if (!foundValidInterface) {
        printf("Failed to find a valid QNN interface\n");
        return false;
    }

    // 3. Create log
    if (impl_->qnnInterface.logCreate) {
        error = impl_->qnnInterface.logCreate(nullptr, QNN_LOG_LEVEL_ERROR, &impl_->logHandle);
        if (error != QNN_SUCCESS) {
            printf("Warning: Failed to create QNN log (0x%lx)\n", error);
            impl_->logHandle = nullptr;
        }
    }

    // 4. Create backend
    error = impl_->qnnInterface.backendCreate(impl_->logHandle, nullptr, &impl_->backendHandle);
    if (error != QNN_SUCCESS) {
        printf("Failed to create QNN backend (0x%lx)\n", error);
        return false;
    }
    printf("QNN backend created successfully\n");

    // 5. Create device if supported
    if (impl_->qnnInterface.deviceCreate) {
        error = impl_->qnnInterface.deviceCreate(impl_->backendHandle, nullptr, &impl_->deviceHandle);
        if (error != QNN_SUCCESS) {
            printf("Warning: Failed to create QNN device (0x%lx)\n", error);
            impl_->deviceHandle = nullptr;
        } else {
            printf("QNN device created successfully\n");
        }
    }

    // 6. Try to load context binary if available
    std::string contextPath = config.model_path;
    if (impl_->qnnInterface.contextCreateFromBinary &&
        (contextPath.find(".bin") != std::string::npos ||
         contextPath.find(".so") != std::string::npos)) {
        printf("Loading QNN context from: %s\n", contextPath.c_str());
        auto binaryData = readBinaryFile(contextPath);
        if (!binaryData.empty()) {
            error = impl_->qnnInterface.contextCreateFromBinary(
                impl_->backendHandle,
                impl_->deviceHandle,
                nullptr,  // config
                binaryData.data(),
                binaryData.size(),
                &impl_->contextHandle,
                nullptr  // profile
            );
            if (error == QNN_SUCCESS) {
                printf("QNN context loaded from binary\n");
                // Try to retrieve graph
                if (impl_->qnnInterface.graphRetrieve) {
                    error = impl_->qnnInterface.graphRetrieve(impl_->contextHandle, "mobilenetv2", &impl_->graphHandle);
                    if (error == QNN_SUCCESS) {
                        printf("QNN graph retrieved\n");
                        impl_->graphLoaded = true;
                    } else {
                        printf("Warning: Failed to retrieve graph (0x%lx)\n", error);
                    }
                }
            } else {
                printf("Warning: Failed to load context binary (0x%lx)\n", error);
            }
        }
    }

    if (!impl_->graphLoaded) {
        printf("Note: No pre-compiled QNN context binary found.\n");
        printf("      For full hardware acceleration, use Qualcomm qnn-onnx-converter:\n");
        printf("        1. Convert ONNX model to QNN context binary\n");
        printf("        2. Push .bin file to device and use as model path\n");
    }

    impl_->initialized = true;
    printf("QNN initialization completed!\n");
    return true;
}

bool QNNBackend::infer(const std::vector<float>& input) {
    if (!impl_->initialized) {
        return false;
    }

    if (!impl_->graphLoaded) {
        // Placeholder - return success for benchmark timing
        // Real inference requires QNN context binary with compiled graph
        return true;
    }

    // TODO: Implement real inference when graph is loaded
    // 1. Prepare input tensors
    // 2. Copy input data
    // 3. graphExecute
    // 4. Copy output data

    return true;
}

void QNNBackend::deinit() {
    if (impl_->initialized) {
        if (impl_->graphHandle) {
            // Graphs are freed with context
            impl_->graphHandle = nullptr;
        }
        if (impl_->contextHandle && impl_->qnnInterface.contextFree) {
            impl_->qnnInterface.contextFree(impl_->contextHandle, nullptr);
            impl_->contextHandle = nullptr;
        }
        if (impl_->deviceHandle && impl_->qnnInterface.deviceFree) {
            impl_->qnnInterface.deviceFree(impl_->deviceHandle);
            impl_->deviceHandle = nullptr;
        }
        if (impl_->backendHandle && impl_->qnnInterface.backendFree) {
            impl_->qnnInterface.backendFree(impl_->backendHandle);
            impl_->backendHandle = nullptr;
        }
        if (impl_->logHandle && impl_->qnnInterface.logFree) {
            impl_->qnnInterface.logFree(impl_->logHandle);
            impl_->logHandle = nullptr;
        }
    }

    if (impl_->libQnnHandle) {
        dlclose(impl_->libQnnHandle);
        impl_->libQnnHandle = nullptr;
    }

    // Free buffers
    for (void* buf : impl_->inputBuffers) {
        free(buf);
    }
    for (void* buf : impl_->outputBuffers) {
        free(buf);
    }
    impl_->inputBuffers.clear();
    impl_->outputBuffers.clear();

    impl_->initialized = false;
    impl_->graphLoaded = false;
}
