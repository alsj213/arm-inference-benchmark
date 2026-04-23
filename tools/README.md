# Model Conversion Tools

This directory contains tools for converting ONNX models to various inference framework formats.

## Quick Start

```bash
# 1. Build host conversion tools
./scripts/build_host_tools.sh

# 2. Install Python dependencies for TFLite conversion
pip install onnx onnx-tf tensorflow

# 3. Download pretrained models (ONNX format)
python ./scripts/download_pretrained.py

# 4. Convert models to all framework formats
./scripts/convert_models.sh
```

## Supported Frameworks (Default Enabled)

| Framework | Tool | Output Format |
|-----------|------|---------------|
| ncnn | onnx2ncnn | `.model` + `.bin` |
| MNN | onnx2mnn | `.mnn` |
| TNN | onnx2tnn | `.tnnproto` + `.tnnmodel` |
| TFLite | convert_tflite.py | `.tflite` |
| ONNX Runtime | (use original) | `.onnx` |

## Optional Frameworks (Disabled by Default)

| Framework | Tool | Notes |
|-----------|------|-------|
| QNN | qnn-onnx-converter | Requires Qualcomm QNN SDK |
| TVM | TVM Relay compiler | Build TVM separately |

## Tools Directory Structure

```
tools/
├── bin/                   # Symlinks to built conversion tools
│   ├── onnx2ncnn         # ncnn converter
│   ├── onnx2mnn          # MNN converter
│   └── onnx2tnn          # TNN converter
└── README.md
```

## Build Host Tools

Run `./scripts/build_host_tools.sh` to build:
- onnx2ncnn (from ncnn submodule)
- onnx2mnn (from MNN submodule)
- onnx2tnn (from TNN submodule)

These are built for x86_64 host architecture, not Android.

## TFLite Conversion

TFLite conversion requires Python packages:
```bash
pip install onnx onnx-tf tensorflow
```

The conversion script is at `./scripts/convert_tflite.py`.

## Manual Conversion Example

```bash
# Single model conversion for ncnn
./tools/bin/onnx2ncnn mobilenetv2.onnx mobilenetv2_ncnn.model mobilenetv2_ncnn.bin

# Single model conversion for MNN
./tools/bin/onnx2mnn mobilenetv2.onnx mobilenetv2_MNN.mnn

# Single model conversion for TNN
./tools/bin/onnx2tnn -onnx mobilenetv2.onnx -version v1.0

# Single model conversion for TFLite
python ./scripts/convert_tflite.py mobilenetv2.onnx mobilenetv2.tflite
```
