#!/usr/bin/env python3
"""导出 MobileViT-S 到 ONNX 格式。

Apple MobileViT-S:
  - 输入: [1, 3, 256, 256] (NCHW)
  - 参数: 5.6M
  - 输出: [1, 1000] (ImageNet-1k 分类 logits)
  - 来源: timm (PyTorch Image Models) / apple/ml-cvnets
"""

import torch
import timm

MODEL_NAME = "mobilevit_s.cvnets_in1k"
OUTPUT_DIR = "models/classification/mobilevit_s"
ONNX_PATH = f"{OUTPUT_DIR}/mobilevit_s.onnx"


def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[1/3] Loading {MODEL_NAME} from timm...")
    model = timm.create_model(MODEL_NAME, pretrained=True)
    model.eval()

    # MobileViT 使用 256x256 输入
    dummy_input = torch.randn(1, 3, 256, 256)
    input_names = ["input"]
    output_names = ["output"]

    print(f"[2/3] Tracing and exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,  # 固定输入尺寸，匹配 benchmark 需求
        opset_version=17,
        do_constant_folding=True,
    )

    print(f"[3/3] Verifying exported model...")
    import onnx
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)

    # 打印模型信息
    print(f"\n✅ MobileViT-S exported successfully!")
    print(f"   Output: {ONNX_PATH}")
    print(f"   Input:  {onnx_model.graph.input[0].name} = {[d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]}")
    print(f"   Output: {onnx_model.graph.output[0].name} = {[d.dim_value for d in onnx_model.graph.output[0].type.tensor_type.shape.dim]}")
    print(f"   Opset:  {onnx_model.opset_import[0].version}")

    # 简单推理验证
    import onnxruntime as ort
    session = ort.InferenceSession(ONNX_PATH)
    ort_input = {input_names[0]: dummy_input.numpy()}
    ort_output = session.run(None, ort_input)
    print(f"   Inference test: output shape = {ort_output[0].shape}")
    print(f"   Top-5 classes: {ort_output[0][0].argsort()[-5:][::-1]}")


if __name__ == "__main__":
    main()
