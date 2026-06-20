#!/usr/bin/env python3
"""
Convert ONNX models to TFLite format using tensorflow and onnx-tf.
"""
import sys
import os
import argparse

try:
    import onnx
    import tensorflow as tf
    from onnx_tf.backend import prepare
except ImportError as e:
    print(f"Error: Missing dependencies. {e}")
    print("Install: pip install onnx onnx-tf tensorflow")
    sys.exit(1)


def convert_onnx_to_tflite(onnx_path, tflite_path):
    """Convert ONNX model to TFLite format."""
    print(f"Converting: {onnx_path} -> {tflite_path}")

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Convert ONNX to TensorFlow
    tf_rep = prepare(onnx_model)

    # Export to SavedModel
    saved_model_dir = "/tmp/tf_savedmodel_" + os.path.basename(onnx_path)
    tf_rep.export_graph(saved_model_dir)

    # Convert SavedModel to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = []  # No optimizations by default (FP32)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]

    tflite_model = converter.convert()

    # Save TFLite model
    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Success: {tflite_path}")
    print(f"Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Convert ONNX to TFLite')
    parser.add_argument('onnx_path', help='Input ONNX model path')
    parser.add_argument('tflite_path', help='Output TFLite model path')
    args = parser.parse_args()

    convert_onnx_to_tflite(args.onnx_path, args.tflite_path)


if __name__ == '__main__':
    main()
