#!/usr/bin/env python3
"""
Generate single-operator ONNX models for performance benchmarking.
This creates a comprehensive set of common neural network operators
with realistic input shapes to test each framework's optimization.

Usage:
    python generate_single_operator_models.py --output_dir models/single_ops
"""

import os
import argparse
import numpy as np

import onnx
from onnx import helper, TensorProto


# 定义要测试的算子和形状配置
OPERATOR_CONFIGS = {
    # ========== 卷积类 ==========
    "Conv2d_1x1_s1": {
        "op_type": "Conv",
        "input_shape": [1, 64, 56, 56],
        "weight_shape": [128, 64, 1, 1],
        "kernel_shape": [1, 1],
        "strides": [1, 1],
        "pads": [0, 0, 0, 0],
        "group": 1,
    },
    "Conv2d_3x3_s1": {
        "op_type": "Conv",
        "input_shape": [1, 64, 56, 56],
        "weight_shape": [128, 64, 3, 3],
        "kernel_shape": [3, 3],
        "strides": [1, 1],
        "pads": [1, 1, 1, 1],
        "group": 1,
    },
    "Conv2d_3x3_s2": {
        "op_type": "Conv",
        "input_shape": [1, 64, 56, 56],
        "weight_shape": [128, 64, 3, 3],
        "kernel_shape": [3, 3],
        "strides": [2, 2],
        "pads": [1, 1, 1, 1],
        "group": 1,
    },
    "Conv2d_dw_3x3": {
        "op_type": "Conv",
        "input_shape": [1, 64, 56, 56],
        "weight_shape": [64, 1, 3, 3],
        "kernel_shape": [3, 3],
        "strides": [1, 1],
        "pads": [1, 1, 1, 1],
        "group": 64,  # depthwise
    },
    "ConvTranspose_2x2": {
        "op_type": "ConvTranspose",
        "input_shape": [1, 64, 28, 28],
        "weight_shape": [64, 128, 2, 2],
        "kernel_shape": [2, 2],
        "strides": [2, 2],
        "pads": [0, 0, 0, 0],
    },

    # ========== 池化类 ==========
    "MaxPool_2x2_s2": {
        "op_type": "MaxPool",
        "input_shape": [1, 64, 56, 56],
        "kernel_shape": [2, 2],
        "strides": [2, 2],
        "pads": [0, 0, 0, 0],
    },
    "AvgPool_2x2_s2": {
        "op_type": "AveragePool",
        "input_shape": [1, 64, 56, 56],
        "kernel_shape": [2, 2],
        "strides": [2, 2],
        "pads": [0, 0, 0, 0],
    },
    "GlobalAvgPool": {
        "op_type": "GlobalAveragePool",
        "input_shape": [1, 128, 7, 7],
    },

    # ========== 激活函数 ==========
    "ReLU": {
        "op_type": "Relu",
        "input_shape": [1, 128, 28, 28],
    },
    "ReLU6": {
        "op_type": "Clip",
        "input_shape": [1, 128, 28, 28],
        "min": 0.0,
        "max": 6.0,
    },
    "Sigmoid": {
        "op_type": "Sigmoid",
        "input_shape": [1, 128, 28, 28],
    },
    "Tanh": {
        "op_type": "Tanh",
        "input_shape": [1, 128, 28, 28],
    },
    "LeakyRelu": {
        "op_type": "LeakyRelu",
        "input_shape": [1, 128, 28, 28],
        "alpha": 0.1,
    },
    "PRelu": {
        "op_type": "PRelu",
        "input_shape": [1, 128, 28, 28],
        "slope_shape": [128],
    },
    "HardSwish": {
        "op_type": "HardSwish",
        "input_shape": [1, 128, 28, 28],
    },
    "GELU": {
        "op_type": "Gelu",
        "input_shape": [1, 128, 28, 28],
    },

    # ========== 归一化 ==========
    "BatchNorm": {
        "op_type": "BatchNormalization",
        "input_shape": [1, 128, 28, 28],
        "channel": 128,
    },
    "InstanceNorm": {
        "op_type": "InstanceNormalization",
        "input_shape": [1, 128, 28, 28],
        "channel": 128,
    },
    "LayerNorm": {
        "op_type": "LayerNormalization",
        "input_shape": [1, 512],
        "normalized_shape": [512],
    },

    # ========== 矩阵运算 ==========
    "MatMul_512x512": {
        "op_type": "MatMul",
        "a_shape": [1, 512],
        "b_shape": [512, 512],
    },
    "MatMul_1024x1024": {
        "op_type": "MatMul",
        "a_shape": [1, 1024],
        "b_shape": [1024, 1024],
    },
    "Gemm": {
        "op_type": "Gemm",
        "a_shape": [1, 512],
        "b_shape": [512, 512],
        "c_shape": [512],
    },

    # ========== 逐元素运算 ==========
    "Add": {
        "op_type": "Add",
        "input_shape": [1, 128, 28, 28],
    },
    "Mul": {
        "op_type": "Mul",
        "input_shape": [1, 128, 28, 28],
    },
    "Div": {
        "op_type": "Div",
        "input_shape": [1, 128, 28, 28],
    },
    "Pow": {
        "op_type": "Pow",
        "input_shape": [1, 128, 28, 28],
        "exponent": 2.0,
    },
    "Sqrt": {
        "op_type": "Sqrt",
        "input_shape": [1, 128, 28, 28],
    },
    "Exp": {
        "op_type": "Exp",
        "input_shape": [1, 128, 28, 28],
    },

    # ========== 形状变换 ==========
    "Resize_bilinear": {
        "op_type": "Resize",
        "input_shape": [1, 64, 28, 28],
        "output_shape": [1, 64, 56, 56],
        "mode": "linear",
    },
    "Reshape": {
        "op_type": "Reshape",
        "input_shape": [1, 128, 7, 7],
        "output_shape": [1, 6272],
    },
    "Transpose_NCHW2NHWC": {
        "op_type": "Transpose",
        "input_shape": [1, 128, 28, 28],
        "perm": [0, 2, 3, 1],
    },
    "Flatten": {
        "op_type": "Flatten",
        "input_shape": [1, 128, 7, 7],
        "axis": 1,
    },
    "Concat": {
        "op_type": "Concat",
        "input_shape": [1, 64, 28, 28],
        "num_inputs": 4,
        "axis": 1,
    },
    "Split": {
        "op_type": "Split",
        "input_shape": [1, 256, 28, 28],
        "split": [64, 64, 64, 64],
        "axis": 1,
    },

    # ========== 降采样 ==========
    "Upsample_Nearest": {
        "op_type": "Upsample",
        "input_shape": [1, 64, 28, 28],
        "scales": [1, 1, 2, 2],
        "mode": "nearest",
    },
    "Upsample_Bilinear": {
        "op_type": "Upsample",
        "input_shape": [1, 64, 28, 28],
        "scales": [1, 1, 2, 2],
        "mode": "linear",
    },

    # ========== Padding ==========
    "Pad_Reflect": {
        "op_type": "Pad",
        "input_shape": [1, 64, 28, 28],
        "pads": [0, 0, 1, 1, 0, 0, 1, 1],
        "mode": "reflect",
    },
    "Pad_Constant": {
        "op_type": "Pad",
        "input_shape": [1, 64, 28, 28],
        "pads": [0, 0, 1, 1, 0, 0, 1, 1],
        "mode": "constant",
    },

    # ========== Reduce ==========
    "ReduceMean": {
        "op_type": "ReduceMean",
        "input_shape": [1, 128, 28, 28],
        "axes": [2, 3],
        "keepdims": 0,
    },
    "ReduceSum": {
        "op_type": "ReduceSum",
        "input_shape": [1, 128, 28, 28],
        "axes": [2, 3],
        "keepdims": 0,
    },

    # ========== Softmax ==========
    "Softmax": {
        "op_type": "Softmax",
        "input_shape": [1, 1000],
        "axis": -1,
    },
    "Softmax_2d": {
        "op_type": "Softmax",
        "input_shape": [1, 128, 28, 28],
        "axis": 1,
    },

    # ========== 常用组合 ==========
    "Conv_BN_ReLU": {
        "op_type": "Fused",
        "fused_ops": ["Conv", "BatchNormalization", "Relu"],
        "input_shape": [1, 64, 56, 56],
        "weight_shape": [128, 64, 3, 3],
        "kernel_shape": [3, 3],
        "strides": [1, 1],
        "pads": [1, 1, 1, 1],
    },
    "DWConv_PWConv": {
        "op_type": "Fused",
        "fused_ops": ["Conv_dw", "Conv_pw"],
        "input_shape": [1, 64, 56, 56],
        "dw_weight_shape": [64, 1, 3, 3],
        "pw_weight_shape": [128, 64, 1, 1],
    },
}


def make_tensor(name, data_type, dims, vals):
    """Create ONNX tensor"""
    return helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=dims,
        vals=vals,
    )


def generate_conv_model(model_name, config):
    """生成卷积模型"""
    input_shape = config["input_shape"]
    weight_shape = config["weight_shape"]

    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)]

    weight_data = np.random.randn(*weight_shape).astype(np.float32)
    bias_data = np.random.randn(weight_shape[0]).astype(np.float32)

    initializers = [
        make_tensor("weight", TensorProto.FLOAT, weight_shape, weight_data.flatten()),
        make_tensor("bias", TensorProto.FLOAT, [weight_shape[0]], bias_data.flatten()),
    ]

    nodes = [
        helper.make_node(
            "Conv",
            inputs=["input", "weight", "bias"],
            outputs=["output"],
            kernel_shape=config["kernel_shape"],
            strides=config["strides"],
            pads=config["pads"],
            group=config.get("group", 1),
        ),
    ]

    graph = helper.make_graph(nodes, model_name, inputs, outputs, initializers)
    model = helper.make_model(graph, producer_name="single_op_benchmark")
    onnx.checker.check_model(model)
    return model


def generate_pool_model(model_name, config):
    """生成池化模型"""
    input_shape = config["input_shape"]

    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)]

    nodes = [
        helper.make_node(
            config["op_type"],
            inputs=["input"],
            outputs=["output"],
            kernel_shape=config["kernel_shape"],
            strides=config["strides"],
            pads=config["pads"],
        ),
    ]

    graph = helper.make_graph(nodes, model_name, inputs, outputs)
    model = helper.make_model(graph, producer_name="single_op_benchmark")
    onnx.checker.check_model(model)
    return model


def generate_global_avg_pool_model(model_name, config):
    """生成全局平均池化模型"""
    input_shape = config["input_shape"]

    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)]

    nodes = [
        helper.make_node(
            "GlobalAveragePool",
            inputs=["input"],
            outputs=["output"],
        ),
    ]

    graph = helper.make_graph(nodes, model_name, inputs, outputs)
    model = helper.make_model(graph, producer_name="single_op_benchmark")
    onnx.checker.check_model(model)
    return model


def generate_activation_model(model_name, config):
    """生成激活函数模型"""
    input_shape = config["input_shape"]
    op_type = config["op_type"]

    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)]

    initializers = []
    node_kwargs = {}

    if op_type == "Clip":
        node_kwargs["min"] = config["min"]
        node_kwargs["max"] = config["max"]
    elif op_type == "LeakyRelu":
        node_kwargs["alpha"] = config["alpha"]
    elif op_type == "PRelu":
        slope_shape = config["slope_shape"]
        slope_data = np.full(slope_shape, 0.1, dtype=np.float32)
        initializers.append(make_tensor("slope", TensorProto.FLOAT, slope_shape, slope_data))
        node_inputs = ["input", "slope"]
        node = helper.make_node(op_type, inputs=node_inputs, outputs=["output"], **node_kwargs)
        graph = helper.make_graph([node], model_name, inputs, outputs, initializers)
        model = helper.make_model(graph, producer_name="single_op_benchmark")
        onnx.checker.check_model(model)
        return model

    nodes = [
        helper.make_node(op_type, inputs=["input"], outputs=["output"], **node_kwargs),
    ]

    graph = helper.make_graph(nodes, model_name, inputs, outputs, initializers)
    model = helper.make_model(graph, producer_name="single_op_benchmark")
    onnx.checker.check_model(model)
    return model


def generate_batchnorm_model(model_name, config):
    """生成 BatchNorm 模型"""
    input_shape = config["input_shape"]
    channel = config["channel"]

    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)]

    scale_data = np.ones(channel, dtype=np.float32)
    bias_data = np.zeros(channel, dtype=np.float32)
    mean_data = np.zeros(channel, dtype=np.float32)
    var_data = np.ones(channel, dtype=np.float32)

    initializers = [
        make_tensor("scale", TensorProto.FLOAT, [channel], scale_data),
        make_tensor("bias", TensorProto.FLOAT, [channel], bias_data),
        make_tensor("mean", TensorProto.FLOAT, [channel], mean_data),
        make_tensor("var", TensorProto.FLOAT, [channel], var_data),
    ]

    nodes = [
        helper.make_node(
            "BatchNormalization",
            inputs=["input", "scale", "bias", "mean", "var"],
            outputs=["output"],
            epsilon=1e-05,
            momentum=0.9,
        ),
    ]

    graph = helper.make_graph(nodes, model_name, inputs, outputs, initializers)
    model = helper.make_model(graph, producer_name="single_op_benchmark")
    onnx.checker.check_model(model)
    return model


def generate_instancenorm_model(model_name, config):
    """生成 InstanceNorm 模型"""
    input_shape = config["input_shape"]
    channel = config["channel"]

    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)]

    scale_data = np.ones(channel, dtype=np.float32)
    bias_data = np.zeros(channel, dtype=np.float32)

    initializers = [
        make_tensor("scale", TensorProto.FLOAT, [channel], scale_data),
        make_tensor("bias", TensorProto.FLOAT, [channel], bias_data),
    ]

    nodes = [
        helper.make_node(
            "InstanceNormalization",
            inputs=["input", "scale", "bias"],
            outputs=["output"],
            epsilon=1e-05,
        ),
    ]

    graph = helper.make_graph(nodes, model_name, inputs, outputs, initializers)
    model = helper.make_model(graph, producer_name="single_op_benchmark")
    onnx.checker.check_model(model)
    return model


def generate_layernorm_model(model_name, config):
    """生成 LayerNorm 模型"""
    input_shape = config["input_shape"]
    normalized_shape = config["normalized_shape"]

    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)]

    scale_data = np.ones(normalized_shape, dtype=np.float32)
    bias_data = np.zeros(normalized_shape, dtype=np.float32)

    initializers = [
        make_tensor("scale", TensorProto.FLOAT, normalized_shape, scale_data),
        make_tensor("bias", TensorProto.FLOAT, normalized_shape, bias_data),
    ]

    nodes = [
        helper.make_node(
            "LayerNormalization",
            inputs=["input", "scale", "bias"],
            outputs=["output"],
            axis=-1,
            epsilon=1e-05,
        ),
    ]

    graph = helper.make_graph(nodes, model_name, inputs, outputs, initializers)
    model = helper.make_model(graph, producer_name="single_op_benchmark")
    onnx.checker.check_model(model)
    return model


def generate_matmul_model(model_name, config):
    """生成 MatMul 模型"""
    a_shape = config["a_shape"]
    b_shape = config["b_shape"]

    inputs = [helper.make_tensor_value_info("a", TensorProto.FLOAT, a_shape)]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)]

    b_data = np.random.randn(*b_shape).astype(np.float32)
    initializers = [
        make_tensor("b", TensorProto.FLOAT, b_shape, b_data.flatten()),
    ]

    nodes = [
        helper.make_node("MatMul", inputs=["a", "b"], outputs=["output"]),
    ]

    graph = helper.make_graph(nodes, model_name, inputs, outputs, initializers)
    model = helper.make_model(graph, producer_name="single_op_benchmark")
    onnx.checker.check_model(model)
    return model


def generate_gemm_model(model_name, config):
    """生成 Gemm 模型"""
    a_shape = config["a_shape"]
    b_shape = config["b_shape"]
    c_shape = config["c_shape"]

    inputs = [helper.make_tensor_value_info("a", TensorProto.FLOAT, a_shape)]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)]

    b_data = np.random.randn(*b_shape).astype(np.float32)
    c_data = np.random.randn(*c_shape).astype(np.float32)

    initializers = [
        make_tensor("b", TensorProto.FLOAT, b_shape, b_data.flatten()),
        make_tensor("c", TensorProto.FLOAT, c_shape, c_data.flatten()),
    ]

    nodes = [
        helper.make_node(
            "Gemm",
            inputs=["a", "b", "c"],
            outputs=["output"],
            alpha=1.0,
            beta=1.0,
            transA=0,
            transB=0,
        ),
    ]

    graph = helper.make_graph(nodes, model_name, inputs, outputs, initializers)
    model = helper.make_model(graph, producer_name="single_op_benchmark")
    onnx.checker.check_model(model)
    return model


def generate_binary_model(model_name, config):
    """生成二元运算模型 (Add, Mul, etc.)"""
    input_shape = config["input_shape"]
    op_type = config["op_type"]

    inputs = [helper.make_tensor_value_info("input1", TensorProto.FLOAT, input_shape)]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)]

    input2_data = np.random.randn(*input_shape).astype(np.float32)
    initializers = [
        make_tensor("input2", TensorProto.FLOAT, input_shape, input2_data.flatten()),
    ]

    nodes = [
        helper.make_node(op_type, inputs=["input1", "input2"], outputs=["output"]),
    ]

    graph = helper.make_graph(nodes, model_name, inputs, outputs, initializers)
    model = helper.make_model(graph, producer_name="single_op_benchmark")
    onnx.checker.check_model(model)
    return model


def generate_unary_model(model_name, config):
    """生成一元运算模型"""
    input_shape = config["input_shape"]
    op_type = config["op_type"]

    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)]

    initializers = []
    node_kwargs = {}

    if op_type == "Pow":
        exponent = config.get("exponent", 2.0)
        node_kwargs = {"exponent": exponent}

    nodes = [
        helper.make_node(op_type, inputs=["input"], outputs=["output"], **node_kwargs),
    ]

    graph = helper.make_graph(nodes, model_name, inputs, outputs, initializers)
    model = helper.make_model(graph, producer_name="single_op_benchmark")
    onnx.checker.check_model(model)
    return model


def generate_resize_model(model_name, config):
    """生成 Resize 模型"""
    input_shape = config["input_shape"]

    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)]

    scales_data = np.array(config["scales"], dtype=np.float32)
    initializers = [
        make_tensor("scales", TensorProto.FLOAT, [4], scales_data),
    ]

    nodes = [
        helper.make_node(
            "Resize",
            inputs=["input", "", "scales"],
            outputs=["output"],
            mode=config["mode"],
        ),
    ]

    graph = helper.make_graph(nodes, model_name, inputs, outputs, initializers)
    model = helper.make_model(graph, producer_name="single_op_benchmark")
    onnx.checker.check_model(model)
    return model


def generate_upsample_model(model_name, config):
    """生成 Upsample 模型"""
    input_shape = config["input_shape"]

    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)]

    scales_data = np.array(config["scales"], dtype=np.float32)
    initializers = [
        make_tensor("scales", TensorProto.FLOAT, [4], scales_data),
    ]

    nodes = [
        helper.make_node(
            "Upsample",
            inputs=["input", "scales"],
            outputs=["output"],
            mode=config["mode"],
        ),
    ]

    graph = helper.make_graph(nodes, model_name, inputs, outputs, initializers)
    model = helper.make_model(graph, producer_name="single_op_benchmark")
    onnx.checker.check_model(model)
    return model


def generate_reshape_model(model_name, config):
    """生成 Reshape 模型"""
    input_shape = config["input_shape"]
    output_shape = config["output_shape"]

    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)]

    shape_data = np.array(output_shape, dtype=np.int64)
    initializers = [
        make_tensor("shape", TensorProto.INT64, [len(output_shape)], shape_data),
    ]

    nodes = [
        helper.make_node("Reshape", inputs=["input", "shape"], outputs=["output"]),
    ]

    graph = helper.make_graph(nodes, model_name, inputs, outputs, initializers)
    model = helper.make_model(graph, producer_name="single_op_benchmark")
    onnx.checker.check_model(model)
    return model


def generate_transpose_model(model_name, config):
    """生成 Transpose 模型"""
    input_shape = config["input_shape"]

    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)]

    nodes = [
        helper.make_node(
            "Transpose",
            inputs=["input"],
            outputs=["output"],
            perm=config["perm"],
        ),
    ]

    graph = helper.make_graph(nodes, model_name, inputs, outputs)
    model = helper.make_model(graph, producer_name="single_op_benchmark")
    onnx.checker.check_model(model)
    return model


def generate_flatten_model(model_name, config):
    """生成 Flatten 模型"""
    input_shape = config["input_shape"]

    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)]

    nodes = [
        helper.make_node(
            "Flatten",
            inputs=["input"],
            outputs=["output"],
            axis=config["axis"],
        ),
    ]

    graph = helper.make_graph(nodes, model_name, inputs, outputs)
    model = helper.make_model(graph, producer_name="single_op_benchmark")
    onnx.checker.check_model(model)
    return model


def generate_concat_model(model_name, config):
    """生成 Concat 模型"""
    input_shape = config["input_shape"]
    num_inputs = config["num_inputs"]

    inputs = [
        helper.make_tensor_value_info(f"input{i}", TensorProto.FLOAT, input_shape)
        for i in range(num_inputs)
    ]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)]

    nodes = [
        helper.make_node(
            "Concat",
            inputs=[f"input{i}" for i in range(num_inputs)],
            outputs=["output"],
            axis=config["axis"],
        ),
    ]

    graph = helper.make_graph(nodes, model_name, inputs, outputs)
    model = helper.make_model(graph, producer_name="single_op_benchmark")
    onnx.checker.check_model(model)
    return model


def generate_split_model(model_name, config):
    """生成 Split 模型"""
    input_shape = config["input_shape"]

    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)]
    num_outputs = len(config["split"])
    outputs = [
        helper.make_tensor_value_info(f"output{i}", TensorProto.FLOAT, None)
        for i in range(num_outputs)
    ]

    split_data = np.array(config["split"], dtype=np.int64)
    initializers = [
        make_tensor("split", TensorProto.INT64, [num_outputs], split_data),
    ]

    nodes = [
        helper.make_node(
            "Split",
            inputs=["input", "split"],
            outputs=[f"output{i}" for i in range(num_outputs)],
            axis=config["axis"],
        ),
    ]

    graph = helper.make_graph(nodes, model_name, inputs, outputs, initializers)
    model = helper.make_model(graph, producer_name="single_op_benchmark")
    onnx.checker.check_model(model)
    return model


def generate_pad_model(model_name, config):
    """生成 Pad 模型"""
    input_shape = config["input_shape"]

    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)]

    pads_data = np.array(config["pads"], dtype=np.int64)
    initializers = [
        make_tensor("pads", TensorProto.INT64, [8], pads_data),
    ]

    nodes = [
        helper.make_node(
            "Pad",
            inputs=["input", "pads"],
            outputs=["output"],
            mode=config["mode"],
        ),
    ]

    graph = helper.make_graph(nodes, model_name, inputs, outputs, initializers)
    model = helper.make_model(graph, producer_name="single_op_benchmark")
    onnx.checker.check_model(model)
    return model


def generate_reduce_model(model_name, config):
    """生成 Reduce 模型"""
    input_shape = config["input_shape"]
    op_type = config["op_type"]

    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)]

    axes_data = np.array(config["axes"], dtype=np.int64)
    initializers = [
        make_tensor("axes", TensorProto.INT64, [len(config["axes"])], axes_data),
    ]

    nodes = [
        helper.make_node(
            op_type,
            inputs=["input", "axes"],
            outputs=["output"],
            keepdims=config.get("keepdims", 0),
        ),
    ]

    graph = helper.make_graph(nodes, model_name, inputs, outputs, initializers)
    model = helper.make_model(graph, producer_name="single_op_benchmark")
    onnx.checker.check_model(model)
    return model


def generate_softmax_model(model_name, config):
    """生成 Softmax 模型"""
    input_shape = config["input_shape"]

    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)]

    nodes = [
        helper.make_node(
            "Softmax",
            inputs=["input"],
            outputs=["output"],
            axis=config["axis"],
        ),
    ]

    graph = helper.make_graph(nodes, model_name, inputs, outputs)
    model = helper.make_model(graph, producer_name="single_op_benchmark")
    onnx.checker.check_model(model)
    return model


def generate_model(model_name, config):
    """根据配置生成模型"""
    op_type = config["op_type"]

    # 卷积类
    if op_type == "Conv":
        return generate_conv_model(model_name, config)
    elif op_type in ["MaxPool", "AveragePool"]:
        return generate_pool_model(model_name, config)
    elif op_type == "GlobalAveragePool":
        return generate_global_avg_pool_model(model_name, config)

    # 激活函数
    elif op_type in ["Relu", "Clip", "Sigmoid", "Tanh", "LeakyRelu", "PRelu",
                     "HardSwish", "Gelu"]:
        return generate_activation_model(model_name, config)

    # 归一化
    elif op_type == "BatchNormalization":
        return generate_batchnorm_model(model_name, config)
    elif op_type == "InstanceNormalization":
        return generate_instancenorm_model(model_name, config)
    elif op_type == "LayerNormalization":
        return generate_layernorm_model(model_name, config)

    # 矩阵运算
    elif op_type == "MatMul":
        return generate_matmul_model(model_name, config)
    elif op_type == "Gemm":
        return generate_gemm_model(model_name, config)

    # 二元运算
    elif op_type in ["Add", "Mul", "Div"]:
        return generate_binary_model(model_name, config)

    # 一元运算
    elif op_type in ["Sqrt", "Exp", "Pow"]:
        return generate_unary_model(model_name, config)

    # 形状变换
    elif op_type == "Resize":
        return generate_resize_model(model_name, config)
    elif op_type == "Upsample":
        return generate_upsample_model(model_name, config)
    elif op_type == "Reshape":
        return generate_reshape_model(model_name, config)
    elif op_type == "Transpose":
        return generate_transpose_model(model_name, config)
    elif op_type == "Flatten":
        return generate_flatten_model(model_name, config)
    elif op_type == "Concat":
        return generate_concat_model(model_name, config)
    elif op_type == "Split":
        return generate_split_model(model_name, config)
    elif op_type == "Pad":
        return generate_pad_model(model_name, config)

    # Reduce
    elif op_type in ["ReduceMean", "ReduceSum"]:
        return generate_reduce_model(model_name, config)

    # Softmax
    elif op_type == "Softmax":
        return generate_softmax_model(model_name, config)

    # 组合算子
    elif op_type == "Fused":
        # 这里可以实现组合算子
        raise NotImplementedError(f"Fused ops not implemented yet: {config['fused_ops']}")

    else:
        raise NotImplementedError(f"Unsupported operator type: {op_type}")


def main():
    parser = argparse.ArgumentParser(description="Generate single-operator ONNX models")
    parser.add_argument("--output_dir", default="models/single_ops",
                        help="Output directory for generated models")
    parser.add_argument("--ops", nargs="+", help="Specific operators to generate (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ops_to_generate = args.ops if args.ops else OPERATOR_CONFIGS.keys()

    print(f"📦 Generating single-operator ONNX models...")
    print(f"   Output directory: {args.output_dir}")
    print()

    success_count = 0
    fail_count = 0

    for model_name in sorted(ops_to_generate):
        if model_name not in OPERATOR_CONFIGS:
            print(f"⚠️  Unknown operator: {model_name}")
            continue

        config = OPERATOR_CONFIGS[model_name]

        try:
            model = generate_model(model_name, config)
            output_path = os.path.join(args.output_dir, f"{model_name}.onnx")
            onnx.save(model, output_path)

            # 验证
            onnx.checker.check_model(output_path)

            # 计算参数量
            param_count = 0
            for init in model.graph.initializer:
                param_count += np.prod(init.dims) if init.dims else 0

            print(f"✅ {model_name:30s} -> {param_count:8d} parameters")
            success_count += 1

        except Exception as e:
            print(f"❌ {model_name:30s} -> Failed: {e}")
            fail_count += 1

    print()
    print(f"📊 Summary: {success_count} success, {fail_count} failed")
    print(f"📁 Models saved to: {args.output_dir}")
    print()
    print("Next steps:")
    print("  1. Convert to all framework formats: ./scripts/convert_models.sh")
    print("  2. Run benchmark on device: adb push and run inference")


if __name__ == "__main__":
    main()
