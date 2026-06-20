#!/usr/bin/env python3
"""
从整模型 ONNX 中提取去重后的单算子

流程:
  1. 加载 6 个整模型 ONNX
  2. 遍历每个模型的每一层, 提取 (op_type, input_shape, attributes)
  3. 跨模型去重: 同类型+同输入形状+同关键属性 → 只保留一个
  4. 用命名规则 {OpType}_{Model}_{ShapeKey} 生成 ONNX 文件

用法:
  python3 scripts/extract_model_operators.py --output_dir models/single_ops_extracted
  python3 scripts/extract_model_operators.py --dry-run  # 只看不去重
"""

import os, sys, argparse, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import onnx
from onnx import helper, TensorProto

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "MBV2": PROJECT_ROOT / "models/classification/mobilenetv2/mobilenetv2.onnx",
    "R50":  PROJECT_ROOT / "models/classification/resnet50/resnet50.onnx",
    "YOLO": PROJECT_ROOT / "models/detection/yolov8n/yolov8n.onnx",
    "BERT": PROJECT_ROOT / "models/nlp/bert/bert.onnx",
    "QWEN": None,   # Qwen GGUF, 无 ONNX — 用手工配置
    "MVIT": PROJECT_ROOT / "models/classification/mobilevit_s/mobilevit_s.onnx",
}

# 模型优先级: 去重时优先保留的模型标签
MODEL_PRIORITY = ["MBV2", "R50", "YOLO", "BERT", "MVIT", "QWEN"]

# 算子分组 (用于命名)
OP_GROUPS = {
    "Conv": "Conv{class}",
    "MatMul": "MatMul",
    "LayerNormalization": "LayerNorm",
    "RMSNorm": "RMSNorm",
    "Softmax": "Softmax",
    "GELU": "GELU",
    "SiLU": "SiLU",
    "GlobalAveragePool": "GlobalAvgPool",
    "MaxPool": "MaxPool",
    "AveragePool": "AvgPool",
    "Concat": "Concat",
    "Reshape": "Reshape",
    "Transpose": "Transpose",
    "Add": "Add",
    "Mul": "Mul",
}


def load_onnx(path):
    """加载 ONNX 模型并构建形状映射（含 shape inference）"""
    m = onnx.load(str(path))
    # 运行 shape inference — 填充所有中间 tensor 的形状
    try:
        from onnx import shape_inference
        m = shape_inference.infer_shapes(m, strict_mode=False)
    except Exception as e:
        print(f"  [WARN] shape_inference failed for {path.name}: {e}")

    shapes = {}
    for vi in list(m.graph.input) + list(m.graph.output) + list(m.graph.value_info):
        dims = [d.dim_value if d.dim_value > 0 else 1 for d in vi.type.tensor_type.shape.dim]
        shapes[vi.name] = tuple(dims)
    for init in m.graph.initializer:
        shapes[init.name] = tuple(init.dims)
    return m, shapes


def make_shape_key(shape):
    """压缩形状信息: (1,64,112,112) → C64_S112 / (768,768) → 768x768"""
    if shape is None or len(shape) == 0:
        return "?"
    if len(shape) == 2:
        return f"{shape[0]}x{shape[1]}"
    if len(shape) >= 4:
        c, h = shape[1], shape[2]
        return f"C{c}_S{h}"
    if len(shape) == 3:
        return f"C{shape[1]}_L{shape[2]}"
    return "x".join(str(d) for d in shape)


def classify_conv(attrs):
    """判断卷积类型: 1x1 / 3x3 / DW"""
    ks = attrs.get("kernel_shape", [1, 1])
    group = attrs.get("group", 1)
    stride = attrs.get("strides", [1, 1])

    if group > 1:
        tag = "DWConv"
    elif tuple(ks) == (1, 1) or ks == [1, 1]:
        tag = "Conv1x1"
    elif tuple(ks) == (3, 3) or ks == [3, 3]:
        tag = "Conv3x3"
    else:
        tag = f"Conv{ks[0]}x{ks[1]}"

    # stride 信息
    if stride and tuple(stride) != (1, 1):
        tag += f"_S{stride[0]}"
    return tag


def extract_operators(model_tag, model_path):
    """从单个模型提取所有算子"""
    if model_path is None or not model_path.exists():
        return []

    m, shapes = load_onnx(model_path)
    ops = []

    for node in m.graph.node:
        op_type = node.op_type
        if op_type in ("Constant", "Identity", "Cast", "Shape", "Gather", "Unsqueeze"):
            continue

        # 获取输入形状
        in_shapes = []
        for inp in node.input:
            if inp in shapes:
                in_shapes.append(shapes[inp])

        # 跳过没有形状信息的节点
        if not in_shapes:
            continue

        # 提取关键属性
        attrs = {}
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.INTS:
                attrs[attr.name] = list(attr.ints)
            elif attr.type == onnx.AttributeProto.INT:
                attrs[attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.FLOAT:
                attrs[attr.name] = attr.f

        ops.append({
            "model": model_tag,
            "op_type": op_type,
            "input_shapes": in_shapes,
            "output_name": node.output[0] if node.output else "",
            "attrs": attrs,
            "node": node,
        })

    return ops


def make_dedup_key(op, focus_ops_only=True):
    """生成去重 key"""
    ot = op["op_type"]
    shapes = op["input_shapes"]

    # 只关注核心计算算子
    if focus_ops_only and ot not in ("Conv", "MatMul", "LayerNormalization",
                                      "Softmax", "GELU", "GlobalAveragePool",
                                      "MaxPool", "AveragePool", "Concat"):
        return None  # 跳过

    if ot == "Conv":
        attrs = op["attrs"]
        group = attrs.get("group", 1)
        ks = tuple(attrs.get("kernel_shape", [1, 1]))
        stride = tuple(attrs.get("strides", [1, 1]))
        tag = classify_conv(attrs)
        # 对 Conv: 用 weight_shape 作为 key (第二个输入)
        weight = shapes[1] if len(shapes) > 1 else None
        input_shape = shapes[0] if shapes else None
        return (tag, input_shape, weight, ks, stride, group)

    elif ot == "MatMul":
        a = shapes[0] if len(shapes) > 0 else None
        b = shapes[1] if len(shapes) > 1 else None
        return (ot, a, b)

    else:
        return (ot, shapes[0] if shapes else None)


def make_operator_name(op):
    """生成可读的算子名: Conv1x1_MBV2_C32_S112_K16"""
    ot = op["op_type"]
    model = op["model"]
    shapes = op["input_shapes"]
    attrs = op["attrs"]

    if ot == "Conv":
        group = attrs.get("group", 1)
        ks = attrs.get("kernel_shape", [1, 1])
        tag = "DWConv" if group > 1 else f"Conv{ks[0]}x{ks[1]}"

        weight = shapes[1] if len(shapes) > 1 else None
        input_s = shapes[0] if len(shapes) > 0 else None

        parts = [tag, model]
        if input_s and len(input_s) >= 4:
            parts.append(f"C{input_s[1]}_S{input_s[2]}")
        if weight and len(weight) >= 1:
            parts.append(f"K{weight[0]}")
        stride = attrs.get("strides", [1, 1])
        if stride and tuple(stride) != (1, 1):
            parts.append(f"S{stride[0]}")

        return "_".join(parts)

    elif ot == "MatMul":
        a = shapes[0] if len(shapes) > 0 else None
        b = shapes[1] if len(shapes) > 1 else None
        parts = ["MatMul", model]
        if b and len(b) >= 2:
            parts.append(f"{b[0]}x{b[1]}")
        return "_".join(parts)

    elif ot == "LayerNormalization":
        s = shapes[0] if shapes else None
        if s and len(s) >= 3:
            return f"LayerNorm_{model}_L{s[1]}_H{s[2]}"
        return f"LayerNorm_{model}"

    elif ot == "Softmax":
        s = shapes[0] if shapes else None
        if s and len(s) >= 3:
            return f"Softmax_{model}_{s[1]}x{s[2]}"
        return f"Softmax_{model}"

    elif ot == "GELU":
        s = shapes[0] if shapes else None
        return f"GELU_{model}" + (f"_{make_shape_key(s)}" if s else "")

    elif ot in ("GlobalAveragePool", "MaxPool", "AveragePool"):
        s = shapes[0] if shapes else None
        parts = [ot, model]
        if s:
            parts.append(make_shape_key(s))
        return "_".join(parts)

    elif ot == "Concat":
        s = shapes[0] if shapes else None
        return f"Concat_{model}_{make_shape_key(s)}"

    else:
        return f"{ot}_{model}"


def _classify_input(shape, ot):
    """根据形状推断输入类型: 'data' | 'weight' | 'bias' | 'unknown'"""
    ndim = len(shape)
    if ot == "Conv":
        if ndim == 1:
            return "bias"
        if ndim == 4:
            _, c, h, w = shape
            # data: 空间维度 >> 核大小 (通常 > 7)
            # weight: 空间维度 == 核大小 (1, 3, 5, 7)
            # 用阈值区分: 空间维 ≤ 7 → weight, > 7 → data
            if h <= 7 and w <= 7:
                return "weight"
            else:
                return "data"
        return "unknown"
    elif ot == "MatMul":
        # 第一个输入是 data, 第二个是 weight (2D)
        return "data"
    elif ot in ("LayerNormalization",):
        # LayerNorm: data (2D+), weight (1D), bias (1D)
        if ndim == 1:
            return "bias"
        return "data"
    else:
        return "data"


def generate_onnx(op, output_path):
    """根据算子信息生成单算子 ONNX 文件 — 按形状自动识别 data/weight/bias 并正确排序"""
    ot = op["op_type"]
    node = op["node"]
    attrs = op["attrs"]

    shapes = op["input_shapes"]

    # 1. 分类每个输入
    classified = {"data": [], "weight": [], "bias": [], "unknown": []}
    for i, inp_name in enumerate(node.input):
        shape = shapes[i] if i < len(shapes) else None
        if shape is None or any(d <= 0 for d in shape):
            classified["unknown"].append((inp_name, None, i))
            continue
        kind = _classify_input(shape, ot)
        classified[kind].append((inp_name, shape, i))

    # 2. 构建标准顺序的输入列表
    # Conv: [data, weight, bias]
    # MatMul: [data, weight]
    # 其他: 按原顺序
    if ot == "Conv":
        ordered = classified["data"] + classified["weight"] + classified["bias"] + classified["unknown"]
    elif ot == "MatMul":
        ordered = classified["data"] + classified["weight"] + classified["unknown"]
    else:
        ordered = classified["data"] + classified["weight"] + classified["bias"] + classified["unknown"]

    # 3. 生成 value_info 和 initializer
    graph_inputs = []
    initializers = []
    input_names = []
    data_count = 0  # 计数 data 输入，MatMul 第二个 data 应该是 weight

    for inp_name, shape, orig_idx in ordered:
        if shape is None:
            continue

        kind = _classify_input(shape, ot)
        # MatMul 特殊处理: 第一个输入是 data，其余都当 weight
        if ot == "MatMul":
            if data_count == 0:
                kind = "data"
            else:
                kind = "weight"
        elif ot in ("LayerNormalization",):
            if kind == "bias":
                pass  # keep bias classification
        if kind == "data":
            data_count += 1

        is_weight_like = (kind == "weight" or kind == "bias")

        if is_weight_like:
            # 权重/偏置 → initializer (随机数据)
            dtype = TensorProto.FLOAT
            data = np.random.randn(*shape).astype(np.float32)
            init = helper.make_tensor(
                name=inp_name, data_type=dtype, dims=list(shape),
                vals=data.flatten().tolist()
            )
            initializers.append(init)
            input_names.append(inp_name)
        else:
            # 真正的数据输入
            vi = helper.make_tensor_value_info(inp_name, TensorProto.FLOAT, list(shape))
            graph_inputs.append(vi)
            input_names.append(inp_name)

    # 4. 推测输出形状 — 使用 data 输入的 shape
    data_shapes = [s for _, s, _ in ordered if _classify_input(s, ot) == "data"]
    data_shape = data_shapes[0] if data_shapes else op["input_shapes"][0]

    if ot == "Conv":
        weight_shapes = [s for _, s, _ in ordered if _classify_input(s, ot) == "weight"]
        bias_shapes = [s for _, s, _ in ordered if _classify_input(s, ot) == "bias"]
        weight_s = weight_shapes[0] if weight_shapes else [1, 1, 1, 1]
        ks = attrs.get("kernel_shape", [1, 1])
        strides = attrs.get("strides", [1, 1])
        pads = attrs.get("pads", [0, 0, 0, 0])
        groups = attrs.get("group", 1)
        out_c = weight_s[0]  # K (output channels)
        out_h = (data_shape[2] + 2*pads[0] - ks[0]) // strides[0] + 1
        out_w = (data_shape[3] + 2*pads[1] - ks[1]) // strides[1] + 1
        out_shape = [data_shape[0], out_c, out_h, out_w]
    elif ot == "MatMul":
        # 第二个输入 (weight) 的最后一维是输出维度
        if len(op["input_shapes"]) >= 2:
            b_s = op["input_shapes"][1]  # weight shape
        else:
            b_s = data_shape
        if len(b_s) >= 2:
            out_shape = list(data_shape[:-1]) + [b_s[-1]]
        else:
            out_shape = [data_shape[0], b_s[-1]]
    elif ot in ("LayerNormalization", "GELU", "Softmax",
                "GlobalAveragePool", "MaxPool", "AveragePool"):
        out_shape = list(data_shape)
    elif ot == "Concat":
        all_shapes = [s for _, s, _ in ordered if s is not None]
        total = sum(s[1] for s in all_shapes[1:])
        out_shape = [all_shapes[0][0], all_shapes[0][1] + total] + list(all_shapes[0][2:])
    else:
        out_shape = list(data_shape)

    output_name = node.output[0] if node.output else "output"

    # 5. 处理 rank 不匹配: MatMul 3D输入 → 插 Reshape
    graph_nodes = []
    actual_input_names = list(input_names)

    if ot == "MatMul" and len(data_shape) > 2:
        # 插入 Reshape 将 [B, M, K] → [B*M, K]
        orig_name = input_names[0]
        flat_name = orig_name + "_flat"
        batch_size = 1
        for d in data_shape[:-1]:
            batch_size *= d
        k_dim = data_shape[-1]

        # Reshape 输入
        reshape_input = helper.make_node(
            "Reshape", inputs=[orig_name, f"{orig_name}_shape"],
            outputs=[flat_name], name=f"{orig_name}_reshape"
        )
        reshape_shape = helper.make_tensor(
            f"{orig_name}_shape", TensorProto.INT64, dims=[2],
            vals=np.array([batch_size, k_dim], dtype=np.int64).tolist()
        )
        graph_nodes.append(reshape_input)
        initializers.append(reshape_shape)

        actual_input_names[0] = flat_name
        # 更新 MatMul 输出名
        matmul_out = output_name + "_mm"
        output_name = matmul_out

        # MatMul 后的 Reshape
        unflat_name = node.output[0] if node.output else "output"
        reshape_output = helper.make_node(
            "Reshape", inputs=[matmul_out, f"{unflat_name}_shape"],
            outputs=[unflat_name], name=f"{unflat_name}_reshape"
        )
        reshape_out_shape = helper.make_tensor(
            f"{unflat_name}_shape", TensorProto.INT64, dims=[len(out_shape)],
            vals=np.array(out_shape, dtype=np.int64).tolist()
        )
        initializers.append(reshape_out_shape)

    # 主算子
    new_attrs = []
    for k, v in attrs.items():
        if isinstance(v, list):
            new_attrs.append(helper.make_attribute(k, v))
        elif isinstance(v, int):
            new_attrs.append(helper.make_attribute(k, v))
        elif isinstance(v, float):
            new_attrs.append(helper.make_attribute(k, v))

    new_node = helper.make_node(
        op_type=ot, inputs=actual_input_names, outputs=[output_name],
        name=output_name
    )
    for a in new_attrs:
        if a.name not in [na.name for na in new_node.attribute]:
            new_node.attribute.append(a)

    graph_nodes.append(new_node)

    if ot == "MatMul" and len(data_shape) > 2:
        graph_nodes.append(reshape_output)

    final_output_name = node.output[0] if node.output else "output"
    outputs = [helper.make_tensor_value_info(final_output_name, TensorProto.FLOAT, out_shape)]

    graph = helper.make_graph(
        nodes=graph_nodes, name=output_name,
        inputs=graph_inputs, outputs=outputs,
        initializer=initializers
    )

    # opset 18 + ir_version 10
    op_id = onnx.OperatorSetIdProto()
    op_id.domain = ""
    op_id.version = 18

    model = helper.make_model(graph, opset_imports=[op_id], ir_version=10)
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    return out_shape


def main():
    parser = argparse.ArgumentParser(description="从整模型提取去重单算子")
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "models/single_ops_extracted"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # 1. 提取所有算子
    all_ops = []
    for tag, path in MODELS.items():
        ops = extract_operators(tag, path)
        all_ops.extend(ops)
        print(f"  {tag}: {len(ops)} 个算子")

    print(f"\n总计: {len(all_ops)} 个算子")

    # 2. 一级去重: 精确匹配 (op_type, shapes, attrs)
    seen = {}
    for op in all_ops:
        key = make_dedup_key(op, focus_ops_only=True)
        if key is None:
            continue
        if key in seen:
            new_pri = MODEL_PRIORITY.index(op["model"]) if op["model"] in MODEL_PRIORITY else 99
            old_pri = MODEL_PRIORITY.index(seen[key]["model"]) if seen[key]["model"] in MODEL_PRIORITY else 99
            if new_pri < old_pri:
                seen[key] = op
        else:
            seen[key] = op

    # 3. 二级去重: 同 (op_type_bucket + input_shape) 只保留 FLOPs 最大的作为代表
    def bucket_key(op):
        """粗粒度 bucket: 同一类操作 + 相同数据输入形状 → 一个桶"""
        ot = op["op_type"]
        shapes = op["input_shapes"]
        attrs = op["attrs"]
        if ot == "Conv":
            tag = classify_conv(attrs)
            data_shape = shapes[0] if shapes else None
            return (tag, data_shape)
        elif ot == "MatMul":
            a_shape = shapes[0] if len(shapes) > 0 else None
            return (ot, a_shape)
        else:
            return (ot, shapes[0] if shapes else None)

    buckets = defaultdict(list)
    for op in seen.values():
        bk = bucket_key(op)
        buckets[bk].append(op)

    deduped = []
    for bk, ops in buckets.items():
        # 按 FLOPs (用输入/输出通道乘积估算) 选最大的
        def flops_est(op):
            shapes = op["input_shapes"]
            ot = op["op_type"]
            if ot == "Conv":
                w = shapes[1] if len(shapes) > 1 else [1]
                s = shapes[0] if shapes else [1]
                return np.prod(w) * (s[2]*s[3] if len(s) >= 4 else 1)
            elif ot == "MatMul":
                a = shapes[0] if len(shapes) > 0 else [1]
                b = shapes[1] if len(shapes) > 1 else [1]
                return a[0] * b[-1] * (a[-1] if len(a) >= 2 else 1)
            return 1
        best = max(ops, key=flops_est)
        deduped.append(best)

    print(f"一级去重: {len(seen)} → 二级bucket去重: {len(deduped)} 个唯一算子\n")

    # 按类型分组排序
    deduped.sort(key=lambda o: (o["op_type"], make_dedup_key(o)[0] if make_dedup_key(o) else ""))

    print(f"去重后: {len(deduped)} 个唯一算子\n")

    # 3. 按类型分组展示
    groups = defaultdict(list)
    for op in deduped:
        key = make_dedup_key(op)
        tag = key[0] if key else op["op_type"]
        groups[tag].append(op)

    for tag in sorted(groups.keys()):
        print(f"## {tag} ({len(groups[tag])} 个)")
        for op in groups[tag]:
            name = make_operator_name(op)
            shapes_str = " → ".join(str(s) for s in op["input_shapes"][:2])
            print(f"    {name:50s}  {shapes_str}")
        print()

    if args.dry_run:
        return

    # 4. 生成 ONNX 文件
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for op in deduped:
        name = make_operator_name(op)
        fpath = output_dir / f"{name}.onnx"
        try:
            out_shape = generate_onnx(op, fpath)
            generated.append((name, fpath, out_shape))
        except Exception as e:
            print(f"  ⚠️ {name}: 生成失败 ({e})")

    print(f"\n生成 {len(generated)} 个 ONNX 文件到 {output_dir}")

    # 5. 输出 JSON 索引
    index = []
    for name, fpath, out_shape in generated:
        # 找到原始 op 信息
        for op in deduped:
            if make_operator_name(op) == name:
                index.append({
                    "name": name,
                    "file": str(fpath.name),
                    "op_type": op["op_type"],
                    "model": op["model"],
                    "input_shapes": [list(s) for s in op["input_shapes"]],
                    "output_shape": list(out_shape) if out_shape else [],
                })
                break

    index_path = output_dir / "_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"索引: {index_path}")


if __name__ == "__main__":
    main()
