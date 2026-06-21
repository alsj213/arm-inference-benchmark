#!/usr/bin/env python3
"""
MNN 模型分析工具 — 分析算子结构和融合状态

用法:
  python3 mnn-model-analyzer.py --model model.mnn.json
  python3 mnn-model-analyzer.py --model model.mnn.json --detail
  python3 mnn-model-analyzer.py --model model.mnn.json --fusion-only
"""

import json, sys, re, argparse
from collections import defaultdict, Counter

# 已知可融合的算子对 (MNN CPU 后端)
FUSABLE_PATTERNS = [
    ("Convolution", "UnaryOp", "Conv+Activation (ReLU/SiLU/GELU)"),
    ("Convolution", "BinaryOp", "Conv+BN"),
    ("LayerNorm", "Convolution", "LayerNorm+MatMul (folding)"),
    ("LayerNorm", "MatMul", "LayerNorm+MatMul (folding)"),
    ("Add", "LayerNorm", "Residual+LayerNorm fusion"),
    ("Softmax", "Attention", "Softmax→Attention 融合"),
    ("Reshape", "Convolution", "Reshape+Conv (layout合并)"),
    ("ConvertTensor", "Convolution", "类型转换+Conv"),
]

# 已融合的算子（不会单独出现）
FUSED_OPS = {
    "Softmax": "已融合到 Attention",
    "Relu": "已融合到 Convolution",
    "Relu6": "已融合到 Convolution",
    "GELU": "可能已融合到 Convolution",
}

def analyze(model_file, show_detail=False, fusion_only=False):
    with open(model_file) as f:
        model = json.load(f)
    
    oplists = model.get('oplists', [])
    if not oplists:
        print("❌ 未找到 oplists")
        return
    
    # 1. 算子统计
    op_types = Counter()
    fused_check = {}
    layer_ops = defaultdict(list)
    standalone = []
    
    for i, op in enumerate(oplists):
        t = op.get('type', 'unknown')
        n = op.get('name', '')
        op_types[t] += 1
        
        # 检查是否已知已融合
        if t in FUSED_OPS:
            fused_check[t] = FUSED_OPS[t]
        
        # 按层分组
        m = re.search(r'/layers\.(\d+)/([^/]+)', n)
        if m:
            layer_ops[int(m.group(1))].append((i, t, n))
        else:
            standalone.append((i, t, n))
    
    total = sum(op_types.values())
    
    if not fusion_only:
        # 基本信息
        print(f"模型: {model.get('bizCode', 'unknown')}")
        print(f"总算子: {total}")
        print(f"版本: {model.get('mnn_uuid', 'N/A')[:16]}...")
        print()
        
        # 算子类型分布
        print("=== 算子类型分布 ===")
        print(f"{'算子类型':<20} {'数量':<8} {'占比':<8} {'融合状态'}")
        print('-' * 55)
        
        for t, cnt in op_types.most_common():
            pct = cnt * 100 / total
            fusion_status = ''
            if t in FUSED_OPS:
                fusion_status = f'✅ {FUSED_OPS[t]}'
            elif t in ('LayerNorm', 'BinaryOp', 'UnaryOp', 'Add'):
                fusion_status = '❌ 未融合'
            print(f'{t:<20} {cnt:<8} {pct:<7.1f}% {fusion_status}')
        
        # 融合判定
        print(f"\n=== 融合判定 ===")
        for fused_op, desc in FUSED_OPS.items():
            if fused_op not in op_types:
                print(f"  ✅ {desc}")
            else:
                print(f"  ⚠️  {fused_op} 仍独立出现 ({op_types[fused_op]}次)")
        
        # 层结构
        if layer_ops:
            print(f"\n=== 层结构 (共 {len(layer_ops)} 层) ===")
            for lid in sorted(layer_ops.keys())[:3]:
                block_ops = defaultdict(list)
                for _, t, n in layer_ops[lid]:
                    b = re.search(r'/layers\.\d+/([^/]+)', n)
                    block = b.group(1) if b else '?'
                    block_ops[block].append(t)
                print(f"\n  Layer {lid}:")
                for block, ops in sorted(block_ops.items()):
                    ops_str = ' → '.join(ops[:5])
                    print(f"    {block}: {', '.join(ops[:4])}...")
            if len(layer_ops) > 6:
                print(f"\n  ... ({len(layer_ops)-6} 层相同模式)")
    
    # 3. Fusion 分析核心：检测可融合但未融合的模式
    print(f"\n=== Fusion 优化机会 ===")
    
    # 用滑窗检测相邻可融合算子对
    found_opportunities = defaultdict(int)
    for i in range(len(oplists) - 1):
        t1 = oplists[i].get('type', '')
        t2 = oplists[i+1].get('type', '')
        for t_a, t_b, desc in FUSABLE_PATTERNS:
            if t1 == t_a and t2 == t_b:
                found_opportunities[desc] += 1
    
    # 也检查 Add→LayerNorm→MatMul 三连
    for i in range(len(oplists) - 2):
        types = [oplists[i+j].get('type','') for j in range(3)]
        if types == ['Add', 'LayerNorm', 'Convolution'] or \
           types == ['Add', 'LayerNorm', 'MatMul']:
            found_opportunities['Add+LayerNorm+MatMul 三合一融合'] += 1
    
    if found_opportunities:
        for desc, cnt in sorted(found_opportunities.items(), key=lambda x: -x[1]):
            print(f"  🚩 {desc}: {cnt} 处可优化")
    else:
        print(f"  ✅ 未发现可融合模式")
    
    # 列出未融合的关键算子
    unfused = [('Add', '残差连接'), ('LayerNorm', 'RMSNorm'), 
               ('BinaryOp', '逐元素运算'), ('UnaryOp', '激活函数')]
    print(f"\n=== 未融合算子列表 ===")
    for t, desc in unfused:
        if t in op_types:
            print(f"  ❌ {t} ({desc}) × {op_types[t]} — 未融合，可优化")
    
    if fusion_only:
        return
    
    # Vision encoder 检查
    print(f"\n=== Vision 编码器 ===")
    vision_ops = [(i, t, n) for i, t, n in standalone if 'visual' in n.lower()]
    if vision_ops:
        print(f"  视觉编码器嵌入在 LLM 模型中: {len(vision_ops)} ops")
    else:
        print("  视觉编码器在单独的文件中 (visual.mnn)")
    
    # 显示非 Layer 算子的分布
    non_layer = Counter()
    for _, t, n in [(i,t,n) for i,t,n in standalone if not re.search(r'/layers\.\d+/', n)]:
        non_layer[t] += 1
    if non_layer:
        print(f"\n=== 非 Layer 算子 ({sum(non_layer.values())} ops) ===")
        for t, cnt in non_layer.most_common(10):
            print(f"  {t}: {cnt}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNN 模型分析工具')
    parser.add_argument('--model', required=True, help='MNN JSON 模型文件')
    parser.add_argument('--detail', action='store_true', help='显示详细层结构')
    parser.add_argument('--fusion-only', action='store_true', help='只显示 fusion 分析')
    args = parser.parse_args()
    analyze(args.model, args.detail, args.fusion_only)
