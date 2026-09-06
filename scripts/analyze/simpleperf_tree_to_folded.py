#!/usr/bin/env python3
"""把 simpleperf report -g 的输出转换为 stackcollapse 折叠格式，供 flamegraph.pl 使用。

simpleperf 树格式:
  Children Self ... Symbol
  28.95% 0.00% ... _start_main          <- 根条目（表格行）
         |
         -- _start_main                 <- 树的根节点
            |
             -- __libc_init
                 |
                 |--99.98%-- main
                 |    |
                 |    |--...-- child
"""
import re
import sys


def parse(path, min_pct=0.0):
    """解析 simpleperf -g 输出，返回折叠行列表 [('root;sub;sym', pct), ...]。"""
    folded = []
    stack = []  # (indent, symbol)
    header_idx = None
    in_header = True

    with open(path, encoding='utf-8') as f:
        lines = f.readlines()

    # 找表头行
    for i, line in enumerate(lines):
        if re.match(r'^\s*Children\s+Self\s+Command', line):
            header_idx = i
            break
    if header_idx is None:
        return folded

    for line in lines[header_idx + 1:]:
        s = line.rstrip('\n')
        if not s.strip():
            continue

        # 根条目（表格行，Overhead 开头）
        m_root = re.match(r'^\s*(\d+\.\d+)%\s+\d+\.\d+%\s+(\S+)\s+\d+\s+\d+\s+(\S+)\s+(\S.*)$', s)
        if m_root:
            pct, cmd, dso, sym = m_root.groups()
            if float(pct) < min_pct:
                stack = []
                continue
            stack = [(0, sym.strip())]
            continue

        # 树节点行: 前缀 + [pct]-- + 符号
        m_node = re.match(r'^(\s*)([\s|\-]*)--(\d+\.\d+%)?--\s*(.*)$', s)
        if not m_node:
            continue  # 分隔线 | 等
        leading, prefix, pct_str, sym = m_node.groups()
        sym = sym.strip()
        if not sym:
            continue

        # 深度 = 前导空格数 + 前缀中非空标记长度（| 每列 +1）
        indent = len(leading) + len(prefix) - prefix.count('-')
        pct = float(pct_str.rstrip('%')) if pct_str else 0.0
        if pct < min_pct and pct > 0:
            continue
        if pct == 0 and prefix.count('-') == 0:
            # 无百分比、无 -- 的节点（如根 __libc_init）标记为 depth 标记
            pass

        # 维护祖先栈
        while stack and indent <= stack[-1][0]:
            stack.pop()
        stack.append((indent, sym))

        if pct > 0:
            callpath = ';'.join(x[1] for x in stack)
            folded.append((callpath, pct))

    return folded


def main():
    if len(sys.argv) < 2:
        print('Usage: simpleperf_tree_to_folded.py <calltree.txt> [min_pct]', file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    min_pct = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

    for callpath, pct in parse(path, min_pct):
        # flamegraph 需要整数计数；用 pct 缩放到整数
        count = max(1, int(pct * 100))
        print(f'{callpath} {count}')


if __name__ == '__main__':
    main()
