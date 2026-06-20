# 什么是 `.cfi` 指令 — ARM64 调用栈展开原理

---

## 1. 先理解"调用栈"

当程序执行时，每次调用函数都会在**栈内存**上分配一块空间（叫"栈帧"）：

```
内存（从高地址到低地址增长）
┌──────────────────────────┐ 高地址
│  main 的栈帧              │
│  ┌──────────────────────┐│
│  │ 返回地址 → 操作系统     ││
│  │ main 的局部变量        ││
│  ├──────────────────────┤│
│  │ infer 的栈帧          ││
│  │ 返回地址 → main        ││
│  │ infer 的局部变量       ││
│  ├──────────────────────┤│
│  │ LoopL2 的栈帧         ││
│  │ 返回地址 → infer       ││  ← 当前执行位置
│  │ LoopL2 的局部变量      ││
│  └──────────────────────┘│
└──────────────────────────┘ 低地址
```

**调用栈展开**（Stack Unwinding）就是从当前函数**往回追溯**，找出是谁调用了我、谁调用了它、... 一直到 `main`。

这就是火焰图中那条"塔"的来源：
```
LoopL2 ← 当前函数
  ↑ 谁调用了我？
infer
  ↑ 谁调用了我？
main
  ↑ 谁调用了我？
操作系统
```

---

## 2. 两种展开调用栈的方式

### 方式一：Frame Pointer（帧指针）

**原理**：ARM64 CPU 有一个专门的寄存器 `x29`（叫 Frame Pointer），每个函数在入口处把**上一个函数的 x29 值保存到栈上**，形成一个链表。

```
x29 寄存器（当前帧指针）
    ↓
┌─────────────────────┐
│ LoopL2 的栈帧        │
│ x29 → infer 的栈帧   │──┐
│ 返回地址 → infer      │  │
└─────────────────────┘  │
                         ↓
                    ┌─────────────────────┐
                    │ infer 的栈帧         │
                    │ x29 → main 的栈帧    │──┐
                    │ 返回地址 → main       │  │
                    └─────────────────────┘  │
                                             ↓
                                        ┌─────────────────────┐
                                        │ main 的栈帧          │
                                        │ x29 → 操作系统        │
                                        │ 返回地址 → 操作系统     │
                                        └─────────────────────┘
```

simpleperf 沿着这个链表走，就能找到完整的调用栈。

**关键**：每个函数必须在入口处**保存 x29**，否则链表就断了。

### 方式二：DWARF（调试信息）

**原理**：编译器在 binary 中嵌入一张**规则表**（叫 `.eh_frame`），描述每个函数的栈帧布局。simpleperf 根据这张表，从栈内存中**计算**出返回地址。

```
.eh_frame 段（编译器生成的规则表）
┌──────────────────────────────────────────┐
│ 函数 LoopL2 的规则：                       │
│   - 返回地址保存在 [sp + 16]               │
│   - 上一个 x29 保存在 [sp + 8]             │
│   - 栈帧大小 = 32 字节                     │
│                                           │
│ 函数 infer 的规则：                        │
│   - 返回地址保存在 [sp + 24]               │
│   - 上一个 x29 保存在 [sp + 16]            │
│   - 栈帧大小 = 48 字节                     │
│                                           │
│ 函数 main 的规则：                         │
│   ...                                     │
└──────────────────────────────────────────┘
```

**关键**：每个函数必须有 `.eh_frame` 条目，否则 simpleperf 不知道如何展开。

---

## 3. C++ 编译器自动处理

当你写 C++ 代码时，**编译器自动生成**这些信息：

```cpp
// C++ 源码
void infer() {
    float data[100];
    compute(data);
}

// 编译器自动生成的 ARM64 汇编（简化版）
infer:
    stp x29, x30, [sp, #-16]!   // 保存 frame pointer 和返回地址
    mov x29, sp                  // 设置新的 frame pointer
    // ... 函数体 ...
    ldp x29, x30, [sp], #16     // 恢复 frame pointer 和返回地址
    ret                          // 返回

// 编译器同时自动生成 .eh_frame 条目
// （不需要你手动写）
```

所以 C++ 代码**天然支持**调用栈展开，无论是 Frame Pointer 还是 DWARF 方式。

---

## 4. 手写汇编的问题

MNN 的优化函数是**手写汇编**，不是编译器生成的：

```asm
// MNN 的 LoopL2（简化版，实际更复杂）
LoopL2:
    // 没有保存 x29！
    // 没有 .cfi 指令！
    
    // 直接开始计算
    ldr q0, [x0]
    ldr q1, [x1]
    fmla v2.4s, v0.4s, v1.4s
    str q2, [x2]
    
    // 直接返回
    ret
```

**问题**：
1. **没有 `stp x29, x30, ...`**：没有保存 Frame Pointer，链表断了
2. **没有 `.cfi` 指令**：没有告诉编译器如何生成 `.eh_frame` 条目，DWARF 也断了

---

## 5. `.cfi` 指令是什么？

**CFI = Call Frame Information**（调用帧信息）

`.cfi` 是汇编指令，告诉汇编器（assembler）如何生成 `.eh_frame` 条目。它们**不会生成机器码**，只是元数据。

```asm
// 有 .cfi 指令的汇编（正确的）
LoopL2:
    .cfi_startproc                  // ← 开始一个函数的 CFI 记录
    stp x29, x30, [sp, #-16]!       // 保存 frame pointer
    .cfi_def_cfa_offset 16          // ← 栈帧偏移了 16 字节
    .cfi_offset x29, -16            // ← x29 保存在 [sp - 16]
    .cfi_offset x30, -8             // ← x30（返回地址）保存在 [sp - 8]
    mov x29, sp                     // 设置 frame pointer
    
    // ... 函数体 ...
    
    ldp x29, x30, [sp], #16         // 恢复
    .cfi_restore x30                // ← x30 恢复了
    .cfi_restore x29                // ← x29 恢复了
    .cfi_def_cfa_offset 0           // ← 栈帧偏移恢复为 0
    ret
    .cfi_endproc                    // ← 结束 CFI 记录
```

### 常用 `.cfi` 指令

| 指令 | 含义 |
|------|------|
| `.cfi_startproc` | 开始一个函数的 CFI 记录 |
| `.cfi_endproc` | 结束一个函数的 CFI 记录 |
| `.cfi_def_cfa_offset N` | 栈帧大小改变了 N 字节 |
| `.cfi_offset reg, N` | 寄存器 reg 保存在 [CFA + N] |
| `.cfi_restore reg` | 寄存器 reg 恢复到调用时的值 |
| `.cfi_def_cfa reg, N` | CFA（Canonical Frame Address）的计算规则 |

---

## 6. 为什么 MNN 的汇编没有 `.cfi`？

```asm
// MNN 实际的 LoopL2（简化版）
LoopL2:
    // 没有 .cfi_startproc
    // 没有保存 x29
    
    // 直接做矩阵乘法
    ldr q0, [x0], #16
    ldr q1, [x1], #16
    fmla v2.4s, v0.4s, v1.4s
    // ... 重复多次 ...
    
    ret
    // 没有 .cfi_endproc
```

**原因**：
1. **性能优先**：保存/恢复 x29 需要额外的指令，影响性能
2. **历史原因**：MNN 的汇编可能是从早期版本继承的，当时没考虑 profiling
3. **工作量大**：给每个汇编函数添加 `.cfi` 指令需要大量工作

**后果**：
- Frame Pointer 模式：x29 没有保存 → 链表断了 → 无法展开
- DWARF 模式：没有 `.eh_frame` 条目 → 不知道如何展开 → 无法展开
- 结果：火焰图中 ~38% 的采样只有 2 层深度

---

## 7. 完整的例子

### 有 `.cfi` 的函数（调用栈完整）

```asm
// 编译器生成的 C++ 函数
MNN::Pipeline::execute:
    .cfi_startproc
    stp x29, x30, [sp, #-32]!
    .cfi_def_cfa_offset 32
    .cfi_offset x29, -32
    .cfi_offset x30, -24
    mov x29, sp
    
    // 调用 LoopL2
    bl LoopL2
    
    ldp x29, x30, [sp], #32
    .cfi_restore x30
    .cfi_restore x29
    .cfi_def_cfa_offset 0
    ret
    .cfi_endproc
```

simpleperf 看到 `.eh_frame`，知道：
- 函数入口：sp -= 32, x29 在 [sp+0], x30 在 [sp+8]
- 可以从栈上读取返回地址和上一个 x29

### 没有 `.cfi` 的函数（调用栈中断）

```asm
// MNN 手写的汇编
LoopL2:
    // 没有 .cfi_startproc
    // 没有 stp x29, x30
    // 没有 .cfi 指令
    
    ldr q0, [x0]
    fmla v2.4s, v0.4s, v1.4s
    
    ret
    // 没有 .cfi_endproc
```

simpleperf 看不到任何信息：
- 不知道栈帧布局
- 不知道返回地址在哪里
- **调用栈在这里中断**

---

## 8. 总结

```
C++ 代码（编译器生成）
├── 自动生成 stp x29, x30（保存帧指针）
├── 自动生成 .cfi 指令（生成 .eh_frame）
├── 调用栈完整 ✓
└── 火焰图正常 ✓

手写汇编（如 MNN 的 LoopL2）
├── 没有 stp x29, x30（不保存帧指针）
├── 没有 .cfi 指令（不生成 .eh_frame）
├── 调用栈中断 ✗
└── 火焰图只有 2 层 ✗
```

**一句话总结**：`.cfi` 指令是汇编代码写给 simpleperf 的"说明书"，告诉它如何从栈内存中找到调用链。没有这个说明书，simpleperf 就迷路了。
