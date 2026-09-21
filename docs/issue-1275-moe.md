# Issue #1275：DeepSeek V4.1 Flash MoE 子问题说明与实现

## 这部分要解决什么

[Issue #1275](https://github.com/hw-native-sys/pypto-lib/issues/1275) 要把
DeepSeek V4.1 Flash 从 Attention 的复制式 TP 布局接到标准的
Sequence Parallel（SP）和 Expert Parallel（EP）链路。对 MoE 来说，关键约束
是：进入 MoE 前，每个 EP rank 只持有自己负责的、互不重复的本地 token；每个
token 的 top-k 路由发往对应的 expert owner；expert 计算完成后，结果回到发起
该 token 的 source rank，并在那里完成 top-k 加权汇总。

原来的 V4.1 Flash MoE 接口仍带有 `token_owners`、`tp_rank`（以及未使用的
`group_base`）。它假定每个 rank 都拿到 TP 复制的完整 token 表，然后在
dispatch/combine 中过滤“owner”行。这是 Attention-TP 的语义，不能用于新的
SP→EP 布局：如果复制 token 直接进入标准 EP dispatch，同一 token 会被重复派发。

本次只实现 issue 的 MoE 子范围。Attention 的 TP AllGather、output
ReduceScatter、embedding/模型入口出口和跨层 token remap 属于 issue 的其他工作，
没有在这个分支中修改。

## 实现方式

### 1. MoE ABI 改为本地 token 语义

- `moe`、`moe_test` 和内部 `_moe_core` 不再接收 `token_owners`、`tp_rank` 或
  `group_base`。
- L3 `l3_moe` 的 `num_tokens` 改为形状 `[EP_SIZE]` 的 `INT32` tensor；第
  `r` 个元素是 rank `r` 的有效本地 token 数，范围为 `[0, MOE_TOKENS]`。
  host driver 在启动 rank `r` 时读取该元素，并把它作为该 rank kernel 的标量
  `num_tokens`。
- 标量命令行参数仍保留为兼容用法，会广播到所有 EP rank；同时支持
  `--num-tokens-per-rank 8,0,...` 来构造不等长或空分片。

这样固定容量 buffer 仍可静态编译，但每张卡可以只有自己的有效前缀，padding 行
不会被 gate、shared expert 或 routed dispatch 当作有效 token。

### 2. dispatch/combine 只处理 source rank 的本地前缀

`ep_transport.py` 的三段 dispatch（路由元数据、计数、payload）删除了 owner
判断。对每个 `t < num_tokens[my_rank]` 的 top-k assignment 都会：

1. 根据 global expert id 写入对应 destination/expert lane；
2. 通过 EP window 发送激活、scale、route id 和 route weight；
3. 在目标 rank 的 routed expert 完成后，按 route id 把结果写回原 source rank。

combine 只按 source rank 和本地 token 行归约
`shared_output[t] + Σ routed_output[t * TOPK + k]`，因此不会重复计算或过滤
Attention-TP 副本。`RECV_MAX = EP_SIZE * MOE_TOKENS` 仍足以容纳每个来源的最大
assignment 数，即使各 rank 的有效 token 数不同。

### 3. shared expert、padding 和空 rank

shared expert 保留固定容量 tile 的设备 ABI，但 gate 会把本地有效前缀之外的
activation 和 routing metadata 清成零；因此 padding 行不参与有效的 shared
结果、dispatch 或 combine。`_moe_core` 先把输出 buffer 清零，再由 combine
写有效行；没有 routed transport 时的诊断路径也只写本地有效行。空 rank 的
count 为 0 时不会产生 dispatch route，仍会完成必要的 EP barrier，输出保持
确定的 padding/residual 结果。

A5 对零任务有额外限制：`spmd(0)` 和 `parallel(0)` 会在设备端形成非法的
AICPU 调度。`gate.py` 现在为零 token rank 发起一个安全的 scratch task，并用
active mask 让它不产生可见的 activation；`expert_routed.py` 对零接收计数发起
一个全 padding tile，并将 `valid_rows` 截为 0。这样空 rank 仍参加同步，但不
读取或写入有效 token。

### 4. golden 与验证器同步

- `golden_moe_core`、standalone dispatch/combine golden 都按每个 source 的
  count 生成路由、接收计数和结果。
- 删除 owner fixture 和所有 `token_owners` 断言。
- `x_next` comparator 比较完整的固定容量输出，而不是用一个共享的
  `valid_rows` 截断每个 rank。这样不等长/空 rank 的尾部不会被比较器静默跳过，
  NaN/Inf 或错误写入能够被发现。

## 修改文件

- `models/deepseek_v4_1_flash/moe.py`
- `models/deepseek_v4_1_flash/ep_transport.py`
- `models/deepseek_v4_1_flash/gate.py`
- `models/deepseek_v4_1_flash/expert_routed.py`
- 本说明文档

没有修改 Attention、embedding、模型 orchestration 或其他模型目录；这四个代码
文件都属于 V4.1 Flash MoE 的 gate、expert 或 EP transport 路径。

## 验证计划与边界

验证使用 `.conda` 的 `pypto-torch214`（Torch 2.14）环境、`task-submit` 的
真实 A5 NPU（命令带 `-p a5`，不使用 `a5sim`）。本次已完成：

| 场景 | 任务 | 结果 |
| --- | --- | --- |
| gate standalone，`num_tokens=0`，A5 单卡 | `task_20260922_001533_64664824777` | PASS |
| EP2/TP1 完整 mHC+MoE，counts `[8, 0]` | `task_20260922_001730_65399129410` | PASS |
| EP2/TP1 完整 mHC+MoE，counts `[16, 16]` | `task_20260922_002602_72381511853` | PASS |
| EP2/TP1 完整 mHC+MoE，counts `[8, 8]` | `task_20260922_003148_74761520852` | PASS |
| EP2 dispatch standalone，counts `[8, 0]` | `task_20260921_231246_24424610709` | PASS |

完整 MoE 的三个输出 `next_pre_mix`、`x_mixed`、`x_next` 均通过比较器；`x_next`
使用完整固定容量张量比较，不再用共享的 `valid_rows` 前缀截断。

该分支的结论只适用于 MoE ABI 和 EP transport。即使这些测试通过，也不能把
issue #1275 的 Attention SP（AllGather/ReduceScatter）和完整模型入口到出口链路
视为已经完成。
