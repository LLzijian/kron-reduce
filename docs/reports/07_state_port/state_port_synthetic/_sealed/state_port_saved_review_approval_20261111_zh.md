# A1-S0 保存证据审核：准确实施批准

2026-11-11，Astra Ultra 科学审核。

**批准一次独立保存证据审核。** 用户已明确授权自主推进研究；本收据把后续工作限定为已完成数值记录的算术核验。新许可 `state_port_saved_review_v1_20261111` 登记并通过准确绑定守卫后，只执行指定审核脚本一次。原科学许可保持 consumed，原运行继续为 `FAIL_PRESERVED`。

| 对象 | SHA256 |
| --- | --- |
| [审核计划](../../configs/proposals/state_port_saved_review_v1.json) | `4f7544efa707ad136a631eae7d0e314d4b11d498e64a0a15f58cc98accde602a` |
| [审核协议](../protocols/state_port_saved_review_20261111_zh.md) | `99f11d32095ab876292e6c22f4fb2b45707bdb3e03432c3d10e45778b98364f3` |
| [外置来源清单](../../configs/proposals/state_port_saved_sources_v1.json) | `9c06a82592879750ef9b4a93b9c8d339f5728905dfd24346571eb577f1631841` |
| [审核脚本](../../scripts/audit_state_port_saved.py) | `b337cd31bec4457a731c467947387ec4ea62b1b033604f015b6b47ba8afae5ed` |
| [原科学计划](../../configs/proposals/state_port_synthetic_v1.json) | `2f0919965b08f56678d3bc8d0ac1ed30d1e1e75971c77f4074e256578a6dc545` |

已只读核对这些准确文件、审核算术和一次性入口。唯一来源运行是 `state_port_synthetic_v1-synthetic_correctness-20261110T161605071933Z-992738b0-967f4fb0`，源提交 `992738b0c4bfbf26e073f40a29f67cea4c4ac429`。来源清单固定 22 文件、1,305,816 字节；文件集合、尺寸与 SHA 在数组解码前核验。指定四个 NPZ 各解码一次，共 1,188,578 压缩字节；另一个失败诊断 NPZ 只核哈希。审核不会导入模型或科学执行模块，也没有模型、RHS、积分、求导、谱分解、LS 或拟合调用。

审核复算原阈值下的差分 Jacobian、路径 loss、传播参数梯度、能量原量归一和路径比较，核对固定记录集合、有限性、两个隔离负例、完整调用预算与日志区间哈希。日志的 `rollout` 类别在局部范围计数中明确映射为账本限额的 `rollouts`；这项静态审查发现的口径错误已在任何真实证据解码前修正。原 JSON 描述确认 16 条长路径均为 301×22，并有 14 组 NumPy 路径能量原量。

入口核 active policy、准确计划、脚本、来源、协议、原计划与批准收据，然后在共享 Git 目录创建排他的已消费预留。新结果以排他写入方式保存至原运行之外；异常保留审核失败账本，不重试。原运行文件、90 字节不完整 summary、原失败账本及 workflow 均不写入。

工程依据为协调者报告的三个测试文件 **18 passed（0.45 s）** 及 Ruff 通过。审核器测试使用临时小数组、授权拒绝情形和人工 330 行日志；165 对 attempted/returned、范围哈希与单复数类别映射均通过，未读取本次运行 NPZ。另一个序列化回归使用独立人工参数，调用两个点态 RHS，没有重跑 S0 参数矩阵；它与本次零科学调用的保存审核分开记录。本准确审核自身只读代码、JSON 和绑定文件哈希，没有解码 NPZ 或调用科学算子。

只有全部保存算术、来源和预算检查通过，才可在外置报告记 `scientific_checks=PASS_FROM_SAVED_EVIDENCE`；否则记录失败或未决。该字段不覆盖 `process_status=FAIL_PRESERVED`，不产生原运行成功摘要或成功 stage evidence。科学主张仍限于 A1-S0 的固定结构与梯度检查，结论保持 `KEEP_BASELINE`。
