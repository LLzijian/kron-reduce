# 逐机耦合与记忆 S0 结果补录

[返回主题页](../README.md)

## 执行协议

<a id="doc-f2d63f33e87f"></a>

<a id="doc-f2d63f33e87f-逐机耦合与记忆-s0-结果补录"></a>

### 逐机耦合与记忆 S0 结果补录

原 S0 保留为失败运行。四个正向组合与删项对照的逐事件评分已经保存。运行在合并结果时无法序列化 NumPy 整数，两个负对照的奇异谱未落盘。

对原运行目录的全部文件核对预登记字节哈希，读取四组已保存 JSON。仅计算原协议的静止 T1M0 与等衰减率 T0M1 两个负对照，将秩转换为 Python 整数并写入新目录。正向模型调用、轨迹解码、拟合、真实数据和检查点读取均为零。新运行最多 2 次传播、640 次 Torch RHS、640 次切线 RHS、4 次 SVD，预期可见历史 Jacobian 秩为 0/9。使用原门槛和完整原名单，保留原失败文件与消耗记录。

## 回执与附件

- [machine_local_topology_closeout_review_20261115.json](_sealed/machine_local_topology_closeout_review_20261115.json)

<details>
<summary>原件与版本</summary>

- [docs/protocols/machine_local_topology_closeout_v1_zh.md](_sealed/machine_local_topology_closeout_v1_zh.md)

</details>
