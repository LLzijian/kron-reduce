# A1-S0 状态端口：准确实施审核

2026-11-11，Astra Ultra 科学审核。模型与推理配置依据协调者任务分派记录为 `gpt-6-astra`、`ultra`。

**准确实施审核通过。** 批准本次固定合成正确性检查在独立一次性范围登记、显式 `synthetic_correctness` 审批记录、干净提交已推送、适用 CI 和实际加载环境守卫通过后执行一次。用户“调整项目研究方案并开始自主研究”的明确指令提供操作授权；本收据给出准确科学范围与实现批准。研究结果须由冻结门槛和保存证据决定。

| 绑定对象 | SHA256 |
| --- | --- |
| [计划](../../configs/proposals/state_port_synthetic_v1.json) | `2f0919965b08f56678d3bc8d0ac1ed30d1e1e75971c77f4074e256578a6dc545` |
| [协议](../protocols/state_port_synthetic_v1_zh.md) | `9449b2e62a89c0ec2fb99c98d6a68fa6985bd7221e3ee4ec43ea2bd98bd2d526` |
| [模型](../../src/kron_reduce/models/state_port.py) | `099495d11945af25dee3f03780cf9c14692716fc38ab35c3d5d1b8f101e0f3f4` |
| [评估核心](../../src/kron_reduce/evaluation/state_port_correctness.py) | `4b9473fa54aba5dcb6101f8c43f3dbbdab85b7ecd43ce1371c7a7b02401dccc8` |
| [独立入口](../../src/kron_reduce/workflows/state_port_synthetic.py) | `bc8a7209d4db2989212e182d2e0bc7125f6b95df3fad78af0f503fbb982a303c` |

已读取最终协议、核心代码和守卫，逐字节核对计划的 **26 项实现绑定，全部匹配**。常值与状态端口均使用同一 18 参数映射 `s_C*tanh(Theta)`；状态端口两侧互易，质量 COI 投影与 Helmert 力基一致。能量函数不依赖端口，能量恒等式无需新增端口时间导数；完整状态与参数 Jacobian 保留 `C'(q)` 及参数映射的梯度。零增量基线、两后端、两步长、两档中心差分、短传播梯度与两个隔离负例承担不同检查职责，接受规则与协议相符。

固定调用树为 4 个范围、3 次 probe、162 条 rollout；NumPy/PyTorch RHS 分别为 **69,017 / 9,769**，5 次 Jacobian 对应 110 次向量反传，另有 2 次标量反传。614 次模型构造和 158 次公共 rollout 参数验证合计 2,316 次参数矩阵特征值检查。SVD、expm、LS、真实数组读取、拟合和优化更新均为零。预算覆盖预留、实际调用及返回；门槛失败先保存记录再停止，不重试或转移余额。保存的差分正负 RHS、路径、loss、梯度和能量原量足以进行约定的算术复核。

独立入口先核显式阶段审批，再核 active policy、准确计划及批准收据；Git common directory 中的一次性预留在科学核心执行前消费。实际加载源码路径、SHA 和固定依赖版本受守卫约束。原模型默认路径与已消费授权保持原状。此次审核未执行候选、RHS、梯度或传播，也未读取 NPZ。

工程测试依据协调者和独立工程代理的工具记录：两个测试文件 **12 passed（0.18 s）**，新增三个模块和两个测试文件 Ruff 通过；未另存本地测试日志。本审核读取了测试代码与其准确哈希，没有为补日志重复运行。提交后的 CI 是执行前必须通过并保存的独立工程证据。

运行完成后，允许按计划对本次 `inputs.npz`、`algebra.npz`、`jacobians.npz`、`paths.npz` 各解码一次，并核本次 JSON、完整工件哈希和已保存量的算术。该核验不得调用模型、RHS、积分器、autograd 或新科学算子，不能覆盖原运行失败。

通过的主张限于固定 A1 实现通过本范围的结构与梯度检查。真实数据收益、训练后的优势、误差根因、稳定性证书和模型升级均不由此成立；A1 train 需要独立设计、登记和准确实施审核。结论保持 `KEEP_BASELINE`。
