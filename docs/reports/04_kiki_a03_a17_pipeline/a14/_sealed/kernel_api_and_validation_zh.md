# A14 数学模块：接口与合成验证

2026-10-24，Astra Ultra。模块位于 `integral_initializer_kernel.py`，SHA256 `5bc7ba32a9a6646dbb119810f0e67f90f11cff20e6deaf356a43424517c59182`。它只接收调用者给定的数值数组和固定 closure，不含数据读取、模型加载、拟合、RHS 或 rollout。真实执行仍须独立登记和整条实现审核；本记录不授予执行权限。

## 固定接口

`prepare_static(closure, velocity_scale, *, owner_key, ledger) -> StaticCache`：要求 n=10、r=2、sine，速度尺度为十维正数。`owner_key` 是模型/seed 身份字符串，不参与方程。返回冻结的模型、尺度与派生矩阵缓存，恰执行 PC 和 `[PC;.01PCA]` 两次 SVD。`cache.data` 是只读 mapping，保存 P/Q/A/PC/S、O、两个 SVD、投影矩阵和 incidence；`model_sha256` 绑定全部六个 closure 数值数组、backbone 与 velocity_scale，`payload_sha256` 绑定缓存全部内容。每次使用前重新检查 owner、模型数组指纹和缓存内容指纹，拒绝跨模型、跨 owner 或被修改缓存。

`build(q, v, time, closure, p, eq, *, cache, owner_key, ledger) -> ArrayRecord`：q/v 为 (11,10)，time 为 (11,)，p/eq 为 (10,)。time 减首值形成数学时钟，保留每段实际浮点间隔。输出沿用 A13 的 L/Ltilde、两套 g/b/btilde、SVD、zh/zTobs、force/dynamic/kinematic residual 等名字；额外返回 `candidate_zh`、静态绑定、逐区间平均力残差。缓存元数据与结果中的嵌套 mapping 应由保存器递归展开，所有数值数组独立保存；不要把整个 mapping 当 object 数组写入 NPZ。

成功事件恰执行 10 次 8×8 expm、1 次 L-SVD、60 次独立正弦力求值和 2 次复用 SVD 的 LS。无 A 逆、ridge、可替换投影入口、额外 sampling 或自选阈值。PC 的 rank 仅用于投影/诊断，不是接受门槛；PC rank=1 时动态 L 仍可能 rank=2。

L rank<2 时 `status='unresolved'`，矩阵、两个右端、力空间残差与运动学残差保留；`zh2/zh4/zTobs2/zTobs4/candidate_zh`、动态残差和隐藏能量为 None，LS 调用为 0。调用者必须把该槽两个候选 rollout 标成 not_attempted，V=false、I 未完成；不能传播最小范数解或回退 encoder。该情况的 10 expm、1 L-SVD、60 force 已消费，其余预定槽可以继续。

## 算子计数与失败语义

`Ledger(limits=None, event_sink=None, context='')` 可由工作流传入类别上限和持久日志回调。类别固定为 expm/svd/force/ls。`attempted` 在日志及调用之前登记，`dispatched` 在 attempted 日志成功后、真实 callable 前登记，`returned` 记录实际返回，`failed` 记录 callable 异常；三者在正常完成时相同。日志失败前已预留的 attempted 不能冒称实际 dispatched。

event_sink 在每个 attempted/returned/failed 转换被调用。日志失败抛 `OperatorJournalError`；预算拒绝抛 `OperatorBudgetError`；两者将账本标记 poisoned，后续算子均拒绝。工作流须整轮停止并保留已耗与未尝试槽，不把这类错误当局部数值问题继续。一般数值算子失败保留调用计数和错误，是否继续其他独立槽按冻结协议处理。模型/缓存身份错误属于来源完整性问题，应停止，不能换缓存重试。

## 已完成的合成正确性证据

6 项小型测试通过，ruff 通过，未重复 A13 的 15 病例运行。测试覆盖：A13 与 A14 的 L/g/b/初末态/残差逐项在绝对 1e−12 内一致；负起点、十个非均匀间隔原样使用；静态缓存复用与精确算子数；跨 owner/改变模型在新算子前拒绝；rank 不足无候选、无 LS；PC rank1 而动态 L rank2 的有效正例；日志前后失败及预算拒绝均不重试。

这次持久验证实际完成的纯合成算子总数为：60 expm、15 SVD、10 LS、360 独立 force。该数包括测试中一次 A13 参照调用及 A14 小型调用；用于日志失败检查的两个普通 Python callable 探针单列，不算数学力求值。真实数据读取、模型加载、RHS、rollout 均为 0。

验证目录：`kernel-validation-20261024T111429513988Z/`。`receipt.json` SHA256 为 `e805c9998e63358540303db7ca32fccc5fe8ecab0af01c2e4725fdd522c97650`，包含全部源码/日志 SHA，状态 PASS。`synthetic_operator_ledgers.json` 保存每条算子记录，SHA256 `a1bcbaa30eb8d1a2c60070709780dc2d2286a6076fcd5c90de0a53519ec87025`。

测试 `test_integral_initializer_kernel.py` SHA256 `19ae89f09433ae9ebaa881ff9d2864abc24051cf80269112f6663896fea6dd33`；持久验证脚本 `validate_kernel.py` SHA256 `f7562c2bdb11d0796d63c5f9008bd12e8e861e9f173a1a65db2b0f77a903f475`。冻结 A13 参照模块 SHA 保持 `e9a1060f8a06f57e8ca4669f5010f437e9ff4791073cafd5aad28be827d86e2a`，未修改其方案、运行或结果。

这证明 A14 数学模块正确复现冻结的估计方程及数值拒绝边界。它不证明真实 kiki 初态可恢复，也不证明闭环历史或未来轨迹会改善；这些必须由独立 A14 及后续明确授权的证据分别回答。
