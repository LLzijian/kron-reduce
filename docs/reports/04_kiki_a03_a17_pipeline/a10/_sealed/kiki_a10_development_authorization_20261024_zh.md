# A10：全体 train 的独立积分精化审计

用户已明确要求持续改进、保留全部试验并推送仓库；Astra Ultra 在 A09 结果复核后批准这一独立数值问题。此次登记沿用该执行授权，A04–A09 仍已消费，不重新拟合或改写任何旧结论。

- 身份：`kiki_a10_development_20261024`；实验 `paper_kiki_a10_train_refinement`，workflow `kiki_a10_train_refinement_v1`，stage `train_refinement`，module `kron_reduce.workflows.kiki_train_refinement`。
- 固定来源：A06 step240 的 seeds 324/325/326；比较源为 A09 已保存的全部 .005 s 路径。模型、清单和路径 SHA 依 Astra 设计逐项绑定。
- 数据：原分配 train 268 例、合格 258 例、未来总点数 72147；仅 train 的既定静态先验、t/delta/omega 和 Pm[10]，不调用 allocate_roles/truth_metadata，不解码标签、事件时间、内部/source val/test、未来 Pm、Pe、电压或隐藏目标。完整文件哈希字节读取另列。
- 新增预算：0 拟合、0 优化更新；唯一新步长 .0025 s，774 个事件×seed 路径、1548 次历史/未来 rollout、896724 次 RK4、3586896 次积分 RHS、225729 次新输出状态物理检查 RHS，总计 3812625。槽位唯一、失败消耗、不重试；不重算旧步长、不增加自动精化。
- 第一条新 RHS 前，逐位核对 A09 已保存真值数组和顺序、点数、原权重、全部覆盖；重建全部旧 .005 s 逐事件诊断及原总体/时长桶汇总，浮点误差≤1e-12，索引/布尔精确。Pm 和时钟以原文件/转换代码绑定，额外保存供后续审计，不声称与不存在的旧数组相等。
- 全774比较均须满足 .005/.0025 s 未来可见最大归一化差≤1e-5、规则判定一致和首触发差≤2点；新路径能量恒等式≤1e-8、质量加权 COI≤1e-10，原余量误差界通过。完整预算与来源/旧回放也须通过，方可判本数值审计 ACCEPT。
- 从全258事件重新计算科学支持条件：至少2事件、2条不同物理线路，三个 seed 在两个步长都无历史触发、真值未来 rho>1e-12、预测未来无触发，新细路径同一真值代表窗口各 C−pM>1e-12。所有全体门槛先通过才允许提出独立时间聚合对照，仍不自动授权训练。

完整失败、原不合格事件和 A09 的 REJECT 均保留。本审计不计算 FSR、不选择/晋升模型、不交付预测、不认证输入范围；固定 `new_training_authorized=False`、`selected_candidate=None`、KEEP_BASELINE。执行前和结果解释由 Astra Ultra 审核。
