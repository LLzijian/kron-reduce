# P1-A03 阶段报告：候选获取完成，科学诊断尚未完成

日期：2026-08-31。

本轮解决了 A02 无法保存旧结果的问题，正式完成 `plan_registered` 和
`candidate_acquired`。旧模型仍是 `IDENTIFICATION_FAILED` 参照候选，没有新模型晋级、
合并或新的轨迹、模态、稳定证书结论。

## 实际完成的工作

- 原始结构结果按字节复制；原文件和副本 SHA256 一致。旧文件中的一处未定义稳定裕度
  `/stability/safety/cases/2/margin` 原样保留，明确不能用于科学结论。
- 单模型、无来源路由声明使用独立的严格 JSON 说明；没有把 NaN 改成零，也没有重算稳定标签。
- 核验旧结果、模型文件、方案、数据角色和新文件清单之间的引用；候选执行器在返回成功前
  必须完成核验，不能以“文件写出来了”代替本阶段通过。
- 回归包含真实旧文件以及重复键、冲突声明、改写未定义值、伪造说明、重新计算哈希后篡改、
  冻结协议改变和输入越界等攻击。它们验证文件获取与约束，不等于完整科学对抗审查。

## 检查结果

| 检查 | 结果与范围 |
|---|---|
| 全量测试 | 274 项通过；0 失败、0 错误、0 跳过 |
| harness audit | PASS；工作流历史连续 |
| data-check | PASS；canonical v1 仍为 315 个事件、32 个文件 |
| 修改源码和测试的 ruff、git diff --check | PASS |
| 新运行的独立合同核验 | PASS；仅证明已保存候选的元数据和引用一致 |
| 获取时 git-check | FAIL_DIRTY_WORKTREE；已如实登记，不能据此声称源码已冻结 |
| 模型单元门、受控实验、科学验收、合并 | 尚未执行 |

测试记录为 `scratch/one_off/full-a03-a634461d/pytest.xml`。没有把整个 harness 的额外
静态检查算作通过：`project_harness/git_state.py` 中两条既有问题不在此次修改范围。
本轮没有新增原始轨迹或原始标签读取；已保存的旧安全评估记录仅作为继承证据保存。

## 不可变运行和证据

方案登记：
`paper_ieee39_modular_energy_p1_a03_plan_registered_f1_seed324-plan_registered-20260831T130319Z-93d29d92`

工件清单 SHA256：
`ba953df15bb2e86d64d0729cc07bec832dd074c7f29a76c34cd9a6f435ffcaf9`

候选获取：
`paper_ieee39_modular_energy_p1_a03_candidate_acquired_f1_seed324-candidate_acquired-20260831T130327Z-93d29d92`

工件清单 SHA256：
`6864dfaf8805782b194d9d30fe3eda7c877cdae04a4465b3b1f85d691946daf3`

独立核验：
`paper/planning/ieee39_modular_energy_equivalent/qa_logs/p1_a03_candidate_acquisition_check.json`

核验记录 SHA256：
`caec4e3c09585bea7c15d57f8ac239bffd0db9b4219957822a769eb4cb15403b`

核验的 `scientific_verdict=null` 表示尚未执行科学验收，不是 `ACCEPT` 或回退裁定；
本文件不是用于推进 acceptance 的证据。`promotion_decision=KEEP_BASELINE`。
参照 run 本来就是失败候选，不把保留它写成“已有接受的部署模型”。

## 下一步：接通能够回答失败原因的真实实验

1. 独立实现只读取 F1 fit、selection、internal audit 的入口；不能直接调用会预先读取
   external/safety 轨迹的旧总执行函数。
2. 明确区分旧检查点回放和重新拟合的五组对照。对照组之间统一初始模型、拟合角色、
   事件覆盖与计算预算；平衡损失比较只改变权重。
3. 接通模块置零、整块删除后重拟合、共享极点与局部可见度统计；不能用参数非零代替
   预测贡献，也不能把稳定的机械功率子方程直接当成能量连接证明。
4. 实际路径测试通过后，才进入模型单元门、源码冻结和受控实验。

具体算法接口及审查要点保存于
`docs/protocols/ieee39_modular_energy_p1_five_arm_execution_design_zh.md`。
该文件是待登记的实现设计，不修改已经冻结的 A03 获取协议，不授权提前读取实验数据。

A03 当前停在 `candidate_acquired`；总工作流仍停在 `plan_frozen`。
P1 的三个科学问题尚未全部回答，完整审查尚未完成。
