# P1-A02 执行报告

日期：2026-08-31。结论：本轮候选获取失败，保留结果，进入下一轮重新设计；没有合并或模型晋级。

## 已完成

- 收紧工件哈希、工作流历史、报告引用和稳定证书命名检查。
- 零秩记忆端口不再生成无效隐状态；模态先检查能量和输出可见度，再报告形状。
- 模块贡献须区分置零效应与删除后重拟合，不能由逐个可替代推断整体等价于 B2。
- 五臂必须显式绑定数据角色及模型文件；损失权重和梯度同时检查完整性、向量求和和来源失衡。
- 最终全量测试 **255 项通过**；项目审计、canonical 315 事件校验及修改文件的静态检查通过。

测试记录：`scratch/one_off/full-7236bd54/pytest.xml`。整个 harness 的额外静态检查仍有
两条既有问题，位于本轮未修改的 `project_harness/git_state.py`；未将其算作通过。

## 实际执行及失败原因

方案登记成功：
`paper_ieee39_modular_energy_p1_a02_plan_registered_f1_seed324-plan_registered-20260831T023343Z-686ab456`。

候选获取失败：
`paper_ieee39_modular_energy_p1_a02_candidate_acquired_f1_seed324-candidate_acquired-20260831T023359Z-686ab456`。

旧参照运行的 `structure_selection_result.json:3462` 含有 `margin: NaN`。它表示该值未定义，
不是有效的稳定裕度。A02 试图把旧结构内容整体放入严格 JSON 的新说明文件，因此写入被拒绝。
单元测试没有覆盖这项旧文件特征；测试通过不能代替实际候选获取成功。

失败目录中的七个已生成文件原样保留，包括 `RUN_FAILED.json`；该目录没有完整工件清单，
不能作为候选通过的依据。旧数据、旧结构文件、数值门槛均未修改。

## 当前状态与下一步

`process_status=FAIL`，`scientific_verdict=RETRY`，`promotion_decision=KEEP_BASELINE`。
A02 状态停在 `plan_registered`，总工作流停在 `plan_frozen`；未进入模型单元门、源码冻结门或合并门。
当前参照模型本来就是失败候选，不把“保留参照工件”写成“已接受的部署模型”。

下一轮 A03：旧结构文件逐字节复制并绑定哈希，另写严格 JSON 的单模型说明；未定义数值保留在
旧证据中并明确不支持主张，不能改成零、静默删除或开启宽松写入。先加入真实旧文件回归，再做
新的方案登记与候选获取。

之后仍需接通五臂受控执行、模块反事实和完整单元门，并补齐总工作流对子流程接受记录、工件哈希、
真实合并提交的检查。本报告没有新增轨迹、振荡频率或稳定证书结论。

完整机器记录：`paper/planning/ieee39_modular_energy_equivalent/qa_logs/p1_a02_candidate_acquisition_retry.json`。
