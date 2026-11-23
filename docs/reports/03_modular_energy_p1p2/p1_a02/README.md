# P1-A02 执行报告

[返回主题页](../README.md)

## 运行结果与归档

- [登记与候选获取归档](../../../../results/backup/02_afsd_structure_discovery/L_ieee39_modular_energy_p1.DESCRIPTION.md)

登记文件与配置

[方案配置](../../../../configs/experiments/paper_ieee39_modular_energy_p1_a02_f1_seed324.toml)

## 执行协议

<a id="doc-fad7f89d1f6f"></a>

<a id="doc-fad7f89d1f6f-ieee39-模块化能量-p1-a02-冻结协议"></a>

### IEEE39 模块化能量 P1 A02 冻结协议

版本：`ieee39_modular_energy_p1_a02_v1`。A02 是 A01 在 `model_unit_gate`
得到 `RETRY` 后的新 attempt；A01 的运行、源码快照和负结论保持不变。


<a id="doc-fad7f89d1f6f-不变部分"></a>

#### 不变部分

- 分支为 `codex/loop-p1-a02`，数据仍为 canonical v1 的 F1，seed 324；test 与
  F2--F5 继续锁定。
- 每批输入只形成一个联合模型，不允许来源或激励路由。
- 五个受控臂、1/5/10 s 视界、2000 次线路 bootstrap、seed 324 和所有数值阈值
  均与 A01 相同。
- `Pm_clear` ZOH 是主口径；observed `Pm(t)` 只作条件上界，不能拟合主模型、进入
  主表或触发晋级。


<a id="doc-fad7f89d1f6f-a02-新增的硬门"></a>

#### A02 新增的硬门

1. 工作流和历史审计共用同一证据验证器；显式失败状态、空值、伪造前驱和不存在的
   运行、报告、哈希或提交均不得推进。
2. 工件清单必须与目录普通文件集合完全一致；缺失、改变、冻结后新增、符号链接和
   路径逃逸均失败。
3. P1 单元门交叉核验 candidate、baseline、结构结果、validation、lineage 和模型
   manifest，显式证明 `model_count=1`、`source_routing=false`、无路由键、无 test、
   无 F2--F5、无未来 `Pm`。P1 必须记录 `level_6_claimed=false`。
4. Level 6 受保护名称不能相信自报 `status`；必须绑定模型、合成数据、IEEE39 边界
   数据、独立组审计及其 SHA256，并由代码重算全部门。A02 不申请 Level 6 名称。
5. 零秩端口使用最小状态实现；零耦合隐状态不能进入可报告模态。模态形状只在速度
   输出可见度通过后归一化。
6. 删除效应在 2% 边界采用无重叠判定：删除 CI 上界等于 2% 可替代，下界等于 2%
   仍未解决，下界严格大于 2% 才能独立支持。逐块可替代不能推出联合退化为 B2；
   B2 结论必须由联合置零和联合删除重拟合反事实直接证明。
7. 模态裁定必须同时通过平衡残差、端口秩、速度可见度、有限窗能量、独立线路数、
   固定 bootstrap、频率、阻尼和 MAC。少于三条独立线路固定为 `UNDETERMINED`。
8. 五臂按名称、顺序、唯一比较基准和共同合同哈希锁定。第 4 臂复用第 3 臂模型，
   只替换 rollout 的 observed `Pm`；第 5 臂只新增因果闭包。
9. 数据角色哈希必须显式提供，不生成能通过审核的默认占位哈希。最终损失表必须
   覆盖全部预登记来源、线路、事件、视界与输出，并逐层等权。梯度按参数组保存
   向量；各项向量相加应与独立计算的总梯度一致，不能用绝对值相加替代。记录
   平方范数贡献份额，并同时检查单项及来源、视界、输出汇总的 80% 支配警报；
   该警报用于损失诊断，不替代既定轨迹验收门。
10. 非零效应不能替代可辨识性检验；任一可辨识性门失败的非零模块不能被归为
    独立支持。参数非零不自动形成预测贡献主张。


<a id="doc-fad7f89d1f6f-实现和验收边界"></a>

#### 实现和验收边界

目前注册执行器仅支持 `plan_registered` 和 `candidate_acquired`。候选元数据
核验只证明保存工件之间的一致性，不证明旧基线可以从干净提交重新生成，也不
证明五臂拟合、模块置零/删除重拟合和模态统计执行器完整。`model_unit_gate`
必须等这些执行路径及其测试实际接通后才能运行，不能以辅助函数测试数量替代。
总工作流的后续阶段也必须补齐子流程接受记录、工件哈希及真实合并提交的交叉
校验后才能推进；当前总工作流继续停在 `plan_frozen`。


<a id="doc-fad7f89d1f6f-loop-路由"></a>

#### Loop 路由

执行顺序仍为 `plan_registered -> candidate_acquired -> model_unit_gate ->
source_commit -> controlled_experiment -> residual_diagnosis -> adversarial_review ->
acceptance`。任一硬门失败时不推进状态机，保存新的负证据并再开 attempt。P1 的
`ACCEPT` 只表示诊断被接受，`promotion_decision` 始终为 `KEEP_BASELINE`。

## 报告与结果

<a id="doc-847a90693c10"></a>

<a id="doc-847a90693c10-p1-a02-执行报告"></a>

### P1-A02 执行报告

日期：2026-08-31。结论：本轮候选获取失败，保留结果，进入下一轮重新设计；没有合并或模型晋级。


<a id="doc-847a90693c10-已完成"></a>

#### 已完成

- 收紧工件哈希、工作流历史、报告引用和稳定证书命名检查。
- 零秩记忆端口不再生成无效隐状态；模态先检查能量和输出可见度，再报告形状。
- 模块贡献须区分置零效应与删除后重拟合，不能由逐个可替代推断整体等价于 B2。
- 五臂必须显式绑定数据角色及模型文件；损失权重和梯度同时检查完整性、向量求和和来源失衡。
- 最终全量测试 **255 项通过**；项目审计、canonical 315 事件校验及修改文件的静态检查通过。

测试记录：`scratch/one_off/full-7236bd54/pytest.xml`。整个 harness 的额外静态检查仍有
两条既有问题，位于本轮未修改的 `project_harness/git_state.py`；未将其算作通过。


<a id="doc-847a90693c10-实际执行及失败原因"></a>

#### 实际执行及失败原因

方案登记成功：
`paper_ieee39_modular_energy_p1_a02_plan_registered_f1_seed324-plan_registered-20260831T023343Z-686ab456`。

候选获取失败：
`paper_ieee39_modular_energy_p1_a02_candidate_acquired_f1_seed324-candidate_acquired-20260831T023359Z-686ab456`。

旧参照运行的 `structure_selection_result.json:3462` 含有 `margin: NaN`。它表示该值未定义，
不是有效的稳定裕度。A02 试图把旧结构内容整体放入严格 JSON 的新说明文件，因此写入被拒绝。
单元测试没有覆盖这项旧文件特征；测试通过不能代替实际候选获取成功。

失败目录中的七个已生成文件原样保留，包括 `RUN_FAILED.json`；该目录没有完整工件清单，
不能作为候选通过的依据。旧数据、旧结构文件、数值门槛均未修改。


<a id="doc-847a90693c10-当前状态与下一步"></a>

#### 当前状态与下一步

`process_status=FAIL`，`scientific_verdict=RETRY`，`promotion_decision=KEEP_BASELINE`。
A02 状态停在 `plan_registered`，总工作流停在 `plan_frozen`；未进入模型单元门、源码冻结门或合并门。
当前参照模型本来就是失败候选，不把“保留参照工件”写成“已接受的部署模型”。

下一轮 A03：旧结构文件逐字节复制并绑定哈希，另写严格 JSON 的单模型说明；未定义数值保留在
旧证据中并明确不支持主张，不能改成零、静默删除或开启宽松写入。先加入真实旧文件回归，再做
新的方案登记与候选获取。

之后仍需接通五臂受控执行、模块反事实和完整单元门，并补齐总工作流对子流程接受记录、工件哈希、
真实合并提交的检查。本报告没有新增轨迹、振荡频率或稳定证书结论。

完整机器记录：`paper/planning/ieee39_modular_energy_equivalent/qa_logs/p1_a02_candidate_acquisition_retry.json`。

<details>
<summary>原件与版本</summary>

- [docs/protocols/ieee39_modular_energy_p1_a02_v1.md](_sealed/ieee39_modular_energy_p1_a02_v1.md)
- [docs/reports/03_modular_energy_p1p2/ieee39_modular_energy_p1_a02_progress_zh.md](_sealed/ieee39_modular_energy_p1_a02_progress_zh.md)

</details>
