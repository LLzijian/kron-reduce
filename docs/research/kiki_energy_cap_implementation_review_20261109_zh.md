# encoder 参照能量球迁移：准确实施审核

2026-11-09，Astra Ultra（任务分派配置 `gpt-6-astra`、`ultra`）。

**准确实施审核通过。** 依用户持续推进的授权，批准新范围 `kiki_energy_cap_transfer_v1_20261109` 在独立登记、干净提交已推送、对应 CI 和实际运行环境守卫通过后，执行唯一一次固定 A06 模型的 train 机制比较。范围为三个 seed、258 个合格事件、774 槽，每槽保留 E/I/C 的两档结果；268 个原分配事件的覆盖记录全部保留。批准不预设科学结果。

| 对象 | SHA256 |
| --- | --- |
| [机器计划](../../configs/proposals/kiki_energy_cap_transfer_v1.json) | `3ca756b4be5c6c73a26864258a7554a34c27b87cd3f03bc75e0389c284a63bf6` |
| [冻结协议](../protocols/kiki_energy_cap_transfer_v1_zh.md) | `0e0cca43e50dbb9a6b03c9311b84c7c413ebe604bfba3d8ab06c9e76a1272219` |
| [来源清单](../../configs/proposals/kiki_energy_cap_saved_sources_v1.json) | `d0dcdfdcebf768ae29d2f1b7e87d40e773f1fa5ad29095b63f1f1ac1f80a3ed1` |
| `scratch/one_off/energy-cap-registration/engineering-tests.txt` | `8c947719d684e6c0ec506c43acef6ca248b6c4ed951bbf2fc6788b34cb2be891` |

已逐项读取并核对计划的 **40 项实现绑定**，全部匹配。最终工程记录为 **70 passed**，Ruff 通过；覆盖人工数组求解适配、评分与裁定、白名单读取、因果冻结、失败保存以及临时 Git 一次性执行门禁。最终修订补齐 NPY 头部和载荷读取失败时的实际字节计数，以及完成时逻辑字节与头部上限断言；不改变候选、容差、判据或预算。

实现保留唯一因果半径 `rho=||z_encoder,h||`，直接使用同槽 A14 的 Ltilde/btilde4/SVD/zh4。冻结球求解器执行零半径优先、正半径秩门、SVD 一致性、KKT 和有限求根检查；非活跃分支精确复用 I。两档历史各自闭环、各自保留隐藏末态，只把可见状态重置为同一获准观测。全部 774 槽初始化与历史冻结后才读取未来时钟，全部候选未来或非活跃不可变引用冻结后才解码 E/I/truth 未来评分载荷。半径与预测不读取未来目标。

读取范围为清单内已保存的 train 证据：8,528 文件、21,699 成员、418,279,836 原始 bytes、239,827,112 数组逻辑 bytes；每文件、每成员一次。三阶段文件数为 5,430/2/3,096。NPY 头部与格式前缀每成员最多 4,108 bytes，先核 shape/dtype/大小，再读取声明载荷。scope 中 `source_payload_reads=0` 保留原 provider 数据禁读含义；本收据明确批准上述独立清单内的保存载荷，不表示本轮没有真实保存数据读取。

令 A 为 active 或零半径的新传播槽数、S_A 为其未来点数之和、R 为正乘子活跃槽数。新路径 `4A`，积分 RHS `240A+24S_A`，物理 RHS `24A+2S_A`，合计≤5,831,802；scope 为 `3,873+2A`、最多 5,421，probe 为 37,199。根更新/评估≤64R/80R。内部 closure 验证 `3+4A`、eigvalsh `9+12A` 单独列明；新 SVD、expm、通用无约束 LS、Gauss 力求积、encoder 与拟合均为零。实际分支计数须与保存账本一致。

公共一次性预算在科学读取或调用前消费，逐 scope 先持久化预留，再记 attempted/returned。数值、物理、秩、预算、日志或来源错误停止全轮，保留已返回数组、最后 RHS、求解诊断和未运行槽，不重试、不缩步补救。run 保存计划、来源清单、模型六参数及必要结果；旧 E/I/truth 使用准确不可变引用。允许两次分账的独立保存证据审核，各自至多读取同一来源文件/成员一次及新 run manifest 所列产物一次，禁止新求解、模型构造、RHS、SVD、expm、拟合或传播。

科学接受仍要求 `V and mechanism_exercised and G_E and D`：六个 seed/步长组相对原 encoder 达到预定 1% 主指标收益、超过粗细差异代理，并通过六项保护和规则保护。G_I 完整独立报告，优于已拒绝参照不能代替 G_E。任何结果均 `KEEP_BASELINE`；不自动授权训练、val/test、模型晋升、证书或结构扩展。所有来源研究的 consumed 授权保持关闭。

本审核仅读代码、Markdown、JSON 和工程日志并核验文件哈希，没有打开来源 NPZ，也没有调用模型、球求解器、RHS 或路径传播。运行批准由[准确 JSON 收据](kiki_energy_cap_implementation_review_20261109.json)与新独立授权共同绑定；执行前条件不能由本文审核结论替代。
