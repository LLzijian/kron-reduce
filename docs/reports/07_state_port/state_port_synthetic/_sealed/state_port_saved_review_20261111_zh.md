# A1-S0 失败运行的独立保存证据审核

2026-11-11。本协议只审核已生成的证据，不执行新的科学计算。

唯一来源为运行 `state_port_synthetic_v1-synthetic_correctness-20261110T161605071933Z-992738b0-967f4fb0`，源提交 `992738b0c4bfbf26e073f40a29f67cea4c4ac429`。其四个范围已关闭，数值记录已写出；随后写汇总 JSON 时遇到 NumPy 布尔值序列化错误，运行保持失败。原授权消费关闭，失败日志、不完整的 90 字节 summary 和所有原文件保持原样。

新授权 `state_port_saved_review_v1_20261111` 仅允许一次保存证据审核。机器计划绑定审核脚本、此协议和外置来源清单；准确审核收据完成后方可执行。入口先核授权与哈希，再在共享 Git 目录预留一次性预算，不重试、不覆盖已有审核。解码范围严格为本运行的 `inputs.npz`、`algebra.npz`、`jacobians.npz`、`paths.npz`，各一次、`allow_pickle=False`；其余 NPZ 只核文件哈希。可读取本运行 JSON、日志及冻结计划，校验清单中全部 22 个文件的哈希。不解析损坏的 summary，不要求不存在的成功 stage evidence。

审核复算保存量的有限差分、loss、归一化误差、矩阵投影关系和阈值，检查记录数量、形状、有限性与索引完整性。核验 15 个点态记录、4 组正确 Jacobian、16 条长路径、2 条名义短路径、144 条差分短路径；两个人工错误须被原定阈值检出。核对原失败记录的 attempted/returned、四范围日志的 165 对操作及日志区间哈希。不得调用模型、RHS、积分、求导、SVD、expm、LS、拟合或新科学算子。

审核报告只写入 `reports/generated/state_port_synthetic_20261111/`。原工作流不得因审核而前进，原运行也不得回填成功汇总或改为成功。报告分开记录 `process_status=FAIL_PRESERVED` 和保存证据的科学检查结论；只有全部算术与完整性检查通过，后者才可记为 `PASS_FROM_SAVED_EVIDENCE`。该结果至多支持另行准备真实 train 机制比较，保持 `KEEP_BASELINE`。
