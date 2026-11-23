# 项目架构

仓库按职责分为五层。

| 层次 | 职责 |
| --- | --- |
| 数据源 | `data/kiki_ieee39_v1/` 登记 KIKI 真实数据，`KIKI_DATA_ROOT` 定位不可变的源文件。另有历史 canonical 数据和独立的合成真值。 |
| 数据约定 | 定义数据格式、表示方法、名单划分和因果输入。 |
| 模型 | 用方程或数据学习的闭合项实现统一的动力学接口。 |
| 试验 | 冻结所用的数据、划分、模型、随机种子和目标函数。 |
| 证据 | 保存逐样本结果、审计记录、哈希、生成报告和论文表格。 |

## 证据与报告目录

- `results/<immutable-run-id>/` 保存可由程序核验的试验证据、参数、指标、哈希、检查记录和本次运行生成的报告。运行目录不可变，获批的本地归档存放在 `results/backup/`。
- `docs/reports/` 保存纳入 Git 的研究报告和试验卡片。[报告索引](reports/README.md)将各主题关联到运行证据或归档，每项定量结论须标明证据来源。
- `paper/` 保存论文和投稿源文件，可链接到整理后的报告。运行证据由 `results/` 保管。

研究方案放在 `docs/research/`，执行协议放在 `docs/protocols/`，架构和[科学约束](scientific_contract.md)放在 `docs/`。临时或尚未验收的检查放在 `scratch/one_off/<run-id>/`。`project_harness audit` 会标记已停用的顶层 `experiments/` 和 `reports/` 路径。

`project_harness` 负责登记、Git 状态、运行管理和阶段审批。模型及评价实现在 `src/kron_reduce`，通过调用方接收配置，避免写入任务编号或结果目录常量。

## 工作流执行

工作流由命令按阶段推进。`configs/workflows/` 定义阶段、证据要求和各项判定对应的去向。`configs/project_manifest.json` 把试验绑定到实现和工作流，`configs/research_policy.json` 记录科学执行权限，`.project_state/` 记录本地进度。`loop plan` 读取这些记录，显示下一阶段和进入条件。

同一工作流的每次修改都持有同一把锁。运行器先检查登记阶段和审批，再校验源文件或启动试验。阶段推进须有有效证据。审计按顺序回放历史，逐次核对判定去向和在此前取得的审批。

目录布局借鉴 Snakemake 对工作流定义、配置和输出的分离方式；将已保存的运行与实际执行分开检查，则借鉴 DVC 实验跟踪和 Nextflow 运行日志。

[Snakemake 布局](https://snakemake.readthedocs.io/en/stable/snakefiles/deployment.html) · [DVC 实验跟踪](https://doc.dvc.org/user-guide/experiment-management) · [Nextflow 运行报告](https://docs.seqera.io/nextflow/reports/)

<a id="engineering-task-allocation" name="engineering-task-allocation"></a>

### 工程分工

协调者明确每项工作的文件范围、交付内容和验收要求。独立且范围清楚的工作交给能力足够、成本较低的子代理。可用时，由 Luna 承担小范围编辑、测试样例和只读检查。提供所需背景，可独立复核的工作并行执行。相互依赖的简单步骤由同一代理完成。

Astra Ultra 负责核心算法设计、科学验收和边界判断。协调者复核改动与证据，执行集成检查，负责 Git 集成和工作流推进。实际运行按登记的命令、审批和证据逐阶段推进。

## 科学实现

物理模型使用惯量中心（COI）坐标，统一采用保持 COI 约束的四阶 Runge–Kutta 积分（RK4）。M0 与图 SINDy M2A 用积分弱形式辨识方程，拟合目标取自轨迹积分。能量结构候选模型把边上的记忆状态纳入端口哈密顿方程，通过数值证书检查后方可使用相应名称。

历史结构选择流程由三个模块负责。`selection/clusterability.py` 判断一套数据能否做空间压缩，`models/prunable_energy_model.py` 管理显式分块动力学和物理证书，`workflows/data_level_structure_selection.py` 按 60/20/20 划分隔离拟合、选择与评价，保存不可变的运行产物。模型冻结后分组报告时直接使用已确定的模型。

自 2026 年 10 月 23 日起，新的真实数据试验保留十个发电机节点、暂停空间聚类，继续研究动力学阶数、隐状态和模块选择。`project_harness.policy` 在创建运行之前检查 KIKI 数据登记、空间分辨率、输入约定和已用 `test` 名单的限制。历史 canonical 数据复现须显式指定模式并匹配冻结哈希。运行器用工作流锁协调各进程，为每次尝试分配不同的 ID，并在失败时保留异常回溯，等待按登记流程处理。

GitHub CI 在不读取原始私有轨迹的情况下检查元数据、策略、因果性和合成物理性质。`data-check` 则单独核验冻结的本地元数据和外部源哈希，不读取留出轨迹数组。

每套轨迹数据部署一个最终模型。拟合和选择阶段可按激励工况提出候选模块，模型冻结后可按这些分组报告模块的可观测性。所有候选模块先联合拟合，再按整模块删除并重新拟合其余部分。数据来源、激励工况、故障身份和稳定性标签不参与模型输入或模型切换。数据支持不足时，缩小认证范围或移除不受支持的模块。

历史授权 `kiki_a04_development_20261023` 允许在原始来源 `train` 内按冻结的内部 `train`、`val` 收集认证前的开发证据。经验支持范围按因果输入估计，规则与 12 次拟合预算在执行前登记。支持情况的检查保留完整对照名单。授权绑定试验、输入约定、名单分配和协议哈希。部署按认证范围接受输入，模型晋升须有独立证据。

<a id="machine-local-cuda" name="machine-local-cuda"></a>

### 逐机模型的 CUDA 预测

在 Windows 且配有兼容的 NVIDIA 驱动时，请在单独的 Python 3.12 环境中安装固定版本的 CUDA 构建。参见 [PyTorch 官方 CUDA 13.0 wheel 索引](https://download.pytorch.org/whl/cu130/torch/)。

```powershell
py -3.12 -m venv scratch/.venv-cu130
./scratch/.venv-cu130/Scripts/python.exe -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
./scratch/.venv-cu130/Scripts/python.exe -m pip install -e '.[dev]' -c requirements/constraints-py312.txt
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
./scratch/.venv-cu130/Scripts/python.exe -m pytest tests/test_nonlinear_history.py -k cuda
```

将已加载的 `FrozenMachineLocalForecaster` 通过 `.to('cuda')` 移至 CUDA。`predict_batch` 返回位于模型设备上的张量，`predict` 返回 NumPy 数组。在 CPU 上也可以加载保存在 CUDA 上的检查点。对于新登记的配对实验，在精确的拟合或测试计划中设置 `"device": "cuda:0"`，并通过 `--experiment` 将已登记 ID 传给工作流入口。计划未指定 `device` 时，入口默认使用 CPU。

CUDA 预测会把十个 10 ms 输出帧组成一个可回放的 GPU 图。RHS 求解预算仍按阶段计数，隐状态求解预算仍按事件计数。CPU 和 CUDA 路径都使用 float64、投影 RK4 和三次 Gauss–Newton 更新。秩检查和最终轨迹检查在图外执行。

在 i5-13400F 和 RTX 4060 Ti 上对 64 个合成事件组成的一批数据计时，表中为预热后五次运行的中位耗时。训练包括以 5 ms 步长推演 3 s、反向传播和一次 Adam 更新。预测包括初始化和以 2.5 ms 步长推演 8 s。数据预先放在所选设备上，计时涵盖模型计算。

| 工作负载 | 优化前 CPU | 优化后 CPU | 优化后 CUDA |
|---|---:|---:|---:|
| 训练批次 | 8.03 s | 4.66 s | 9.52 s |
| 冻结预测批次 | 9.42 s | 5.74 s | 2.11 s |
