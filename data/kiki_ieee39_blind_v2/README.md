# KIKI IEEE 39 节点故障后数据 `kiki_ieee39_blind_v2`

本版本包含 94 个 ANDES 2.0.0 原阶仿真事件，其中 62 个稳定、32 个失稳，覆盖 21 个故障组。所有事件位于同一运行点，采用三相总线故障和恢复后的拓扑。M5 已用这批数据完成独立盲测，后续任务可将它作为**已评估的冻结对照集**，用于结果复核、物理分析和冻结模型比较。仓库内的新科学比较沿用项目的登记与预算规则。[封存回执](../../docs/reports/postfault_structure/blind_dataset_build_m5_v1_20261121_evidence.json) · [M5 结果](../../docs/reports/postfault_structure/blind_confirmation_m5_v1_20261121_zh.md)

## 找到并校验数据

在本仓库，数据目录是 `data/kiki_ieee39_blind_v2`。原始 `dataset.h5` 为 439,887,050 字节，保存在本机工作区。Git 提供下表中的索引、清单、审计和说明文件。跨工作区使用时，把 `--dataset-dir` 指向持有同一批封存文件的目录。

| 文件 | 用途 |
|---|---|
| `dataset.h5` | 原阶仿真轨迹、惯量和平衡点。 |
| [dataset_index.csv](dataset_index.csv) | 每事件一行。包含真值、有效帧数、时间和故障条件。 |
| [splits.json](splits.json) | `blind` 名单，也是固定读取顺序。 |
| [manifest.json](manifest.json) · [checksums.sha256](checksums.sha256) | 版本、来源指纹与封存文件校验值。 |
| [qa_report.md](qa_report.md) · [qa.json](qa.json) · [disjointness_audit.json](disjointness_audit.json) | 录取与去重口径。状态去重使用清除后约 0.1 ms 的最早归档帧。 |
| [schema.json](schema.json) | 可供程序读取的数据字典，列出 CSV 全部字段、HDF5 路径、形状、单位和缺失约定。 |

在仓库根目录运行。第一条在只有 Git 文件的环境也能用，检查元数据和名单。第二条还校验本地 HDF5 的 SHA256、数组形状和类型。两条均不解码轨迹数组。

```powershell
.\.venv\Scripts\python.exe -B scripts/check_kiki_ieee39_blind_v2.py
.\.venv\Scripts\python.exe -B scripts/check_kiki_ieee39_blind_v2.py --verify-hdf5
```

外部任务可用自己的 Python 3.12 调用同一脚本，并传入 `--dataset-dir`。HDF5 校验和下面的示例使用 `h5py`，示例另外使用 `numpy`，本仓库版本见 [Python 依赖约束](../../requirements/constraints-py312.txt)。报告引用请记录数据集 ID 与封存清单 SHA256 `3ae5a16d70547b6aa7ffd2022f101530a34b89424d3ff329acc6d3597091d99f`。这份说明和数据字典属于使用资料，原封存清单继续校验同一组 7 个文件。

## 连接索引与轨迹

- 按 `splits.json["blind"]` 遍历，使用 `event_id` 连接索引与 HDF5 的 `/events/<event_id>`。跨数据集汇总时用 `kiki_ieee39_blind_v2/<event_id>` 做完整 ID。编号因候选录取而可能跳号。
- 用 `(fault_bus, fault_line)` 做故障分组。同组可以有多个切除时长。线路名和 `fault_location` 共同标识故障母线的线路端点。
- 全部数组为 float64。事件数组的第一维为有效帧数 `T`。机组列顺序是 `GENROU_1` 至 `GENROU_10`，母线列按 1 至 39 排列。稳定样本有 1,501 帧，失稳样本有 16 至 130 帧。跨事件拼接时保留各自有效长度。
- `t` 以最早归档采样点为零，间隔 0.01 秒。距清除的时间等于 `record_start_delay_s + t`。`event_time_s` 以记录起点为零，`los_s_from_clearing` 以清除为零。两项在稳定样本的 CSV 中均为空，读取为 `None` 或 `null`。

| HDF5 路径 | 形状 | 口径 |
|---|---|---|
| `/events/<event_id>/t` | `(T,)` | 秒。 |
| `/events/<event_id>/delta` · `omega` | `(T, 10)` | 功角 rad。转速是绝对 pu，额定值 1 pu。 |
| `/events/<event_id>/Pm` · `Pe` · `Qe` · `te` | `(T, 10)` | 100 MVA 系统基准的归档机械输入、有功、无功和电磁转矩。 |
| `/events/<event_id>/terminal_v` · `bus_v` · `bus_a` | `(T, 10)` 或 `(T, 39)` | 端电压和母线电压幅值为 pu。母线相角为 rad。 |
| `/events/<event_id>/full_state` · `algebraic_state` | `(T, 220)` · `(T, 479)` | 列含义分别见 HDF5 根属性 `state_mapping`、`algebraic_names`，两者都是 JSON 字符串。 |
| `/inertia` · `/equilibrium/delta` · `/equilibrium/Pm` | 各 `(10,)` | 共享的系统基准惯量系数、平衡点功角和机械输入。额定角速度见根属性 `omega_s`。 |

失步真值要求功角跨度大于 `π rad` 且最大质心相对转速幅值大于 `0.02 pu`，连续 5 帧满足。失稳记录止于确认帧。稳定记录保留 15 秒并通过原仿真器的尾段稳定判据。62/32 是生成配额，适合报告分组性能。[评估协议](../../docs/protocols/blind_confirmation_m5_v1_zh.md)

## Python 读取示例

下面的函数给出一个事件在 `measured_history_100ms_zoh_v1` 下的模型输入。先用完整校验确认本机文件，再按固定名单取事件。观测是索引 0–10 的 11 帧。起报机械输入只取索引 10，预测时使用固定 `p0`。

```python
import json
from pathlib import Path

import h5py
import numpy as np


def causal_input(store, event_id):
    event = store["events"][event_id]
    omega_s = float(store.attrs["omega_s"])
    mass = store["inertia"][...] / omega_s
    delta_star = store["equilibrium/delta"][...]
    pm_star = store["equilibrium/Pm"][...]

    def coi(value):
        return value - np.sum(value * mass, axis=-1, keepdims=True) / mass.sum()

    q = coi(coi(event["delta"][:11] - delta_star))
    v = coi((event["omega"][:11] - 1.0) * omega_s)
    delta_pm = event["Pm"][10] - pm_star
    p0 = delta_pm - delta_pm.sum() * mass / mass.sum()
    return {"q": q, "v": v, "p0": p0, "equilibrium": coi(delta_star), "mass": mass}


directory = Path("data/kiki_ieee39_blind_v2")  # 外部任务填校验通过的目录
event_ids = json.loads((directory / "splits.json").read_bytes())["blind"]
with h5py.File(directory / "dataset.h5", "r") as store:
    inputs = causal_input(store, event_ids[0])
```

TASK3 评分从索引 10 的起报点开始对齐。后续真值取索引 11 到 `min(T, 811) - 1`。稳定事件覆盖起报后 8 秒，失稳事件按实际有效窗计分。原始标签、失步时间、故障组、未来数据、`Pe` 和电压属于评分或诊断侧。复用已有盲测结果时，可以直接读取 [M5 报告和预测回执](../../docs/reports/postfault_structure/blind_confirmation_m5_v1_20261121_zh.md)。
