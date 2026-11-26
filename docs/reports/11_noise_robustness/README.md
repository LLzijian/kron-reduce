# 量测噪声与坏数据压力测试

本轮考察冻结 V4 和 ES-ROM（M3b_G2）在 100 ms 量测窗口受污染时的 8 s 预测表现。鲁棒移动窗估计联合恢复十机惯量中心相对状态和记忆初值，保留 ES-ROM 的冻结动力学。

[执行协议](protocol.md) · [试验配置](../../../configs/experiments/noise_robustness_esrom_v1.toml) · [门禁配置](../../../configs/workflows/noise_robustness_esrom_v1.json)
