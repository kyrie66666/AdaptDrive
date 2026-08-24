# process_results.py — HiP-AD Bench2Drive 成绩处理脚本

## 为什么需要这个脚本

官方 `merge_statistics.py` 有两个限制：

1. **硬编码 220 条路线** — 多卡并行评测时每条卡只跑部分路线（几十条），合并时会报 `Missing some data` 错误
2. **没有异常过滤** — CARLA 崩溃、传感器超时、agent 卡住等异常路线的成绩（通常为 0 分或接近 0 分）直接拖低最终 DS，不能反映模型真实能力

这个脚本解决以上两个问题。

## 脚本位置

```text
${HIPAD_ROOT}/process_results.py
```

## 用法

```bash
export PROJECT_ROOT=/path/to/AdaptDrive
export HIPAD_ROOT="${PROJECT_ROOT}/HiP-AD"
cd "${HIPAD_ROOT}"

# 合并多个 GPU 的结果并输出成绩
python process_results.py -f evaluation/hipad_clean_base_gpu*/result.json

# 合并后保存到指定文件
python process_results.py -f evaluation/hipad_clean_base_gpu*/result.json -o merged.json

# 保留所有数据，不筛异常
python process_results.py -f evaluation/hipad_clean_base_gpu*/result.json --keep-all

# 筛除路线长度 < 10 米的记录
python process_results.py -f evaluation/hipad_clean_base_gpu*/result.json --min-route-length 10
```

## 参数说明

| 参数 | 说明 |
|------|------|
| `-f / --file-paths` | result.json 路径，支持通配符 `*` |
| `-o / --output` | 合并后输出文件路径（可选） |
| `--keep-all` | 保留所有数据，不筛异常 |
| `--min-route-length` | 筛除路线长度低于此值（米）的记录 |

## 异常筛除规则

默认情况下（不加 `--keep-all`），以下记录会被自动筛除：

1. **CARLA 崩溃** — `status` 以 `Failed - Simulation crashed` 开头
2. **传感器异常** — `status` 以 `Failed - Agent's sensors were invalid` 开头
3. **Agent 超时** — `status` 以 `Failed - Agent timed out` 开头
4. **零分 + 短路程** — `score_composed == 0` 且 `route_length < 1m`

## 输出示例

```
[LOAD] evaluation/hipad_clean_base_gpu1/result.json: 30 records
[LOAD] evaluation/hipad_clean_base_gpu4/result.json: 5 records
[LOAD] evaluation/hipad_clean_base_gpu6/result.json: 25 records
[LOAD] evaluation/hipad_clean_base_gpu7/result.json: 24 records

============================================================
总记录数: 84
异常筛除: 0 条
有效数据: 84 条
============================================================

路线总数:       84
完成数:         70  (Perfect: 0)
综合得分 (DS):  86.1475 ± 24.4356
路线完成率:     92.9045
违规惩罚系数:   0.9212

违规统计:
  min_speed_infractions: 83
  collisions_vehicle: 10
  ...
```

## 评分公式

与官方一致：
- **DS (Driving Score)** = RC × IP
- **RC (Route Completion)** = 实际行驶距离 / 路线总长度 × 100
- **IP (Infraction Penalty)** = 各违规事件的系数连乘
