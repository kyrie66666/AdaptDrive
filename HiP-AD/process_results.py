#!/usr/bin/env python3
"""
HiP-AD 成绩处理脚本——支持任意条路线、自动过滤异常数据。

用法：
  # 单文件
  python process_results.py -f evaluation/hipad_clean_base_gpu1/result.json

  # 多文件合并
  python process_results.py -f evaluation/hipad_clean_base_gpu*/result.json

  # 合并后输出到指定文件
  python process_results.py -f evaluation/hipad_clean_base_gpu*/result.json -o merged_result.json

  # 加 --keep-all 保留所有数据（不筛异常）
  python process_results.py -f evaluation/hipad_clean_base_gpu*/result.json --keep-all
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

# ============================================================
# 异常判定规则
# ============================================================

# 这些 status 前缀的 route 会被标记为异常并在默认模式下筛除
CRASH_PREFIXES = [
    "Failed - Simulation crashed",
    "Failed - Agent's sensors were invalid",
    "Failed - Agent timed out",
    "Failed - No traffic manager recorded",
]

# score_composed 为 0 且 route_length 极短的，通常是 CARLA 崩了
MIN_ROUTE_LENGTH_FOR_ZERO_SCORE = 1.0  # 米


def is_anomalous(record: Dict) -> Tuple[bool, str]:
    """判断一条 route record 是否异常。返回 (是否异常, 原因)。"""
    status = record.get("status", "")
    scores = record.get("scores", {})
    meta = record.get("meta", {})

    # 1. 按 status 前缀匹配
    for prefix in CRASH_PREFIXES:
        if status.startswith(prefix):
            return True, status

    # 2. 分数为 0 且路线长度极短（CARLA 崩了没跑）
    if scores.get("score_composed", 1) == 0.0 and meta.get("route_length", 0) < MIN_ROUTE_LENGTH_FOR_ZERO_SCORE:
        return True, "Zero score + short route (likely CARLA crash)"

    # 3. 路线长度明显异常（远小于均值，可能是卡住后崩溃）
    #    这个由外部传入的阈值判断

    return False, ""


# ============================================================
# 统计计算
# ============================================================

def compute_stats(records: List) -> Dict:
    """对清洗后的 records 计算成绩。"""
    n = len(records)
    if n == 0:
        return {"count": 0}

    composed = [r["scores"]["score_composed"] for r in records]
    route_scores = [r["scores"]["score_route"] for r in records]
    penalty_scores = [r["scores"]["score_penalty"] for r in records]

    def mean(vals):
        return sum(vals) / len(vals)

    def std(vals):
        m = mean(vals)
        return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5

    completed = sum(1 for r in records if r["status"] in ("Completed", "Perfect"))
    perfect = sum(1 for r in records if r["status"] == "Perfect")

    infraction_counts = defaultdict(int)
    for r in records:
        for k, v in r.get("infractions", {}).items():
            if v:
                infraction_counts[k] += 1

    return {
        "count": n,
        "completed": completed,
        "perfect": perfect,
        "score_composed_mean": round(mean(composed), 4),
        "score_composed_std": round(std(composed), 4),
        "score_route_mean": round(mean(route_scores), 4),
        "score_penalty_mean": round(mean(penalty_scores), 4),
        "infractions": dict(infraction_counts),
    }


# ============================================================
# 主逻辑
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="HiP-AD 成绩处理")
    parser.add_argument("-f", "--file-paths", nargs="+", required=True, help="result.json 文件路径（支持通配符）")
    parser.add_argument("-o", "--output", default=None, help="合并后输出路径（可选）")
    parser.add_argument("--keep-all", action="store_true", help="保留所有数据，不筛异常")
    parser.add_argument("--min-route-length", type=float, default=None,
                        help="筛除 route_length 低于此值（米）的记录")
    args = parser.parse_args()

    # ---- 读取所有文件 ----
    all_records = []
    sensors = None
    total_progress_1 = 0  # sum of progress[0] (completed count per file)
    total_progress_2 = 0  # sum of progress[1] (total count per file)
    for fp in args.file_paths:
        if not os.path.exists(fp):
            print(f"[WARN] 文件不存在，跳过: {fp}")
            continue
        data = json.load(open(fp))
        checkpoint = data.get("_checkpoint", {})
        records = checkpoint.get("records", [])
        all_records.extend(records)
        # 收集传感器信息
        if data.get("sensors") and sensors is None:
            sensors = data["sensors"]
        prog = checkpoint.get("progress", [0, 0])
        total_progress_1 += prog[0]
        total_progress_2 += prog[1]
        print(f"[LOAD] {fp}: {len(records)} records")

    if not all_records:
        print("[ERROR] 没有找到任何记录")
        sys.exit(1)

    # ---- 去重 ----
    seen = set()
    unique_records = []
    dup_count = 0
    for r in all_records:
        rid = r.get("route_id", "")
        if rid not in seen:
            seen.add(rid)
            unique_records.append(r)
        else:
            dup_count += 1
    if dup_count:
        print(f"[DEDUP] 去除 {dup_count} 条重复记录")
    records = unique_records

    # ---- 异常检测 ----
    anomalies = []
    clean = []
    for r in records:
        bad, reason = is_anomalous(r)
        if args.min_route_length is not None:
            rl = r.get("meta", {}).get("route_length", 0)
            if rl < args.min_route_length:
                bad, reason = True, f"route_length {rl:.1f}m < threshold {args.min_route_length:.1f}m"

        if bad and not args.keep_all:
            anomalies.append((r["route_id"], reason))
        else:
            clean.append(r)

    # ---- 输出摘要 ----
    print(f"\n{'='*60}")
    print(f"总记录数: {len(records)}")
    if not args.keep_all:
        print(f"异常筛除: {len(anomalies)} 条")
        for rid, reason in anomalies:
            print(f"  ✗ {rid}: {reason}")
        print(f"有效数据: {len(clean)} 条")
    else:
        print("(保留所有数据，未筛除)")
    print(f"{'='*60}")

    # ---- 计算成绩 ----
    stats = compute_stats(clean if not args.keep_all else records)
    print(f"\n路线总数:       {stats['count']}")
    print(f"完成数:         {stats['completed']}  (Perfect: {stats['perfect']})")
    print(f"综合得分 (DS):  {stats['score_composed_mean']:.4f} ± {stats['score_composed_std']:.4f}")
    print(f"路线完成率:     {stats['score_route_mean']:.4f}")
    print(f"违规惩罚系数:   {stats['score_penalty_mean']:.4f}")
    if stats.get("infractions"):
        print(f"\n违规统计:")
        for k, v in sorted(stats["infractions"].items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")

    # ---- 可选：写出合并文件 ----
    if args.output:
        output = {
            "_checkpoint": {
                "records": clean if not args.keep_all else records,
                "progress": [len(clean if not args.keep_all else records), total_progress_2],
                "global_record": {},
            },
            "entry_status": "Started",
            "eligible": True,
            "sensors": sensors or [],
            "values": [
                {"name": "DS", "val": stats["score_composed_mean"]},
                {"name": "RC", "val": stats["score_route_mean"]},
                {"name": "IP", "val": stats["score_penalty_mean"]},
            ],
            "labels": ["DS", "RC", "IP"],
        }
        json.dump(output, open(args.output, "w"), indent=2)
        print(f"\n[SAVE] 已写入: {args.output}")


if __name__ == "__main__":
    main()
