import os
import copy
import json
import glob
import argparse
import numpy as np
from prettytable import PrettyTable

# 与 process_results.py 保持一致的异常判定：只排除 CARLA 侧崩溃，保留所有 agent 真实跑过的路线
CRASH_PREFIXES = [
    "Failed - Simulation crashed",
    "Failed - Agent's sensors were invalid",
    "Failed - Agent timed out",
    "Failed - No traffic manager recorded",
]
MIN_ROUTE_LENGTH_FOR_ZERO_SCORE = 1.0  # 米


def should_exclude(record):
    """判断一条 route record 是否因 CARLA 侧异常需要排除。"""
    status = record.get("status", "")
    for prefix in CRASH_PREFIXES:
        if status.startswith(prefix):
            return True
    # 分数为 0 且路线长度极短（CARLA 崩了没跑）
    scores = record.get("scores", {})
    meta = record.get("meta", {})
    if scores.get("score_composed", 1) == 0.0 and meta.get("route_length", 0) < MIN_ROUTE_LENGTH_FOR_ZERO_SCORE:
        return True
    return False


def is_success(record):
    success_flag = False
    if record['status'] in ['Completed', 'Perfect']:
        success_flag = True
        for k, v in record['infractions'].items():
            if len(v) > 0 and k != 'min_speed_infractions':
                success_flag = False
                break
    return success_flag

def draw_table(route_ids, scenario_names, driving_scores, success_routes):
    table = PrettyTable()
    table.field_names = ["index", "route_id", "scenario_names", "driving_score", "success"]
    for i in range(len(driving_scores)):
        table.add_row([i, route_ids[i], scenario_names[i], driving_scores[i], success_routes[i]])
    return table


def statistic_route_json(route_dir, remove_update=False):
    if remove_update:
        print("WARNING: it will remove and update the failed route file and record !!!!")

    route_paths = glob.glob(f'{route_dir}/*.json')
    route_paths.sort()

    route_ids = []
    town_names = []
    scenario_names = []

    logging_infos = []
    driving_scores = []
    success_routes = []

    total_completed_routes = 0

    for route_path in route_paths:
        if 'merged.json' in route_path:
            continue

        with open(route_path) as file:
            data = json.load(file)
            data_checkpoint = data['_checkpoint']

            records = data_checkpoint['records']
            progress = data_checkpoint['progress']
            global_record = data_checkpoint['global_record']

            # finish all clips
            if len(global_record):
                completed_routes = 0
                excluded_count = 0
                for record in records:
                    if should_exclude(record):
                        excluded_count += 1
                        continue
                    route_ids.append(record['route_id'].split("_")[1])
                    town_names.append(record['town_name'])
                    scenario_names.append(record['scenario_name'])
                    driving_scores.append(record['scores']['score_composed'])
                    if is_success(record):
                        completed_routes += 1
                        total_completed_routes += 1
                        success_routes.append(1)
                    else:
                        success_routes.append(0)

                logging_info = "loading {}, success:{}/{}, progress:{}/{} ".format(
                    os.path.basename(route_path), completed_routes, progress[1], progress[0], progress[1])
                logging_infos.append(logging_info)
            else:
                valid_records = []
                completed_routes = 0
                excluded_count = 0
                for record in records:
                    if not should_exclude(record):
                        route_ids.append(record['route_id'].split("_")[1])
                        town_names.append(record['town_name'])
                        scenario_names.append(record['scenario_name'])
                        valid_records.append(record)
                        driving_scores.append(record['scores']['score_composed'])
                        if is_success(record):
                            completed_routes += 1
                            total_completed_routes += 1
                            success_routes.append(1)
                        else:
                            success_routes.append(0)
                    else:
                        excluded_count += 1
                        if remove_update:
                            failed_paths = glob.glob(os.path.join(route_path, '*' + record.get('save_name', '')))
                            if len(failed_paths):
                                print("this record failed".format())
                                failed_path = failed_paths[0]
                                if os.path.exists(failed_path):
                                    if 'meta' in os.listdir(failed_path):
                                        os.system('rm -r {}'.format(failed_path))

                valid_progress = [len(valid_records), progress[1]]

                updated_checkpoint = {
                    'global_record': {},
                    'progress': valid_progress,
                    'records': valid_records,
                }

                update_data = copy.deepcopy(data)
                update_data['_checkpoint'] = updated_checkpoint

                if remove_update:
                    print("update json file: {}".format(route_path))
                    with open(os.path.join(route_path), 'w') as file:
                        json.dump(update_data, file, indent=4)

                logging_info ="loading {}, success:{}/{}, progress:{}/{}".format(
                    os.path.basename(route_path), completed_routes, valid_progress[0], valid_progress[0], valid_progress[1])
                logging_infos.append(logging_info)

    driving_score = np.average(driving_scores)
    success_score = total_completed_routes / (len(driving_scores) + 1e-5) * 100

    # print
    table = draw_table(route_ids, scenario_names, driving_scores, success_routes)
    print(table)
    print()

    for logging_info in logging_infos:
        print(logging_info)
    print()

    print("completed_routes:{}/{}, driving_score:{:.2f}, success_score:{:.2f}".format(
        total_completed_routes, len(driving_scores), driving_score, success_score))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="统计路线 success rate 和 driving score")
    parser.add_argument('-d', '--route-dir', type=str, required=True,
                        help='存放 route json 文件的目录')
    args = parser.parse_args()
    statistic_route_json(args.route_dir)
