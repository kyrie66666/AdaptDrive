# Roach 静态 Town 地图工具

本目录完整迁移 `carla-roach` 的静态地图生成能力，生成原项目定义的十层资产：

```text
road
shoulder
parking
sidewalk
stopline
lane_marking_yellow_broken
lane_marking_yellow_solid
lane_marking_white_broken
lane_marking_white_solid
lane_marking_all
```

其中 `road`、`lane_marking_all`、`lane_marking_white_broken` 是当前 Roach BirdView 真值运行时直接
读取的层；其余层仍按原始资产契约保留，避免地图生成器变成只适用于当前一个 head 的裁剪版本。

工具不依赖 Roach 的 RL 训练环境，不修改外部 `carla-roach`，也不把地图资产写入 Git。默认行为和原
项目一样，由生成命令自行启动 CARLA；区别是只终止本命令创建的 CARLA 进程组，不执行原脚本的全局
`killall`。`--connect-existing` 仅用于调试或复用用户明确指定的专用 server。

正常规模 Town 使用与原 Roach 一致的 `global` raster 路径。Town01/Town06 实测表明内存估计是保守上界；
每个 Town 仍会在分配 surface 前检查显式内存预算。Town11/Town12/Town13 的完整 global raster 已确认
超过 100 GiB gate，因此使用 `tiled` 生成路径：按 tile 逐块 rasterize/write，但 HDF5 中仍暴露与 global
asset 相同的十个全局 shape layer dataset，runtime crop 读取语义不变。

## 1. 创建独立环境

由用户执行：

```bash
cd "$PROJECT_ROOT"
conda create -n roach-map python=3.8 pip
conda activate roach-map
python -m pip install -r Bench2Drive/tools/roach_map/requirements.txt
```

不要安装完整 `carla-roach/environment.yml`，也不要执行 `pip install carla`。使用当前 CARLA server
自带的 Python API：

```bash
export PROJECT_ROOT=/path/to/I2R-AD
export CARLA_ROOT=/path/to/carla-0.9.15
export ROACH_BEV_MAP_ROOT=/path/to/generated/roach_bev_maps
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:${PYTHONPATH:-}"
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export CARLA_LAUNCH_USER=carla
```

验证依赖：

```bash
python -c "import carla, cv2, h5py, numpy, psutil, pygame; print(carla.Client, h5py.__version__)"
```

## 2. CARLA 生命周期

默认不需要手工启动 server。生成器的启动条件与本项目已经跑通过的
`Bench2Drive/leaderboard/rl/sim_backend.py` 对齐：

```text
root 容器通过 su - carla 启动
为 carla 用户创建并 chown XDG_RUNTIME_DIR
SDL_AUDIODRIVER=dummy
显式传 CUDA_VISIBLE_DEVICES
按需传 DISPLAY
只在用户显式提供时设置 VK_ICD_FILENAMES
RenderOffScreen + nosound + rpc port + graphicsadapter
等待 RPC ready，再保留 server warmup 时间
结束时只清理本命令拥有的进程组
```

当前容器曾出现“错误强制注入 `nvidia_egl_icd.json` 导致 `VK_ERROR_DEVICE_LOST`/Signal 11”的问题，
因此工具不会自动设置 `VK_ICD_FILENAMES`。需要指定时使用 `--vk-icd-filenames`。

当前容器还存在 Vulkan adapter 编号与 `nvidia-smi` 物理编号不一致：已验证
`-graphicsadapter=2` 对应物理 1 号 NVIDIA GPU。若希望复用该已验证组合，可设置：

```bash
export DISPLAY=:99
unset VK_ICD_FILENAMES
```

然后在命令中传：

```text
--cuda-visible-devices 1
--graphics-adapter 2
--carla-launch-user carla
--xdg-runtime-dir /tmp/carla-runtime-roach-map-2000
--server-warmup-seconds 30
```

如果换机器或换容器，不能照搬 `graphicsadapter=2`；应重新核对 Vulkan 枚举。生成器使用
`$CARLA_ROOT/CarlaUE4.sh` 启动一个归本命令所有的进程，完成所有 Town 后正常终止。

可选参数：

```text
--cuda-visible-devices
--graphics-adapter
--server-log
--server-startup-timeout-seconds
--server-shutdown-timeout-seconds
--server-warmup-seconds
--server-extra-arg
--carla-launch-user
--xdg-runtime-dir
--vk-icd-filenames
--display
```

如果用户已经启动了专用 server，可以显式加入 `--connect-existing`。禁止连接正在训练/评测的实例，
因为生成器会调用 `client.load_world(town)`。

## 3. Town01 预检查

`--dry-run` 会自行启动 CARLA、加载 Town01、计算 bounds 和完整十层内存估计，然后关闭 CARLA；不会
创建 pygame surface 或写地图文件：

```bash
python Bench2Drive/tools/generate_roach_static_maps.py \
  --towns Town01 \
  --host 127.0.0.1 \
  --port 2000 \
  --cuda-visible-devices 1 \
  --graphics-adapter 2 \
  --carla-launch-user carla \
  --xdg-runtime-dir /tmp/carla-runtime-roach-map-2000 \
  --dry-run
```

如需连接手工启动的 server：

```bash
python Bench2Drive/tools/generate_roach_static_maps.py \
  --towns Town01 \
  --host 127.0.0.1 \
  --port 2000 \
  --connect-existing \
  --dry-run
```

关注 JSON 日志中的：

```text
bounds.width_in_pixels
memory_estimate.estimated_peak_gib
server_version
```

## 4. 生成 Town01 global asset

```bash
python Bench2Drive/tools/generate_roach_static_maps.py \
  --towns Town01 \
  --host 127.0.0.1 \
  --port 2000 \
  --cuda-visible-devices 1 \
  --graphics-adapter 2 \
  --carla-launch-user carla \
  --xdg-runtime-dir /tmp/carla-runtime-roach-map-2000 \
  --output-dir "$ROACH_BEV_MAP_ROOT" \
  --pixels-per-meter 5.0 \
  --max-estimated-memory-gb 8
```

输出：

```text
$ROACH_BEV_MAP_ROOT/Town01.h5
$ROACH_BEV_MAP_ROOT/Town01.h5.manifest.json
```

默认拒绝覆盖。只有确认旧文件可以替换时才显式传 `--overwrite`。`--allow-unsafe-global` 不会由
`--overwrite` 隐式打开。`--max-estimated-memory-gb` 只是生成前的安全阈值，不会预分配对应内存；本机
有约 100 GiB 可用内存时可以显式设为 `100`，但不能大于实际可用内存。

## 5. 批量生成、切图重试与断点续跑

一个自管理 CARLA 进程可以连续处理多个 Town：

```bash
python Bench2Drive/tools/generate_roach_static_maps.py \
  --towns Town02 Town03 Town04 Town05 Town07 \
  --host 127.0.0.1 \
  --port 2000 \
  --cuda-visible-devices 1 \
  --graphics-adapter 2 \
  --carla-launch-user carla \
  --xdg-runtime-dir /tmp/carla-runtime-roach-map-2000 \
  --output-dir "$ROACH_BEV_MAP_ROOT" \
  --pixels-per-meter 5.0 \
  --max-estimated-memory-gb 100
```

连续切换地图时，CARLA 偶尔会在 `load_world()` 返回后短暂报告上一张地图。生成器默认会校验返回的
world、轮询当前 world，并最多重新加载三次。相关参数为：

```text
--world-load-attempts 3
--world-load-settle-seconds 20
```

批处理中断后不需要覆盖或重新计算已完成资产。既可以只传未完成 Town，也可以在原 Town 列表上加入
`--skip-existing`。跳过前仍会检查现有 HDF5 的 schema、manifest/hash 和嵌入的 Town 名，损坏或标错的
资产会直接报错。例如 Town10HD 失败、Town15 尚未执行时，推荐分别恢复，避免某个 Town 的 CARLA
加载问题影响另一个 Town：

```bash
python Bench2Drive/tools/generate_roach_static_maps.py \
  --towns Town10HD \
  --host 127.0.0.1 \
  --port 2000 \
  --cuda-visible-devices 1 \
  --graphics-adapter 2 \
  --carla-launch-user carla \
  --xdg-runtime-dir /tmp/carla-runtime-roach-map-2000 \
  --output-dir "$ROACH_BEV_MAP_ROOT" \
  --max-estimated-memory-gb 100 \
  --world-load-attempts 3 \
  --world-load-settle-seconds 30
```

如果三次后仍无法切换，最终错误会列出 CARLA server 的 `available_maps`。此时应先确认当前 CARLA
安装确实包含 canonical `Town10HD`，不能把仍为上一地图的结果写成 Town10HD 资产，也不能未经核对
擅自改用 `Town10HD_Opt`。

当前本机 CARLA 0.9.15 内容目录已核对存在 canonical `Town10HD.umap` 和 `Town10HD.uexp`，也存在
Town11、Town12、Town13、Town15 的主 `.umap`，因此本次 Town10HD 更符合连续切图未完成，而不是地图
安装缺失。

## 6. 离线验证

validator 不连接 CARLA，默认同时接受 global/tiled：

```bash
python Bench2Drive/tools/validate_roach_static_maps.py \
  "$ROACH_BEV_MAP_ROOT/Town01.h5"
```

它检查：

- asset format/storage mode；
- Town/CARLA/OpenDRIVE metadata；
- `pixels_per_meter`、world offset、shape 和 dtype；
- 十个 required dataset、chunk/compression；
- 非零比例和 unique values；
- sidecar manifest 和文件 SHA-256。

若要强制检查 storage mode：

```bash
python Bench2Drive/tools/validate_roach_static_maps.py \
  --storage-mode global "$ROACH_BEV_MAP_ROOT/Town01.h5"

python Bench2Drive/tools/validate_roach_static_maps.py \
  --storage-mode tiled "$ROACH_BEV_MAP_ROOT/Town11.h5"
```

只检查 metadata/shape、不流式扫描所有像素：

```bash
python Bench2Drive/tools/validate_roach_static_maps.py \
  --metadata-only "$ROACH_BEV_MAP_ROOT/Town01.h5"
```

对 Town11/Town12/Town13 这类大 tiled asset，优先用 `--metadata-only` 做 schema/manifest/hash 检查；
完整流式像素扫描会遍历十个超大 layer，通常不作为每次生成后的默认 gate。

## 7. 生成 Town11/Town12/Town13 tiled assets

推荐先用一个自管理 CARLA 进程顺序生成三个大 Town；`--skip-existing` 支持断点续跑：

```bash
cd "$PROJECT_ROOT"
conda activate roach-map
unset VK_ICD_FILENAMES
export DISPLAY=:99

python Bench2Drive/tools/generate_roach_static_maps.py \
  --towns Town11 Town12 Town13 \
  --carla-root "$CARLA_ROOT" \
  --output-dir "$ROACH_BEV_MAP_ROOT" \
  --storage-mode tiled \
  --tile-size-pixels 8192 \
  --tile-clip-padding-meters 100 \
  --max-estimated-memory-gb 32 \
  --host 127.0.0.1 \
  --port 2011 \
  --timeout-seconds 180 \
  --server-startup-timeout-seconds 240 \
  --server-warmup-seconds 30 \
  --world-load-attempts 3 \
  --world-load-settle-seconds 30 \
  --cuda-visible-devices 1 \
  --graphics-adapter 2 \
  --carla-launch-user carla \
  --display :99 \
  --server-log "$ROACH_BEV_MAP_ROOT/tiled_generation_carla.log" \
  --skip-existing \
  --progress-every 500
```

当前容器已验证物理 GPU 1 对应 `--graphics-adapter 2`；换机器必须重新探测。若 CARLA 启动或权限失败，
不要改用 root-owned CARLA 路径覆盖问题，先确认 `su - carla`、`DISPLAY=:99`、端口和 GPU 映射。

tiled 设计固定如下：

```text
storage_mode:             tiled
tile_size_pixels:         8192
tile_overlap_pixels:      0
tile_clip_padding_meters: 100
HDF5 layer datasets:      仍为 [global_width, global_width] uint8
runtime-facing semantics: 与 global asset 相同 layer 名称、ppm、world offset 和 crop 语义
```

生成后做 metadata/hash validator：

```bash
python Bench2Drive/tools/validate_roach_static_maps.py \
  --metadata-only --storage-mode tiled \
  "$ROACH_BEV_MAP_ROOT/Town11.h5" \
  "$ROACH_BEV_MAP_ROOT/Town12.h5" \
  "$ROACH_BEV_MAP_ROOT/Town13.h5"
```

2026-07-06 生成状态：Town11、Town12、Town13 已完成正式 tiled asset 生成，并通过
`roach-map` 环境下的 metadata/hash/tile-schema validator。若命令输出 `/Town11.h5` 这类路径，说明
`ROACH_BEV_MAP_ROOT` 未在当前 shell 中 export，并不是资产不存在。

| Town | storage | shape | tiles | HDF5 bytes | SHA-256 前 12 位 |
| --- | --- | ---: | ---: | ---: | --- |
| Town11 | tiled | `108426 x 108426` | `14 x 14 = 196` | `145359072` | `3a59644c4f26` |
| Town12 | tiled | `47990 x 47990` | `6 x 6 = 36` | `39345966` | `ed448f365ad2` |
| Town13 | tiled | `66331 x 66331` | `9 x 9 = 81` | `69735869` | `c5e9d5533a8b` |

## 8. 可视化

静态地图可视化不连接 CARLA。global asset 默认生成 overview；tiled asset 默认根据 manifest 选择 road
非零最多的 tile 中心 crop，避免整张大图载入内存：

```bash
python Bench2Drive/tools/visualize_roach_static_maps.py \
  "$ROACH_BEV_MAP_ROOT/Town01.h5" \
  "$ROACH_BEV_MAP_ROOT/Town11.h5" \
  --output-dir "$ROACH_BEV_MAP_ROOT/visualizations" \
  --crop-size-pixels 2048 \
  --max-overview-pixels 2048
```

输出 PNG 会叠加 road、shoulder/parking/sidewalk、lane marking 和 stopline，主要用于快速确认资产不是空图、
没有整体镜像/旋转、stopline/lane 层存在。

2026-07-06 已生成并人工抽查的 Town11～13 tiled crop：

```text
roach_bev_maps/visualizations/Town11_crop_r76800_c76800_2048x2048.png
roach_bev_maps/visualizations/Town12_crop_r27648_c27648_2048x2048.png
roach_bev_maps/visualizations/Town13_crop_r27648_c44032_2048x2048.png
```

三张图均非空，能看到 road、lane marking、road boundary/sidewalk 类边缘和 stopline；未发现明显整体镜像、
旋转或 tile 边界断裂。

## 9. 与原 Roach Town01 做 world-coordinate parity

旧 Roach 资产由 CARLA 0.9.9.4 生成，不能要求和 0.9.15 新资产逐文件一致，但必须按 world offset 对齐后
量化十层资产差异：

```bash
python Bench2Drive/tools/compare_roach_static_maps.py \
  /path/to/carla-roach/carla_gym/core/obs_manager/birdview/maps/Town01.h5 \
  "$ROACH_BEV_MAP_ROOT/Town01.h5"
```

比较器检查相同 pixels-per-meter、像素网格对齐和世界坐标重叠区域，并逐层输出 IoU、binary mismatch、
exact mismatch 和 unique values。由于 CARLA/OpenDRIVE 版本不同，差异不一定代表迁移错误；需要结合
OpenDRIVE hash 和 overlay 定位。

## 10. Smoke tests

不需要 CARLA，但需要 `roach-map` 环境中的 `h5py`：

```bash
python Bench2Drive/test_roach_static_map_schema_smoke.py
```

不连接 CARLA server、也不需要 `h5py` 的 rasterizer 几何 smoke：

```bash
python Bench2Drive/test_roach_map_rasterizer_smoke.py
python Bench2Drive/test_roach_map_owned_server_smoke.py
python Bench2Drive/test_roach_map_world_load_smoke.py
```

## 11. 第一阶段验收顺序

```text
schema/rasterizer smoke
  -> Town01 dry-run
  -> Town01 global generate
  -> schema/hash validator
  -> 与原 Roach Town01 十层资产做 world-coordinate parity
  -> Town06 dry-run/generate，校准面积-内存倍率
  -> 批量生成其他正常规模 Town，并逐个做 schema/hash validator
  -> Town11/Town12/Town13 使用 tiled 生成，metadata/hash validator 和 crop 可视化（2026-07-06 已完成）
```

Town01 的目的不是设置繁琐门槛，而是确认 CARLA 0.9.15 下十层内容、坐标、内存和文件 schema 正常；
通过后即可批量处理适合 global 的 Town。大 Town 不以实际 OOM 作为测试手段。

## 12. 运行时 Roach BEV target 接入状态

2026-07-06 已接入运行时 target 生成与 transient cache：

- 静态 map asset 仍由本目录工具生成，放在 `$ROACH_BEV_MAP_ROOT/*.h5`；
- 运行时 target 由 `Bench2Drive/leaderboard/rl/roach_bev_target.py` 读取静态 asset，并结合 env 当前帧的
  route、vehicle/walker、traffic-light stopline、stop-sign trigger volume 生成 Roach contract masks；
- target shape 固定为 `[15,192,192]`，dtype 为 `uint8`，通道顺序为：

```text
road, route, lane,
vehicle_h-16, vehicle_h-11, vehicle_h-6, vehicle_h-1,
walker_h-16, walker_h-11, walker_h-6, walker_h-1,
traffic_light_stop_h-16, traffic_light_stop_h-11, traffic_light_stop_h-6, traffic_light_stop_h-1
```

- target 坐标约定与 Roach 一致：`pixels_per_meter=5.0`，`pixels_ev_to_bottom=40`，ego-centric BEV；
- `Bench2Drive/leaderboard/rl/env.py` 中的 generator 默认关闭，仅当 trainer 配置
  `adapter_prediction_enabled=True` 且 `adapter_prediction_train_semantic=True` 时启用；
- `env.reset()` 和 `env.step()` 返回 observation 时会生成对应 `sensor_frame` 的 target，并写入
  `FrameKeyedRoachBevTargetCache`；
- trainer 使用 `env.pop_roach_bev_target(expected_frame=int(observation["sensor_frame"]))` one-shot 消费；
- target 不进入 observation 常驻字段，不长期塞进 `info`，不进入 replay；
- `sensor_frame_exact=False`、target missing、frame mismatch、生成失败时，prediction update 会跳过 semantic
  loss 并记录 skip metrics。

对应 smoke：

```bash
conda run -n hipad python Bench2Drive/test_roach_bev_target_cache_smoke.py
```

静态 asset 可视化仍使用第 8 节的 `visualize_roach_static_maps.py`；运行时 `[15,192,192]` target 的 RGB
debug 渲染由 `render_roach_bev_target(masks)` 提供，env 可通过 `--roach-bev-target-debug-dir` 和
`--roach-bev-target-debug-interval` 导出当前帧 target PNG。
