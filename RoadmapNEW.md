# ngp-baseline-nv 精简版 RoadmapNEW（NeRF synthetic 训练专用）

目标：只保留 **nerf_synthetic（Blender NeRF）数据集的训练 pipeline** 所需代码，其余 GUI / 多任务 / 复杂特性全部尽可能删掉，同时保证：

- 命令行训练入口：`ngp-baseline-nv-app`。
- 数据路径：`--scene <nerf_synthetic_root>` + `--config configs/nerf/*.json`。
- 训练主干：`main.cu -> Testbed::frame -> Testbed::train_and_render -> Testbed::train -> Testbed::training_prep_nerf / train_nerf`。

本 Roadmap 按阶段列出「已完成」与「待精简」的内容，后续改动可以严格对照执行。

---

## 阶段 0：当前结构概览（NeRF-only 基线）

**核心模块（必须保留）：**

- 入口与控制：
  - `main.cu`：命令行解析、构建 `Testbed`、循环调用 `frame()`。
  - `include/neural-graphics-primitives/testbed.h`
  - `src/testbed.cu`
  - `src/testbed_nerf.cu`
- 公共基础：
  - `include/neural-graphics-primitives/common.h`
  - `include/neural-graphics-primitives/common_device.cuh`
  - `include/neural-graphics-primitives/common_host.h`
  - `include/neural-graphics-primitives/bounding_box.cuh`
  - `include/neural-graphics-primitives/random_val.cuh`
  - `include/neural-graphics-primitives/thread_pool.h`
  - `src/common_host.cu`
  - `src/thread_pool.cpp`
- NeRF 训练：
  - `include/neural-graphics-primitives/nerf.h`
  - `include/neural-graphics-primitives/nerf_device.cuh`
  - `include/neural-graphics-primitives/nerf_network.h`
  - `include/neural-graphics-primitives/nerf_loader.h`
  - `src/nerf_loader.cu`
- 渲染 / 累积 / 输出：
  - `include/neural-graphics-primitives/render_buffer.h`
  - `src/render_buffer.cu`
  - `include/neural-graphics-primitives/marching_cubes.h`
  - `src/marching_cubes.cu`（仅用于 NeRF 的 mesh 导出，可视为「可选增强」）
- 训练优化相关：
  - `include/neural-graphics-primitives/adam_optimizer.h`
  - `include/neural-graphics-primitives/trainable_buffer.cuh`
  - `include/neural-graphics-primitives/dlss.h`（可选，取决于是否保留 DLSS）
  - `include/neural-graphics-primitives/envmap.cuh`（可选，若关闭 envmap 训练可进一步删减）
  - `include/neural-graphics-primitives/discrete_distribution.h`
- I/O 与资源：
  - `include/neural-graphics-primitives/tinyexr_wrapper.h`
  - `src/tinyexr_wrapper.cu`
  - `include/neural-graphics-primitives/json_binding.h`
  - `src/camera_path.cu`, `include/neural-graphics-primitives/camera_path.h`（相机路径与离线渲染支持，可选）
  - `include/neural-graphics-primitives/shared_queue.h`
  - `include/neural-graphics-primitives/triangle.cuh`

**已从工程中物理删除的非 NeRF 功能：**

- GUI / VR / Python：
  - 所有与 `NGP_GUI` / ImGui / GLFW / OpenXR / Python binding 相关的头/源文件与代码块（已删）。
- SDF / Image / Volume：
  - `sdf.h`, `sdf_device.cuh`, `fused_kernels/trace_sdf.cuh`。
  - Testbed 中 `Sdf` / `Image` / `Volume` 结构体与全部接口已删。
- Mesh SDF + Takikawa encoding + OBJ 加载：
  - `triangle_bvh.cuh`, `triangle_octree*.cuh`, `takikawa_encoding.cuh`, `tinyobj_loader_wrapper.*`, `src/triangle_bvh.cu` 等已删。
- CMake：
  - 仅构建与 NeRF pipeline 直接相关的 `.cu/.cpp` 文件。

> 现状：本仓库已经是 **NeRF-only + 无 GUI** 的基线，实现了原始 instant-ngp 的绝大部分 NeRF 训练逻辑，但仍包含不少「高级特性」（envmap、畸变优化、多数据格式等），需要进一步围绕 nerf_synthetic 训练精简。

---

## 阶段 1：锁定 NeRF synthetic 训练路径（已完成）

**目标：** 明确并保留唯一训练路径，作为后续精简的「金线」。

1. **入口主流程（已完成）**
   - `main.cu::main_func`：
     - 解析命令行参数（`--scene`, `--config/--network`, `--snapshot/--load_snapshot`, `--no-train` 等）。
     - 构造 `Testbed testbed;`，调用：
       - `testbed.load_file(...)`（可加载网络配置/快照/相机路径）。
       - `testbed.load_training_data(get(scene_flag));`（仅针对场景路径）。
       - `testbed.reload_network_from_file(...)`。
     - 训练循环：
       - `while (testbed.frame()) { tlog::info() << "iteration=" << m_training_step << " loss=" << m_loss_scalar.val(); }`

2. **Testbed 训练主干（已完成）**
   - `Testbed::frame`（`src/testbed.cu`）：
     - 控制训练步数 / 渲染跳帧（`m_render_skip_due_to_lack_of_camera_movement_counter`）。
     - 处理相机路径录制（`m_camera_path`）。
     - 调用 `train_and_render(skip_rendering)`。
   - `Testbed::train_and_render`：
     - 若 `m_train == true`：调用 `train(m_training_batch_size)`。
     - 确保网络已加载（否则 `reload_network_from_file()`）。
     - 进行相机平滑与自动对焦（可在后续阶段选择性保留/删除）。
     - 调用 NeRF 渲染逻辑（`testbed_nerf.cu` 中实现）。

3. **NeRF 训练实现（已完成）**
   - `Testbed::training_prep_nerf` / `Testbed::train_nerf` / `Testbed::train_nerf_step`：
     - 在 `src/testbed_nerf.cu` 中，负责：
       - 构造射线 / density grid 更新；
       - 构造 batch，调用 tiny-cuda-nn 网络前向/反向；
       - 维护 error map、density grid EMA 等。
   - `Testbed::network_dims_nerf`：
     - 定义 NeRF 网络输入/输出/坐标维度。

4. **数据加载路径（已完成，但仍支持多数据格式）**
   - `src/nerf_loader.cu` + `nerf_loader.h`：
     - 定义 `NerfDataset`、`TrainingImageMetadata` 等结构体。
     - 从磁盘读取：
       - NeRF Blender synthetic：`transforms_*.json` + PNG/EXR 图像。
       - 以及其它扩展格式（LLFF、NSVF、深度/mask、envmap 等）。

---

## 阶段 2：精简头文件集合（已完成的裁剪 + 建议保留集合）

**2.1 已彻底删除的头文件（已完成）**

- SDF / Image / Volume / Mesh-SDF / OBJ：
  - `sdf.h`, `sdf_device.cuh`
  - `fused_kernels/trace_sdf.cuh`
  - `takikawa_encoding.cuh`
  - `triangle_octree.cuh`, `triangle_octree_device.cuh`
  - `triangle_bvh.cuh`, `tinyobj_loader_wrapper.h`
- GUI / Python / 其它非 NeRF 任务头文件：全部已删或在原仓库中未引入。

**2.2 建议长期保留的头文件（针对 nerf_synthetic 训练）**

> 这些头文件已经都是 NeRF-only 必需的，后续 Roadmap 的精简不会物理删除它们，而是内部「瘦身」字段/函数。

- 编码与公用：
  - `common.h`, `common_device.cuh`, `common_host.h`, `bounding_box.cuh`, `random_val.cuh`。
- NeRF 训练与网络：
  - `nerf.h`, `nerf_device.cuh`, `nerf_network.h`, `nerf_loader.h`, `render_buffer.h`, `marching_cubes.h`。
- 优化与训练基础：
  - `adam_optimizer.h`, `trainable_buffer.cuh`, `thread_pool.h`, `discrete_distribution.h`。
- 资源管理与 I/O：
  - `tinyexr_wrapper.h`, `json_binding.h`, `camera_path.h`, `shared_queue.h`。
- 可选特性：
  - `dlss.h`, `envmap.cuh`, `triangle.cuh`。

---

## 阶段 3：将 NeRF 模式收紧为「synthetic-only」数据路径（TODO）

**目标：** `nerf_loader` 只支持 Blender NeRF synthetic（`transforms_*.json` + RGB PNG/EXR），去掉其它复杂数据格式与多分支逻辑。

1. **简化 `NerfDataset` 结构（`nerf_loader.h`）**
   - **保留字段**（训练必需）：
     - 相机位姿与内参：`xforms`, `metadata[i].focal_length`, `metadata[i].principal_point`, `metadata[i].lens`。
     - 图像数据：`pixelmemory`, 分辨率、是否 HDR 等。
     - 归一化信息：`offset`, `scale`, `aabb_scale`, `render_aabb`, `render_aabb_to_local`, `up`。
   - **删除/折叠字段**（synthetic 用不到）：
     - 多数据集格式特有的字段（例如 NSVF/LLFF 的 path/白/黑透明规则等），如果仍有必要，可在加载时一律视为「普通 NeRF synthetic」处理。
     - 不参与 `testbed_nerf` 的字段（depth/mask/light_dirs/特殊 latent 维度等），逐个通过 `rg` 检查使用点并删除或改为常量。

2. **收窄 `nerf_loader.cu` 的数据分支（TODO）**
   - 在 loader 中：
     - 保留对 `transforms_*.json` + 图片的解析逻辑。
     - 删除：
       - LLFF / NSVF / 多层级结构的路径推断与特殊处理。
       - `white_2_transparent`、`black_2_transparent`、`mask_color` 等仅为特定数据集服务的分支。
       - 深度图、mask 图、特殊 light_dirs / envmap / 自定义 metadata 的读取与处理。
   - 保证在 `Testbed::load_nerf` 和 `testbed_nerf.cu` 中，对 `NerfDataset` 的访问仍完整覆盖训练所需字段。

3. **更新注释与断言（TODO）**
   - 明确在 loader 里只支持 Nerf synthetic 数据：
     - 若发现非预期的 JSON 结构或文件布局，直接抛异常，并提示「只支持 Nerf synthetic 格式」。

---

## 阶段 4：精简 NeRF 训练/渲染功能到「最小可用」集合（TODO）

**目标：** 保留能稳定训练 Nerf synthetic 的最小功能集合，删除所有「锦上添花」的高级特性：

- 可选删除/收紧的功能点（集中在 `testbed_nerf.cu` 与 `testbed.h`）：

1. **训练模式与损失（TODO）**
   - 目前 `ETrainMode` 支持 `Nerf / Rfl / RflRelax`，并在 NeRF 训练中存在多种损失组合。
   - 若项目不需要 RFL / RflRelax 的对比实验，可：
     - 固定为其中一种模式（例如标准 NeRF 或 RflRelax），移除其余模式枚举和逻辑分支。

2. **相机 / 内参优化（TODO）**
   - `m_nerf.training` 中包含：
     - 外参/内参优化（`optimize_extrinsics`, `optimize_focal_length`, `cam_pos_gradient` 等）。
   - 若只需要「固定数据集」训练，可：
     - 关闭这些优化开关，删除对应的梯度与 Adam 优化器分支。

3. **envmap 与畸变图训练（TODO）**
   - 结构体 `m_envmap`, `m_distortion`（`testbed.h`）及其训练/推理逻辑在 `testbed_nerf.cu` 内。
   - 若 nerf_synthetic 训练不需要环境光贴图或畸变图训练，可：
     - 删除相关结构与训练路径，令渲染永远采用「无额外 envmap / distortion」分支。

4. **error map / sharpness map / 复杂采样策略（TODO）**
   - 用于训练时的重要性采样加速（error map + sharpness grid）。
   - 根据性能要求决策：
     - 若希望保留训练效率，可保留 error map 主干，删掉一些可选调节开关。
     - 若追求代码极简且可接受略慢的训练，可整体移除 error map 路径，统一使用均匀/简单分布采样。

5. **多 GPU / DLSS / Foveated rendering（TODO）**
   - `Testbed` 中的：
     - 多卡渲染（`m_devices` > 1，`m_use_aux_devices`）。
     - DLSS（`m_dlss`, `m_dlss_provider`, `CudaRenderBuffer::enable_dlss`）。
     - Foveated rendering（`m_foveated_rendering` 等）。
   - 若训练主要用于离线实验、且只在单卡上运行，可逐步删除：
     - 所有多卡渲染分支，仅保留 `primary_device()` 路径。
     - DLSS 相关字段与逻辑（同时可删除 `dlss.h`）。
     - Foveated rendering 与动态分辨率逻辑，只保留固定分辨率渲染。

6. **mesh 导出与 marching cubes（可选保留）**
   - 若应用场景只需要训练而不需要导出 mesh，可：
     - 删除 `marching_cubes.h/cu` 与 `optimise_mesh_step`, `compute_mesh_vertex_colors`, `compute_and_save_marching_cubes_mesh` 等。
   - 若仍需在训练后导出 NeRF 的 iso-surface mesh，可保留现有的 NeRF-only marching cubes 实现。

---

## 阶段 5：精简命令行接口与配置（TODO）

**目标：** 让命令行和配置文件与「仅 nerf_synthetic 训练」一致，避免无效选项。

1. **CLI 参数精简（`main.cu`）**
   - 保留：
     - `--scene`（必需）
     - `--config` / `--network`（NeRF 网络配置）
     - `--snapshot` / `--load_snapshot`（训练恢复/导出）
     - `--no-train`（只渲染不训练，可选）
   - 删除或忽略：
     - 与 GUI/VR 相关的参数（`--width`, `--height`, `--vr` 等），并从帮助文档中移除。

2. **配置文件精简（`configs/nerf/*.json`）**
   - 只保留 Nerf synthetic 训练实际用到的字段：
     - 网络结构、编码参数（hashgrid/MLP 等）、优化器超参。
   - 删除或固定：
     - 与 SDF/image/volume 或 GUI 调试相关的字段。

---

## 阶段 6：最终清理与验证（TODO）

1. **全局搜索并删除悬空引用**
   - 使用 `rg` 检查：
     - 不再存在对已删头文件/源文件的 include。
     - 不再存在对已删结构体/枚举/函数的使用。

2. **CMake 构建验证**
   - `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release`
   - `cmake --build build --config Release -j8`

3. **功能验证**
   - 使用标准 NeRF synthetic 数据集运行：
     - `./ngp-baseline-nv-app --scene <nerf_synthetic_root> --config configs/nerf/base.json`
   - 观察：
     - 训练 loss 下降。
     - 若保留渲染导出功能，可在训练期间周期性输出测试视角的图像/EXR 以确认视觉效果。

---

## 小结

目前仓库已经完成了 **Phase 2 & 3** 中最重的部分（彻底移除 GUI / VR / Python / SDF / Image / Volume 等功能），并在 NeRF-only 基础上成功编译通过。  
本 RoadmapNEW 将后续工作聚焦在：

- 将 `nerf_loader` 与 `NerfDataset` 收紧为 **Nerf synthetic-only**。
- 深度裁剪 `testbed_nerf.cu` / `testbed.h` 中的高级选项（envmap、畸变、相机优化、多 GPU、DLSS 等）。
- 精简命令行和配置，使之只暴露「NeRF synthetic 训练」必需的控制点。

按此 Roadmap 逐步推进，可以最终得到一个代码量明显更小、职责明确、只围绕「训练 NeRF synthetic」的 instant-ngp 精简版。  

## Status update (2025-11-17)

- Phase 3 / `nerf_loader` synthetic-only path:
  - `src/nerf_loader.cu` now only supports Blender NeRF synthetic-style datasets (`transforms_*.json` + PNG/EXR). Depth images, per-pixel ray files, and dynamic mask branches have been removed from the loader.
  - When encountering Mitsuba-related JSON fields such as `normal_mts_args` or `from_mitsuba`, `load_nerf` now throws a clear `std::runtime_error` explaining that only NeRF synthetic format is supported.
- `NerfDataset` / JSON binding alignment:
  - `include/neural-graphics-primitives/json_binding.h` has been updated so that the serialized/deserialized fields match the current `NerfDataset` definition (e.g. the obsolete `from_mitsuba` field has been removed).
  - Extended fields used by advanced training logic (`has_rays`, `has_light_dirs`, `n_extra_learnable_dims`, etc.) are still present and used by `testbed_nerf`; they are planned to be simplified in later phases.
