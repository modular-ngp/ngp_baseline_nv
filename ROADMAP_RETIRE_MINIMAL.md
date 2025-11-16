# ngp-minimal 按需搬迁 Roadmap（独立 NeRF-synthetic 训练最小实现）

> 目标：在仓库根目录下的 `ngp-minimal/` 目录中，从零开始实现一个 **独立的、只支持 NeRF-synthetic 数据集训练的最小版 instant-ngp**。
>
> 关键要求：
> - **不直接复用完整版代码**：`ngp-minimal/` 中不 `#include` 任何 `../include` 或 `../src` 的头/源文件，也不链接 `ngp-baseline-nv` 静态库；完整版仅作为“参考实现”和验证基准。
> - **按需搬迁 + 同名接口**：
>   - 以 NeRF-synthetic 训练路径为根，梳理出实际会用到的类/函数/数学工具集合（“使用子图”）。
>   - 在 `ngp-minimal` 里，用 **相同的类名/函数名/调用关系**（例如 `Testbed::frame`、`NerfDataset`、`train_nerf_step` 等），重新实现这一子集。
>   - 对于原版中未被这条路径使用的字段/方法/辅助工具，一律不搬；只实现训练 NeRF-synthetic 所必须的最小子集。
> - **与完整版交叉验证**：通过日志、数据结构导出、图像或 snapshot 对比，验证 minimal 实现与完整版在同一任务上的行为尽量一致。

---

## 0. 核心方法：按需搬迁 = “先画子图，再重建”

1. **从入口出发锁定调用子图**
   - 入口路径：`main` → `main_func` → `Testbed` 相关调用 → NeRF-synthetic 训练循环。
   - 在完整版里通过静态阅读（必要时配合日志）列出：
     - 用到的类：`Testbed`、`Testbed::Nerf`、`NerfDataset`、`NerfNetwork` 等。
     - 用到的成员函数：如 `Testbed::frame`、`Testbed::train_and_render`、`Testbed::train`、`Testbed::training_prep_nerf`、`Testbed::train_nerf_step`、`load_nerf` 等。
     - 这些函数内部再递归展开，记录实际用到的数学函数、工具函数、结构体字段。
   - 得到一张「**NeRF-synthetic 训练路径子图**」：节点是函数/类/字段，边是调用或依赖。

2. **在 ngp-minimal 中重建这一子图**
   - 在 `ngp-minimal/` 下，用相同的接口名称（类名/函数名/参数列表尽量一致）重建：
     - 例如：在 `ngp-minimal` 中也有一个 `class Testbed`，其中只定义 `frame/train/load_nerf/...` 等被训练路径实际调用的方法。
     - `struct NerfDataset` 也存在，但字段只有训练路径真正用到的那一部分。
   - 实现逻辑允许简化，但对外行为（输入输出语义）要与原版兼容，以便对比。
   - 不实现子图之外的任何东西（例如 GUI、VR、多种模式、SDF 等）。

3. **每一步都用完整版对照验证**
   - 对于数据加载、网络前向结果、训练 loss 曲线等，尽可能设计工具或脚本在 Full vs Minimal 之间做对比。
   - 发现差异后，再针对性补齐 minimal 实现中缺失的必要细节。

---

## 阶段总览

1. **Phase 1：基线确认 & 训练路径调用图梳理（从完整版出发，锁定“使用子图”）**
2. **Phase 2：`ngp-minimal` 工程骨架 + 独立 CMake + CLI（不依赖完整版）**
3. **Phase 3：按需搬迁数据层（`NerfDataset` + loader），同名精简实现**
4. **Phase 4：按需搬迁网络/采样/渲染核心（NeRF 网络 + raymarch）**
5. **Phase 5：按需搬迁 `Testbed` 与训练循环（保持调用关系，但只实现用到的部分）**
6. **Phase 6：冗余收缩、行为一致性评估与复杂度对比**

下面按阶段拆解成 Milestone，并给出可验证的中间结果和与完整版的交叉验证方式。

---

## Phase 1：基线确认 & 调用图梳理（使用子图）

### Milestone 1.1：选定基线场景与配置（完整版）

- 参考场景：使用仓库中的 `data/nerf-synthetic`：
  - 必选：`lego`；
  - 建议再选一个，如 `chair` 或 `drums`。
- 参考配置：
  - 主配置：`configs/nerf/base.json`；
  - 可准备一个“小网络”配置（如 `base_0layer.json` 或临时简化配置）用于快速回归。

**验证产物：**

- 使用当前完整版运行：  
  `./ngp-baseline-nv-app --scene data/nerf-synthetic/lego --config configs/nerf/base.json --no-gui`
  - 记录训练日志：至少包含 `iteration=<n> loss=<scalar>`，保存到 `logs/full_lego_base.log`。
  - 可选：记录某些 iteration 的 PSNR 或导出几张渲染图做视觉对照。

### Milestone 1.2：从 main 到 train 的调用路径整理

- 在完整版中阅读/标记：
  - `main.cu`：
    - `main` / `main_func` 的 CLI 参数解析逻辑；
    - 如何构造 `Testbed`，如何调用 `load_file/load_training_data/reload_network_from_file/frame`。
  - `Testbed` 相关：
    - `Testbed::load_training_data`（只关心 NeRF 路径）；
    - `Testbed::load_nerf` / `load_nerf_post`；
    - `Testbed::frame` → `train_and_render` → `train` → `training_prep_nerf` / `train_nerf_step`；
    - NeRF 专用的辅助函数：如 `update_density_grid_nerf`、ray 采样逻辑等。
  - `nerf_loader`：
    - `load_nerf` 如何处理 NeRF-synthetic（Blender）格式。

**产物：**

- 一份“函数调用子图”清单，例如：
  - 顶层：`main_func` → `Testbed::load_training_data` / `Testbed::frame`。
  - NeRF Data：`Testbed::load_nerf` → `ngp::load_nerf` → `NerfDataset::set_training_image`。
  - Training：`Testbed::train` → `train_nerf` / `train_nerf_step` → 一系列内部核函数。

这些清单将直接决定 **哪些函数/类/字段需要在 ngp-minimal 中重建**。

### Milestone 1.3：字段级使用分析（按需搬迁的字段集合）

- 对于关键结构体（仅针对 NeRF 路径）：
  - `NerfDataset`：哪些字段在 NeRF-synthetic 训练中被访问？（如 `metadata`, `xforms`, `offset`, `scale`, `aabb_scale`, `render_aabb`, `up` 等）哪些从未被用到（如某些扩展 metadata、mask、depth 等）。
  - `Testbed::Nerf::Training`：实际用于 batch 采样、density grid 更新、loss 计算的成员有哪些？哪些是 GUI 或特殊模式才用到？
  - 常用数学类型：`vec3`, `mat4x3`, `BoundingBox` 的哪些方法在训练路径中有调用？

**产物：**

- 为每个核心结构体列出两个集合：
  - **需要搬迁的字段/方法**；
  - **可以舍弃的字段/方法**。

后续 Phase 3–5 将只在 `ngp-minimal` 中实现“需要搬迁”的子集。

### Phase 1 当前仓库分析记录（基于现有代码）

> 本小节是对当前仓库中 NeRF-synthetic 训练路径的“使用子图”第一次系统梳理，后续可以在实现过程中继续补充细节。

#### 1. 参考场景与命令（完整版）

- 数据集：使用仓库中已存在的 `data/nerf-synthetic`：
  - 推荐基线：`data/nerf-synthetic/lego` 与 `data/nerf-synthetic/chair`。
- 配置：
  - 主配置：`configs/nerf/base.json`。
  - 快速回归可选：`configs/nerf/base_0layer.json`（极小网络）。
- 典型训练命令（完整版）：
  - `./ngp-baseline-nv-app --scene data/nerf-synthetic/lego --config configs/nerf/base.json --no-gui`
  - `./ngp-baseline-nv-app --scene data/nerf-synthetic/chair --config configs/nerf/base.json --no-gui`
- 日志格式（我们后续在 ngp-minimal 里会对齐）：
  - `iteration=<n> loss=<scalar>`（由 `main.cu` 中 `while (testbed.frame())` 输出）。

#### 2. 顶层调用链子图（从 CLI 到训练）

以当前 `main.cu` 为准，当只使用 `--scene` 与 `--config`（不拖文件、不用 GUI）时，实际走到的主要函数链为：

- `main` / `wmain`
  - 解析 `argv` 为 `std::vector<std::string>`，调用：
  - `ngp::main_func(const std::vector<std::string>& arguments)`
    - 使用 `args::ArgumentParser` 解析：`--scene`, `--config/--network`, `--snapshot/--load_snapshot`, `--no-train` 等；
    - 构造：`Testbed testbed;`
    - 若给定 `--scene`：`testbed.load_training_data(scene_path);`
    - 若给定 `--snapshot`：`testbed.load_snapshot(snapshot_path);`  
      否则若给定 `--config`：`testbed.reload_network_from_file(config_path);`
    - 设置：`testbed.m_train = !no_train_flag;`
    - 主循环：
      - `while (testbed.frame()) { tlog::info() << "iteration=" << m_training_step << " loss=" << m_loss_scalar.val(); }`

- `Testbed::load_training_data(const fs::path& path)`（只考虑 NeRF 路径）：
  - 校验路径存在；
  - `ETestbedMode scene_mode = mode_from_scene(path.str());`
  - `set_mode(scene_mode);`（NeRF 场景会设置为 `ETestbedMode::Nerf`）；
  - `m_data_path = path;`
  - `switch (m_testbed_mode) { case ETestbedMode::Nerf: load_nerf(path); ... }`
  - `m_training_data_available = true;`

- `Testbed::set_mode(ETestbedMode mode)`（与训练相关的部分）：
  - 重置模式相关成员：`m_mesh = {}; m_nerf = {};`；
  - 清空网络/优化器状态：`m_encoding`, `m_loss`, `m_network`, `m_nerf_network`, `m_optimizer`, `m_trainer`, `m_envmap`, `m_distortion`；
  - 清空设备相关数据：`for (auto&& device : m_devices) device.clear();`
  - 设置 `m_testbed_mode = mode;` 并调用 `reset_camera();`
  - 对 NeRF 模式有 DLSS 多 GPU 相关的逻辑，未来 ngp-minimal 可以简化为“单 GPU + 无 DLSS”。

- `Testbed::load_nerf(const fs::path& data_path)`（NeRF 数据加载入口）：
  - 构造 JSON 路径列表（目录下所有 `.json`）：
    - train/test/val：`transforms_*.json`；
  - 记录之前的 `aabb_scale`；
  - `m_nerf.training.dataset = ngp::load_nerf(json_paths, m_nerf.sharpen);`
  - 如 `aabb_scale` 变化且已有网络配置，则调用 `reset_network()`；
  - 紧接着调用 `load_nerf_post()`（在 `testbed_nerf.cu` 中），完成 density grid 等训练状态初始化。

- `Testbed::frame()`（只看训练路径相关逻辑）：
  - 根据 `m_train` 与摄像机变化情况决定是否 `skip_rendering`；
  - 若在录制 camera path，则调用 `prepare_next_camera_path_frame()`（可视为可选路径）；
  - 调用 `train_and_render(skip_rendering);`；
  - 返回 `true`（主循环就停在这里）。

- `Testbed::train_and_render(bool skip_rendering)`（NeRF-relevant 部分）：
  - 若 `m_reload_network` 等标记置位，则调用 `reload_network_from_file` 或 `reset_network`；
  - 若 `m_train == true`，则调用 `train(m_training_batch_size);`；
  - 渲染部分会调用 `render_frame` / `render_nerf`，用于产生当前视角输出（ngp-minimal 可以初期只保留训练路径，渲染可选）。

- `Testbed::train(uint32_t batch_size)`（在 NeRF 模式下）：
  - 根据 `m_testbed_mode` 选择：`case ETestbedMode::Nerf: training_prep_nerf(batch_size, stream); train_nerf_step(batch_size, m_nerf.training.counters_rgb, stream);`；
  - 更新训练计数器（`m_training_step`、各种统计信息等）。

- `Testbed::training_prep_nerf` / `Testbed::train_nerf_step`（`src/testbed_nerf.cu`）：
  - `training_prep_nerf`：根据当前 `m_training_step` 调用 `update_density_grid_nerf`，刷新 occupancy/density grid；
  - `train_nerf_step`：
    - 分配训练临时缓冲（rays、coords、mlp 输出、梯度等）；
    - 生成训练样本（调用 `generate_training_samples_nerf` 内核）；
    - 前向网络（`m_network->inference_mixed_precision` 或 fused JIT 内核）；
    - 计算损失（`compute_loss_kernel_train_nerf` 或 fused JIT）；
    - 回写梯度并通过 `m_trainer->training_step` 完成反向与优化；
    - 可选：envmap 训练、error map 更新、camera / extra dims 优化等高阶功能。

以上就是当前 NeRF-synthetic 训练在完整版中的主路径；`ngp-minimal` 的目标是只重建这条链上**实际被使用**的节点，并确保类名 / 函数名 / 调用顺序尽量一致。

#### 3. 初步结构体字段使用分析（示例：NerfDataset & Testbed::Nerf）

> 这里先给出一个粗粒度视图，后续在实现过程中可以延伸为更细的字段级列表。

- `struct NerfDataset`（`include/neural-graphics-primitives/nerf_loader.h`）
  - **在 NeRF-synthetic 训练中明显被使用的字段：**
    - `std::vector<TrainingImageMetadata> metadata;` / `GPUMemory<TrainingImageMetadata> metadata_gpu;`
      - 包含每张图像的 `resolution`, `focal_length`, `principal_point`, `lens`, `rolling_shutter`, `pixels`, `depth` 等；
      - 训练时用于 ray 生成和 loss 计算。
    - `std::vector<TrainingXForm> xforms;`
      - 训练时用于从训练视角构造相机位姿/方向。
    - `std::vector<std::string> paths;`
      - 主要用于日志和调试，可视为「次要但有用」。
    - `GPUMemory<uint8_t> pixelmemory[...]` / `GPUMemory<float> depthmemory[...]` / `GPUMemory<Ray> raymemory[...]`：
      - pixelmemory：必需（训练用 RGB）；
      - depthmemory/raymemory：仅在启用深度监督或预计算 rays 时使用，默认配置下可以视作可选。
    - `GPUMemory<float> sharpness_data;` / `ivec2 sharpness_resolution;`
      - 在 error map 可视化与与 sharpness 结合采样时使用；
      - 在默认不启用 error overlay 时，仅在可视化路径中使用，可考虑不搬或延后搬迁。
    - `BoundingBox render_aabb; mat3 render_aabb_to_local; vec3 up; vec3 offset;`
      - 用于定义训练/渲染空间的边界和方向，是 ray marching 与 density grid 的基础。
    - `size_t n_images; float scale; int aabb_scale;`
      - 训练循环中广泛使用（采样、grid 大小等）。
  - **只在高阶功能中出现、可考虑暂不搬迁的字段：**
    - `GPUMemory<float> envmap_data; ivec2 envmap_resolution; bool is_hdr;`
      - 仅在 envmap 训练路径中使用。
    - `bool wants_importance_sampling;`
      - 一定程度上影响采样策略，但可以在 minimal 里先实现简化策略。
    - `uint32_t n_extra_learnable_dims; bool has_light_dirs;`
      - 与额外 latent code / 光照方向扩展训练相关，可初期跳过。

- `Testbed::Nerf::Training`（在 `testbed.h` 中的嵌套结构）
  - **明显在训练路径中用到的成员：**
    - `dataset`：即上面的 `NerfDataset`；
    - `density_grid`, `density_grid_mean`, `n_steps_between_error_map_updates` 等 density grid/占空间统计相关；
    - `counters_rgb`：记录 batch 尺寸与统计；
    - `loss_type`, `depth_loss_type`, `random_bg_color`, `linear_colors` 等 loss & 渲染策略配置；
    - `train_mode`（`ETrainMode::Nerf`）、`snap_to_pixel_centers` 等训练选项。
  - **主要用于高阶特性，可以在 minimal 里简化或先不实现的成员：**
    - `error_map` 及其 `cdf_*` 字段：用于 error-based 重要性采样和可视化；
    - `train_envmap` 和相关 envmap 训练成员；
    - `optimize_extrinsics`, `optimize_distortion`, `optimize_focal_length`, `optimize_exposure` 及对应梯度/优化器：用于在线标定和曝光学习。

在后续实现 ngp-minimal 时，我们会根据这种分类：

- 对「训练主路径中必用」的字段与方法，在 ngp-minimal 里用相同名字实现；
- 对「只在高阶或可视化路径中使用」的部分则不搬迁，除非后面确实需要扩展。这样就真正做到了“按需搬迁 + 去冗余”。

---

## Phase 2：`ngp-minimal` 工程骨架 + 独立 CLI

### Milestone 2.1：独立 CMake 与可执行入口

- 在 `ngp-minimal/` 下建立：
  - `ngp-minimal/CMakeLists.txt`：定义最小工程目标 `ngp-minimal-app`；
  - `ngp-minimal/src/main.cu`：定义 `int main(int argc, char** argv)`。

- 要求：
  - 不 `target_link_libraries(ngp-minimal-app PRIVATE ngp-baseline-nv)`；
  - 所有 include 必须在 `ngp-minimal` 自己的 include 路径或第三方路径中；
  - 在顶层 `CMakeLists.txt` 中通过 `add_subdirectory(ngp-minimal)` 引入，但不影响原目标。

**验证：**

- 构建：`cmake --build build --config Release -j`，确认生成：  
  `build/ngp-minimal/ngp-minimal-app.exe`。

- 运行：`ngp-minimal-app --help`，输出基础帮助信息（此时可先只打印占位说明）。

### Milestone 2.2：最小 CLI 与统一入口函数

- 在 `main.cu` 中实现：
  - 使用 `args.hxx` 等库解析参数（独立于完整版）。
  - 支持至少以下参数：
    - `--scene <path>`：NeRF-synthetic 数据路径根目录；
    - `--config <path>`：网络配置 JSON；
    - `--snapshot` / `--load_snapshot`（可后续实现）；
    - `--no-train`（仅加载，验证数据管线）。
  - 将解析结果封装到 `CliOptions` 结构体，并调用：  
    `int app_main(const CliOptions& opts);`

**验证：**

- 暂时在 `app_main` 中只打印解析结果，确认 CLI 行为正确。

---

## Phase 3：数据层按需搬迁（同名精简版 `NerfDataset` + loader）

### Milestone 3.1：在 ngp-minimal 定义同名 `NerfDataset`（精简字段）

- 在 `ngp-minimal/include/` 中定义：

```cpp
namespace ngp {

struct TrainingImageMetadata {
    ivec2 resolution;
    vec2  focal_length;
    vec2  principal_point;
    // 如需简单 lens 支持，可保留一个枚举 + 少量参数，否则先固定为 pinhole。
};

struct TrainingXForm {
    mat4x3 start;
    mat4x3 end;
};

struct NerfDataset {
    std::vector<TrainingImageMetadata> metadata;
    std::vector<TrainingXForm>        xforms;
    std::vector<std::string>          paths;

    // 简化像素存储方式：可以是按图像拆分，也可以是全局打平；只要与采样逻辑一致即可。
    std::vector<GPUMemory<uint8_t>>   pixelmemory;

    BoundingBox aabb;
    vec3 up;
    vec3 offset;
    float scale;
    int   aabb_scale;
};

}
```

- 注意：类名 `NerfDataset` 与完整版一致，但字段只包含 Phase 1.3 中标记为“需要搬迁”的子集。

### Milestone 3.2：实现 `load_nerf_synthetic(const std::string& root)`

- 在 `ngp-minimal/src/nerf_loader.cu` 中实现：
  - 从 `<root>/transforms_train.json`、`transforms_test.json`、`transforms_val.json` 中加载 frames；
  - 解析 `file_path`、`transform_matrix`、`camera_angle_x` 等字段；
  - 通过约定的方式（例如按原论文）计算 `focal_length`、`principal_point`；
  - 读取对应 PNG/EXR 图像到 GPU/CPU 内存；
  - 初始化 `NerfDataset` 的所有必需字段（metadata/xforms/pixelmemory/aabb/...）。

**与完整版的交叉验证：**

- 在完整版中编写一个小工具（或临时代码路径）：
  - 使用原版 `ngp::load_nerf` 加载相同场景，导出：
    - `n_images`；
    - 每张图像的 `resolution`, `focal_length`, `principal_point`；
    - 每个 xform 的平移和朝向（如 `start[3]`、`start[2]` 向量）；
    - 全局 `offset/scale/aabb/up/aabb_scale`。

- 在 `ngp-minimal` 中：
  - 使用 `load_nerf_synthetic` 加载同一场景，导出同样信息。
  - 编写脚本对比二者差异（数值误差允许在一定范围内）。

**Milestone 完成条件：**

- `NerfDataset` 在 minimal 中可以成功加载 `data/nerf-synthetic/lego`；
- 与完整版导出的元数据对比，关键字段基本一致（误差在浮点容忍度内）。

---

## Phase 4：网络/采样/渲染核心按需搬迁（NeRF 网络 + ray marching）

### Milestone 4.1：同名 `NerfNetwork` 封装（基于 tiny-cuda-nn）

- 在 `ngp-minimal/include/nerf_network.h` 内定义：

```cpp
namespace ngp {

class NerfNetwork {
public:
    NerfNetwork(const NetworkConfig& cfg);

    void forward(const GPUMatrixDynamic<float>& inputs,
                 GPUMatrixDynamic<float>& outputs,
                 cudaStream_t stream);

    void backward_and_update(...);
};

}
```

- 名称与原版一致（`NerfNetwork`），但只实现 NeRF-synthetic 训练需要的接口。

### Milestone 4.2：体渲染与损失核心（按需搬迁）

- 在 `ngp-minimal/src/train_nerf.cu`（或类似命名）中实现：
  - 从 rays/pixels 采样一批训练样本；
  - 执行 NeRF 体积渲染（ray marching + alpha 组合）；
  - 使用 L2 / Huber 等损失函数对 RGB 进行监督；
  - 返回 batch loss 以及对网络参数的梯度，并通过优化器更新。

- 函数命名尽量保持与原版一致，例如：
  - `train_nerf_step` / `generate_training_samples_nerf` 等，只实现实际被调用的那部分。

**验证方式：**

- 在 Full 端与 Minimal 端分别固定一小批样本，比较：
  - 前向颜色结果；
  - 单步 train step 前后的 loss 变化趋势（允许有差异但要合理）。

---

## Phase 5：`Testbed` 与训练循环按需搬迁

### Milestone 5.1：在 ngp-minimal 中定义同名 `Testbed`（精简字段/方法）

- 在 `ngp-minimal/include/testbed.h` 中定义：

```cpp
namespace ngp {

class Testbed {
public:
    Testbed();

    void load_training_data(const std::string& scene_root);
    void reload_network_from_file(const std::string& config_path);

    void load_snapshot(const std::string& path);
    void save_snapshot(const std::string& path) const;

    bool frame(); // 一步训练 + 可选评估

    uint64_t training_step() const { return m_training_step; }
    float    loss_scalar() const { return m_loss_scalar; }

    bool m_train = true;

private:
    NerfDataset m_dataset;
    NerfNetwork m_network;
    // 训练参数：batch size、density grid 等 —— 仅保留 NeRF-synthetic 路径用到的。
    uint64_t m_training_step = 0;
    float    m_loss_scalar = 0.0f;
};

}
```

- 类名 `Testbed` 与原版一致；
- 成员只保留 Phase 1.3 中标记为 “NeRF-synthetic 训练路径实际访问” 的子集（例如不包含 GUI、SDF 模式、envmap 等）。

### Milestone 5.2：实现 `frame()` 调用链（保持调用关系）

- 在 `ngp-minimal/src/testbed.cu` 中实现：
  - `Testbed::frame`：
    - 若 `m_train == true`，则调用内部的 `train_step_nerf()`；
    - 可选在某些步数上做评估或打印进度；
    - 更新 `m_training_step` 和 `m_loss_scalar`。
  - `train_step_nerf()` 内部调用 Phase 4 中的训练函数以及 dataset 提供的采样接口。

- CLI 层（Phase 2）不变：
  - `app_main` 中构造 `Testbed`，调用：

```cpp
ngp::Testbed testbed;
testbed.load_training_data(opts.scene);
testbed.reload_network_from_file(opts.config);
testbed.m_train = !opts.no_train;

while (testbed.frame()) {
    log_info("iteration={} loss={}", testbed.training_step(), testbed.loss_scalar());
}
```

**与完整版交叉验证：**

- 使用 Phase 1 的基线场景与配置，在相同迭代步（例如前 2k/5k 步）对比：
  - loss 下降趋势是否一致；
  - 收敛速度是否接近。

如有较大差异，则检查数据加载/网络/训练核心是否有逻辑区别，再做修正。

---

## Phase 6：冗余收缩、行为一致性评估与复杂度对比

### Milestone 6.1：删除未用接口与字段

- 在 `ngp-minimal` 内再次用 `rg` 或 IDE 搜索：
  - 标记所有未被引用的字段/方法/辅助函数；
  - 分批删除并每次做一次小规模训练回归测试（例如极简配置 + 小步数）。

### Milestone 6.2：精简依赖

- 如果某些第三方库最终没有使用：
  - 从 `CMakeLists.txt` 中移除对应的 `find_package` 和 `target_link_libraries`；
  - 确保构建仍然通过。

### Milestone 6.3：整体评估

- 与完整版对比：
  - 源文件数量、代码行数；
  - 构建时间；
  - 依赖库数量。

- 行为层面：
  - loss 曲线、训练时间、最终 PSNR/视觉效果对比。

---

## 测试样例汇总（供实现过程使用）

1. **训练日志对比**  
   - Full vs Minimal：相同场景和配置下，在固定步数上对比 loss 记录。
2. **Dataset 对齐**  
   - 对比 Full 与 Minimal 的 `NerfDataset` / `NerfDataset` 元数据（分辨率、内参、姿态、aabb 等）。
3. **网络输出对比（可选）**  
   - 在一小批固定输入上，对比 Full 与 Minimal 的网络输出差异。
4. **渲染图像对比（可选）**  
   - 在相同迭代步导出渲染结果，做 PSNR/SSIM 或目测对比。
5. **极简网络快速回归测试**  
   - 使用小网络/小 batch/少步数，作为每次改动后的快速 sanity check。

---

## 小结

新的 Roadmap 明确了：

- `ngp-minimal/` 将是一个 **完全独立** 的实现，不包含任何直接复用的完整版代码；
- 通过「**按需搬迁**」策略，我们从完整版中先抽取 NeRF-synthetic 训练的“使用子图”，再在 minimal 中用 **相同的类名/函数名/调用关系** 重建这条路径，但只实现真正用到的子集；
- 每个阶段都设计了可验证的中间成果，并通过与完整版的交叉对比逐步保证行为的一致性。

这样最终得到的 `ngp-minimal`：

- 结构上高度贴近原始 instant-ngp 的 NeRF-synthetic 流程，便于理解与后续拓展；
-
  代码规模和依赖则显著缩减，只保留与你当前“训练 NeRF-synthetic 数据集”需求相关的核心部分。 
