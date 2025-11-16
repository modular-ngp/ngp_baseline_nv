# ngp-minimal 设计与 Retire Roadmap（NeRF-synthetic 训练最小实现）

> 目标：在项目根目录下创建 `ngp-minimal/`，在保持现有 NeRF 训练框架「结构尽量不变」（类/函数调用关系基本一致）的前提下，得到一套 **只支持 NeRF-synthetic 训练** 的“最小代码路径”，并通过一系列可验证的 milestone 与测试样例，确保其行为与当前完整版保持一致。

---

## 总体原则与范围

- **仅支持的功能**
  - 数据集：Blender NeRF synthetic（`transforms_*.json` + PNG/EXR）。
  - 模式：单一 NeRF 训练（不含 SDF / Image / Volume / 反射等）。
  - 界面：纯 CLI、无 GUI / VR / Python 绑定。
  - 设备：单 GPU（可保留多 GPU 结构但实现裁剪到单卡路径）。

- **尽量保留的框架结构**
  - 保留 `Testbed` / `Testbed::Nerf` / `NerfDataset` / `train_nerf` 等核心类和函数名，以便阅读和对照 upstream/完整版文档。
  - 保留训练主干调用关系：
    - `main (ngp-minimal-app)`  
      → `TestbedMinimal`（或裁剪后的 `Testbed`）  
      → `frame` → `train_and_render` → `train`  
      → `training_prep_nerf` / `train_nerf_step`。

- **可以删除 / 裁剪的内容（在 minimal 里）**
  - 所有在「NeRF-synthetic 单机训练」场景下不会被调用到的结构体字段、枚举值和函数。
  - 环境光 / 失真训练（envmap/distortion）、DLSS、多 GPU、foveated rendering 等高级选项。
  - 与 GUI/交互有关的 CLI 参数、路径、回调。

- **验证哲学**
  - 每个 milestone 都必须：
    - 有 **明确可执行的命令**（构建/运行/对比日志）用于验证。
    - 尽量设计 **和完整版可交叉验证** 的指标（loss 曲线、模型输出、统计信息等）。
  - 最终 minimal 版本的训练效果（loss 下降趋势和最终 PSNR / 视觉效果）要与当前完整版在同一数据集上 **接近或一致**。

---

## 整体阶段划分（Top-level Milestones）

1. **Phase 0：基线锁定与测试规范**
2. **Phase 1：最小训练路径梳理（call graph & 数据流）**
3. **Phase 2：`ngp-minimal/` 框架搭建 + 与完整版共享库**
4. **Phase 3：NeRF-synthetic 训练功能完整迁移（0 裁剪）**
5. **Phase 4：结构保持下的逐层裁剪（类/函数成员精简）**
6. **Phase 5：更深层的算法/数据结构裁剪与清理**
7. **Phase 6：最终一致性验证与性能/复杂度评估**

下面按阶段详细拆分每个 milestone 和可验证的中间结果。

---

## Phase 0：基线锁定与测试规范

目的：先给「完整版」建立稳定的基线测试与数据集/配置，后续所有 minimal 的验证都统一对照这条基线。

### Milestone 0.1：标准数据集与配置确定

- 选择至少 2 个 nerf-synthetic 场景（建议）：
  - `lego`（结构复杂、遮挡多）。
  - `chair` 或 `drums` 作为第二个对照场景。
- 统一目录结构假定：
  - `D:\datasets\nerf_synthetic\lego\transforms_train.json`  
    `D:\datasets\nerf_synthetic\chair\transforms_train.json` 等。
- 固定 config：
  - 推荐使用现有 `configs/nerf/base.json` 作为默认配置。

**验证方式**
- 使用当前 **完整版** 可执行程序：
  - `./ngp-baseline-nv-app --scene D:\datasets\nerf_synthetic\lego --config configs/nerf/base.json`
- 记录：
  - 初始若干 iteration 的 loss 值（如每 100 iter 记录一次）。
  - 若已有导出/截图脚本，保存几张中间渲染结果作为后续视觉对照。

### Milestone 0.2：日志与比较工具约定

- 约定训练过程中的最小对比信息：
  - 每 N 步（如 100 或 500）输出：`iter=<n> loss=<scalar>`。
  - 可选：PSNR / step time。
- 设计简易比较方式（后续可实现）：
  - `tools/compare_logs.py full.log minimal.log`  
    输出：
    - loss 曲线的差异统计（如 L1/L2 差）。
    - 若差异超过阈值（例如 5%）则标记为不一致。

**中间结果**
- 有一份记录完整版训练日志的基线文件（例如 `logs/full_lego_base.log`）。
- 文档中约定「之后所有 minimal 验证都要能生成对应 minimal 日志，与此基线比较」。

---

## Phase 1：最小训练路径梳理（call graph & 数据流）

目的：清楚知道「从 main 到 NeRF 训练」所有必须经过的函数、类和结构体成员，为后续裁剪提供边界。

### Milestone 1.1：静态调用链分析

- 针对当前项目中的 NeRF 训练路径，梳理：
  - `main.cu`：
    - CLI 参数解析（`--scene`, `--config`, `--snapshot`, `--load_snapshot`, `--no-train`）。
    - 创建 `Testbed` / 设置模式 / 加载数据 / 进入 frame 循环。
  - `Testbed::frame` → `Testbed::train_and_render` → `Testbed::train` → `Testbed::training_prep_nerf` / `train_nerf_step`。
  - `Testbed::load_nerf` → `ngp::load_nerf`（`nerf_loader.cu`）→ `NerfDataset::set_training_image`。
  - NeRF 网络初始化路径：
    - `Testbed::reset_network` → 构造 encoding + network + optimizer 等。
- 输出一份简要的调用链图（可以画在这个 roadmap 文档或单独的注释文件中）。

**验证方式**
- 无代码改动，仅通过阅读 +（可选）`rg` / IDE call hierarchy。
- 结果：列出一个「必须保留」函数列表（例如 `Testbed::frame`, `Testbed::load_nerf`, `Testbed::update_density_grid_nerf`, `train_nerf_step` 等）。

### Milestone 1.2：数据流与关键状态成员清单

- 对于 `Testbed` / `Testbed::Nerf` / `NerfDataset`：
  - 标记训练过程中必用字段：
    - `NerfDataset`：`metadata`, `xforms`, `pixelmemory`, `n_images`, `aabb_scale`, `offset`, `scale`, `render_aabb`, `render_aabb_to_local`, `up` 等。
    - `Testbed::Nerf::Training`：batch 大小、loss 类型、density grid、优化器等。
  - 标记「在 NeRF-synthetic 单机训练中完全不访问」的字段，为之后精简提供候选列表。

**验证方式**
- 目前只产出清单，不改代码。
- 清单将成为 Phase 4/5 的裁剪依据。

---

## Phase 2：`ngp-minimal/` 框架搭建 + 共享库

目的：在不修改现有工作流的前提下，先搭建一个新的可执行目标 `ngp-minimal-app`，**初期完全复用现有库行为**，确保构建/运行环境和 CLI 一致。

### Milestone 2.1：目录与 CMake 架构

- 新建目录结构：
  - `ngp-minimal/`
    - `CMakeLists.txt`
    - `main_minimal.cu`（后续可扩展为 `src/` 目录）
- CMake 设计：
  - 初期编译一个新可执行 `ngp-minimal-app`，**直接链接现有 `ngp-baseline-nv`/`ngp-resources` 静态库**。
  - `ngp-minimal-app` 的 `main_minimal.cu` 内部实现一个 `minimal_main_func(arguments)`，其 CLI 和训练主逻辑与当前 `main.cu::main_func` 保持对齐（支持 `--scene`、`--config`、`--snapshot/--load_snapshot`、`--no-train` 等）。

**验证方式**
- `cmake --build build --config Release -j` 后确保：
  - 新增 target：`ngp-minimal-app` 成功生成，可执行路径为 `build/ngp-minimal/ngp-minimal-app.exe`。
  - 使用与完整版相同的基础 CLI 调用（至少 `--scene` / `--config`）能够成功加载 `data/nerf-synthetic/lego` 等数据集并进入训练/渲染循环（例如可先加 `--no-train` 仅验证加载流程）。
- 与 Phase 0 的基线对比：
  - `ngp-minimal-app` 与 `ngp-baseline-nv-app` 在同样 command 下训练，应产生非常接近的 loss 曲线（如果完全复用 Testbed，则应该完全一致）。

### Milestone 2.2：CLI 对齐与行为对照测试

- 对 `ngp-minimal-app` 的 CLI 进行限制与对齐：
  - 保留：`--scene`, `--config`, `--snapshot`, `--load_snapshot`, `--no-train`。
  - 对于不支持的 GUI/VR 参数：
    - 要么完全忽略（但打印 warning）。
    - 要么在 minimal 里不给出这些选项（推荐）。
- 测试用例：
  1. 训练 + 保存 snapshot：
     - `ngp-baseline-nv-app --scene ... --config ... --snapshot full_lego.msgpack`
     - `ngp-minimal-app     --scene ... --config ... --snapshot minimal_lego.msgpack`
  2. 从 snapshot 恢复：
     - 两者分别用各自 snapshot 继续训练若干步。

**验证方式**
- 检查日志：
  - 前若干 step 的 loss 是否基本一致。
  - 继续训练 1000 步左右后，loss 差异是否在可接受范围内。
- 检查 snapshot 大体结构（可用小工具打印部分 JSON/msgpack 元数据，非必须立即实现）。

---

## Phase 3：NeRF-synthetic 训练完整迁移（0 裁剪）

目的：在 `ngp-minimal/` 下 **复制** 完整训练路径相关代码（而不是链接现有实现），保证 minimal 目录内部已经有一份完整但尚未裁剪的 NeRF 训练实现。

### Milestone 3.1：核心头文件与实现复制

- 按 Phase 1 的调用链，将以下模块复制到 `ngp-minimal/`：
  - `include`：
    - `common.h`, `common_device.cuh`, `common_host.h`, `bounding_box.cuh`, `random_val.cuh`（如需要）。
    - `nerf.h`, `nerf_device.cuh`, `nerf_network.h`, `nerf_loader.h`.
    - `render_buffer.h`, `trainable_buffer.cuh`（若还依赖 envmap 可暂时带上）。
    - 精简版 `testbed.h`：起初直接复制全文。
  - `src`：
    - `testbed.cu`, `testbed_nerf.cu`, `nerf_loader.cu`, `render_buffer.cu`, `common_host.cu` 等。
- 初期复制版本 **不做任何删减**，仅调整命名空间/包含路径以适配 `ngp-minimal` 的 CMake。

**验证方式**
- `ngp-minimal-app` 改为 **仅使用 `ngp-minimal/` 下的实现**：
  - 不再链接原始 `ngp-baseline-nv` 静态库（或仅复用 vendor/tcnn 等第三方库）。
- 重新运行 Phase 0 的基线训练命令：
  - 期望行为：loss 曲线与原始实现几乎重合（浮点误差级别差异）。

### Milestone 3.2：数据加载 & snapshot 兼容性验证

- 使用相同数据集和 config：
  - 分别用原版和 minimal 版从零训练至固定迭代数（如 10k iter），分别保存 snapshot。
- 测试：
  - 用原版加载 minimal 生成的 snapshot，看能否继续训练/渲染。
  - 用 minimal 版加载原版 snapshot，同样继续训练/渲染。

**验证方式**
- 两个方向都能成功训练且没有崩溃或明显异常（loss 突变、图像完全错误等）。
- 若 snapshot 格式有细微差异，可在后续 Phase 中决定是否需要完全兼容；但至少 minimal 自己的 snapshot 循环（保存→加载→继续）要稳定。

---

## Phase 4：结构保持下的逐层裁剪（类/函数成员精简）

目的：在 `ngp-minimal/` 内部开始 **按字段/函数级别裁剪**，但保持类/函数名称和主要调用关系不变，以方便对照和后续维护。

### Milestone 4.1：`NerfDataset` 与 loader 精简（与 RoadmapNEW Phase 3 对齐）

- 在 `ngp-minimal` 中进一步精简 `NerfDataset` 和 `load_nerf`：
  - 删除（或不复制）非 NeRF-synthetic 相关字段：
    - Mitsuba 相关字段、白/黑透明度特例、mask_color 动态 mask、depth 图加载、per-pixel rays 等。
  - 保留：
    - `metadata`（内含 `focal_length`, `principal_point`, `lens`, `resolution`）。
    - `xforms`, `paths`, `render_aabb`, `render_aabb_to_local`, `up`, `offset`, `scale`, `aabb_scale`, `n_images`。
- Loader 中仅支持：
  - 单一 `frames` 数组 + `file_path` + `transform_matrix`。
  - optional 的 `render_aabb`, `aabb_scale`, `offset`, `scale`, `sharpen` 等。

**验证方式**
- 为 `ngp-minimal` 写一个小型单元测试/工具（或简单 main）：
  - 给定相同 JSON 路径，分别调用：
    - 原版 `ngp::load_nerf`（通过原库）。
    - minimal 版 `load_nerf`。
  - 比较：
    - `n_images` 是否一致。
    - 每张图片的 `resolution` / `focal_length` / `principal_point` / `lens.mode` 是否一致。
    - `xforms` 的平移和旋转部分是否接近（允许浮点误差）。
- 再通过训练对比：
  - 用 minimal 版完整训练一次（同 Phase 0），loss 曲线应与原版 loader 路径非常接近。

### Milestone 4.2：`Testbed` 结构裁剪（NeRF-only）

- 在 `ngp-minimal` 的 `testbed.h` / `testbed.cu` 中：
  - 删除 SDF/Image/Volume 等 mode 的枚举值及分支（但 `ETestbedMode::Nerf` 保持原值以便未来互转）。
  - 去除 GUI 相关成员（窗口句柄、输入事件、键盘状态等）。
  - 将多模式逻辑（例如 `set_mode` 内的 `switch(mode)`）缩减为仅处理 `Nerf` 与 `None`。
- 保留：
  - 所有 NeRF 训练相关成员（`m_nerf`, `m_aabb`, `m_training_step` 等）。
  - 与 snapshot/配置直接相关的字段（如 `m_network_config_path`, `m_seed` 等）。

**验证方式**
- 再次构建 `ngp-minimal-app` 并完整训练：
  - 验证训练是否稳定。
  - 再与原版 loss 曲线对比，确保无明显退化（期望差异来自随机种子/浮点顺序，而非逻辑减少）。

### Milestone 4.3：训练选项与损失函数精简

- 针对 `Testbed::Nerf::Training` 中的选项：
  - 保留：
    - 主要 loss 类型（可固定为 `Huber` 或 `L2`，与 `configs/nerf/base.json` 对齐）。
    - batch size、learning rate、多级网格参数。
  - 删除或锁死：
    - 训练模式枚举 `ETrainMode` 中除标准 NeRF 外的其他模式（Rfl/RflRelax 等）。
    - 与 GUI 调试相关的 error overlay / sharpness map / error map 更新策略（如果不再使用）。

**验证方式**
- 配置简化后重新训练：
  - 使用相同 config（必要时同步调整 minimal 版 config，去除不再支持的字段）。
  - 校验 loss 曲线、训练速度。

---

## Phase 5：更深层的算法/数据结构裁剪与清理

目的：在不破坏行为的前提下，进一步移除「永远不会被调用」的底层辅助函数、工具类和 CUDA kernels，从而达到真正的最小实现。

### Milestone 5.1：CUDA kernel 级裁剪

- 在 `testbed_nerf.cu`、`fused_kernels/train_nerf.cuh` 等文件中：
  - 标记仅在 SDF/Image/高级调试模式中使用的内核，确认在 NeRF-synthetic 流程中从未触发。
  - 删除这些内核及调用点。

**验证方式**
- 使用 `rg` 检查删除的符号是否仍有引用。
- 重新构建并运行训练，确保行为仍然正确。

### Milestone 5.2：数学/辅助工具裁剪

- 针对 `common_device.cuh` 中复杂的镜头模型、投影变换等：
  - 保留仅用于 NeRF-synthetic 的 pinhole / OpenCV 模型。
  - 删除暂不需要的 FTheta、LatLong、复杂畸变模型（如果未被 nerf-synthetic 使用）。

**验证方式**
- 通过 Phase 4 中的 dataset 对比工具，确认：
  - `lens.mode` 与最小实现下仍然正确。
  - 实际渲染出的视角与原版一致（必要时通过在同一 iteration dump 图像进行目测或 PSNR 对比）。

### Milestone 5.3：配置文件与 CLI 进一步收缩

- 为 `ngp-minimal` 单独维护一套 `configs/nerf-minimal/*.json`：
  - 只保留 minimal 代码分支会用到的字段。
  - 去掉 envmap/distortion/debug-only 等配置项。
- 精简 `ngp-minimal-app` CLI：
  - 明确列出支持参数及默认值。
  - 对所有不支持的参数直接报错或忽略并提示。

**验证方式**
- 使用新 config 跑完整训练：
  - 确认训练稳定。
  - 与 Phase 0 基线对比：loss 曲线略有差异是允许的，但最终收敛性能不应明显变差。

---

## Phase 6：最终一致性验证与性能/复杂度评估

目的：确认 `ngp-minimal` 达到目标——**功能正确、结构清晰、冗余最小**，并给出对比数据。

### Milestone 6.1：功能一致性回归测试

- 测试矩阵：
  - 数据集：`lego`, `chair`（至少两个）。
  - 配置：`base.json` / `small.json`（一个正常规模，一个小网络）。
  - 命令：
    - 原版：`ngp-baseline-nv-app ...`
    - minimal：`ngp-minimal-app ...`
- 对比指标：
  - loss 曲线（每 1000 iter 记录一次）。
  - 训练总时间（粗略对比）。
  - 若有渲染导出：比较若干视角 PSNR / 目测。

**验证方式**
- 使用前面设计的 log 比较脚本或手工对比，给出结论：
  - 是否在统计意义上「等价」。

### Milestone 6.2：代码规模与依赖复杂度对比

- 对比项：
  - 源文件数量（`src/` 与 `include/`）。
  - 代码行数（可用 `cloc` 或简单统计）。
  - 构建时间。
  - 依赖库数量（仅 tcnn + logging + filesystem + json 等）。

**验证方式**
- 记录一份对比表（Full vs Minimal）。
- 这份表可以放回本 Roadmap 或另建 `docs/ngp-minimal-report.md`。

### Milestone 6.3：维护与演进策略

- 约定后续维护方式：
  - 当完整版 NeRF pipeline 有算法更新时，如何「选择性」同步到 `ngp-minimal`（例如只对某些核心函数做 cherry-pick）。
  - 最好在 `ngp-minimal` 中加入少量注释指明「此处对应完整版的哪个函数/版本」，便于 diff。

---

## 测试样例设计建议汇总

在上面的各个阶段已经提到了一些测试，这里集中列出可实现的测试样例，方便逐步实现：

1. **训练日志对比测试（核心）**
   - 输入：相同数据集 + config。
   - 输出：两份日志文件，比较 iteration-loss 序列。

2. **NerfDataset 等价性测试**
   - 输入：同一 `transforms_*.json`。
   - 输出：原版和 minimal 的 `NerfDataset` 关键信息（`n_images`, `resolution`, `focal_length`, `principal_point`, `up`, `aabb_scale` 等），要求数值一致或浮点误差级接近。

3. **Snapshot 互操作测试**
   - 从双方分别保存 snapshot，再互相加载继续训练，确保都不会崩溃且 loss 连续。

4. **渲染图像对比（可选加分项）**
   - 在固定 iteration（如 10k iter）导出若干视角图像，使用 PSNR/SSIM 或目测对比。

5. **极小网络配置测试**
   - 使用极小网络配置（如 `base_0layer.json`）快速验证训练 + 兼容性，便于在本地快速 回归。

---

## 小结

上述 Roadmap 将「从现有完整版抽取最小 NeRF-synthetic 训练实现」拆解为多个可独立验证的阶段，每个阶段都有：

- 清晰的目标（如：迁移完整实现、精简特定结构、删除未用 CUDA kernel 等）。
- 明确的中间产物（新的可执行程序、简化的 `NerfDataset`、独立的 config 等）。
- 与完整版交叉验证的手段（日志、数据结构比较、snapshot 互操作、图像对比）。

按照该 Roadmap 推进，可以在保持可控风险和可验证性的前提下，逐步得到一个结构相似、依赖更少、代码量显著缩减的 `ngp-minimal`，用于专注 NeRF-synthetic 训练的教学、研究或部署场景。 
