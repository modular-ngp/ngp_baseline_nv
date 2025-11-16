# ngp-baseline-nv 精简 Roadmap（仅保留 nerf‑synthetic 训练）

## 整体目标与保留内容

- 目标：让仓库尽可能精简，只保留从命令行训练 nerf‑synthetic（Blender NeRF）数据集所需的最小代码路径。
- 保留主线调用链：
  - `main.cu: main_func`（命令行解析）
  - → `Testbed` 构造与模式设置
  - → `Testbed::load_training_data`（内部调用 `load_nerf`）
  - → `Testbed::reload_network_from_file`
  - → `while (testbed.frame())`
  - → `Testbed::train_and_render → Testbed::train_nerf` 及相关核函数。
- 尽量只做删除操作，避免大范围重构；优先删除 GUI/VR/Python/SDF/Image/Volume 等与 nerf‑synthetic 训练无关的部分。

---

## 阶段 1：锁定“必须保留”的训练路径

1. 梳理并理解核心调用链
   - 从 `main.cu` 入口开始，跟踪到 `Testbed::frame` 内部的训练流程。
   - 确认无 GUI 模式下（`--no-gui`，或未定义 `NGP_GUI`）仍能正常训练。

2. 初步“白名单”文件集合（建议保留）
   - 入口与模式选择：
     - `main.cu`
     - `include/neural-graphics-primitives/common_host.h`
     - `src/common_host.cu`
   - NeRF 核心：
     - `src/testbed_nerf.cu`
     - `include/neural-graphics-primitives/nerf.h`
     - `include/neural-graphics-primitives/nerf_device.cuh`
     - `include/neural-graphics-primitives/nerf_network.h`
   - 数据加载：
     - `src/nerf_loader.cu`
     - `include/neural-graphics-primitives/nerf_loader.h`
   - 通用基础：
     - `include/neural-graphics-primitives/common.h`
     - `include/neural-graphics-primitives/common_device.cuh`
     - `include/neural-graphics-primitives/common_host.h`
     - `include/neural-graphics-primitives/bounding_box.cuh`
     - `include/neural-graphics-primitives/render_buffer.h`
     - `src/render_buffer.cu`
     - `include/neural-graphics-primitives/adam_optimizer.h`
     - `include/neural-graphics-primitives/trainable_buffer.cuh`
     - `include/neural-graphics-primitives/thread_pool.h`
     - `src/thread_pool.cpp`
     - `include/neural-graphics-primitives/tinyexr_wrapper.h`
     - `src/tinyexr_wrapper.cu`
     - `include/neural-graphics-primitives/tinyobj_loader_wrapper.h`
     - `src/tinyobj_loader_wrapper.cu`
     - `include/neural-graphics-primitives/triangle*.cuh`
     - `src/triangle_bvh.cu`
   - 配置：
     - `configs/nerf/*.json`

3. 验证训练路径
   - 在 CMake 中不定义 `NGP_GUI`/`NGP_PYTHON`/`NGP_VULKAN`。
   - 使用 nerf‑synthetic 数据集测试：
     - 示例命令：`./ngp-baseline-nv-app --scene <dataset_root> --config configs/nerf/base.json --no-gui`
   - 确认训练 loop 正常运行后，再进入后续删除阶段。

---

## 阶段 2：去掉 GUI / VR / Python（只删代码块，避免重构）

### 2.1 收紧头文件：`include/neural-graphics-primitives/testbed.h`

1. 删除 GUI/交互相关接口声明
   - 删除或移除以下声明（仅 GUI/VR/交互使用）：
     - `update_imgui_paths`
     - `overlay_fps`
     - `imgui`
     - `mouse_drag`
     - `mouse_wheel`
     - `keyboard_event`
     - `want_repl`
     - 窗口/VR接口：
       - `init_window`
       - `destroy_window`
       - `init_vr`
       - `update_vr_performance_settings`
       - `begin_frame`
       - `handle_user_input`
       - `begin_vr_frame_and_handle_vr_input`
       - `draw_gui`
       - `gather_histograms`
       - `vr_to_world` 等只用于 VR 的接口。

2. 删除 `NGP_GUI` 区域
   - 去掉头部：
     - `#ifdef NGP_GUI`
       - `#include <neural-graphics-primitives/openxr_hmd.h>`
   - 在类成员中删掉：
     - `GLFWwindow* m_glfw_window`
     - `SecondWindow` 结构体及 `m_second_window`
     - `std::unique_ptr<OpenXRHMD> m_hmd`
     - `OpenXRHMD::FrameInfoPtr m_vr_frame_info`
     - `m_vr_use_depth_reproject` 等 VR 状态
     - `m_render_window`、`m_gather_histograms` 等 GUI 状态。
   - 删除与 GUI 渲染纹理相关的字段：
     - `m_pip_render_texture`
     - `m_rgba_render_textures`
     - `m_depth_render_textures`

3. 删除 Python bindings（`NGP_PYTHON`）
   - 头文件顶部：
     - 删除 `#ifdef NGP_PYTHON` 中的 `pybind11` 相关 include。
   - 类内部：
     - 删除 `#ifdef NGP_PYTHON` 中所有 pybind 接口，如：
       - `compute_marching_cubes_mesh`
       - `render_to_cpu`
       - `render_to_cpu_rgba`
       - `view`
       - `screenshot`
       - `override_sdf_training_data` 等。

4. 清理不再需要的前向声明
   - 删除 `struct GLFWwindow;` 与 `class GLTexture;` 前向声明。
   - 确保剩余代码不再引用这些类型。

### 2.2 实现删除：`src/testbed.cu`

1. 删除 GUI/ImGui/GLFW/OpenGL 相关 include
   - 顶部 `#ifdef NGP_GUI` 中：
     - ImGui, ImGuizmo, GL/GLFW, `cuda_gl_interop` 相关 include 整块删除。

2. 删除 GUI 辅助函数
   - `#ifdef NGP_GUI` 区域中：
     - 删除 `imgui_colored_button`
     - 删除 `Testbed::overlay_fps`
     - 删除 `Testbed::imgui`（大块 ImGui 窗口布局、按钮、滑条、菜单等）
     - 删除所有键盘/鼠标回调处理逻辑。

3. 删除窗口及 VR 初始化/帧循环逻辑
   - 删除整个 `Testbed::init_window` 实现。
   - 删除 `Testbed::destroy_window`、`Testbed::init_vr`、`Testbed::update_vr_performance_settings` 实现。
   - 删除 `begin_frame`、`handle_user_input`、`begin_vr_frame_and_handle_vr_input`、`draw_gui`、`gather_histograms` 等实现。

4. 简化 `Testbed::frame()`
   - 在 `Testbed::frame()` 中：
     - 删除 `#ifdef NGP_GUI` 块中关于 `m_render_window`、`begin_frame`、`draw_gui`、`m_hmd` 的逻辑。
   - 保留并简化为：
     - 处理 `m_camera_path`（如果需要保留相机路径渲染，可观望）。
     - 更新 `skip_rendering`。
     - 调用 `train_and_render(skip_rendering)` 完成训练与渲染（仅在 CUDA buffer 上，无 GUI）。
   - 若不再使用，可以删掉 `m_gui_redraw` 及 `redraw_gui_next_frame()` 及其调用。

### 2.3 渲染缓冲去 GUI 化：`render_buffer.{h,cu}`

1. `include/neural-graphics-primitives/render_buffer.h`
   - 删除 OpenGL 类型别名：
     - `typedef unsigned int GLenum;`
     - `typedef int GLint;`
     - `typedef unsigned int GLuint;`
   - 删除 `#ifdef NGP_GUI` 中的 `GLTexture` 类定义。
   - 确保 `CudaRenderBuffer` 只依赖 `SurfaceProvider` 的 CUDA 实现（`CudaSurface2D`）。

2. `src/render_buffer.cu`
   - 删除 `#ifdef NGP_GUI` 中的全部实现：
     - GL/GLFW/CUDA interop include。
     - `GLTexture`、`GLTexture::CUDAMapping` 的实现。

3. Testbed 使用纯 CUDA 目标
   - 在 Testbed 初始化视图时：
     - 将构造逻辑替换为使用 `std::make_shared<CudaSurface2D>()` 作为 RGBA 以及深度 SurfaceProvider。
   - 这样可彻底断开与 OpenGL/GLFW 的依赖，只使用 CUDA surface。

### 2.4 删除 Python 支持（实现部分）

- 确保构建系统中不定义 `NGP_PYTHON`。
- 删除源文件中任何 `#ifdef NGP_PYTHON` 块（如果存在），只保留 C++/CUDA 路径。

---

## 阶段 3：仅保留 NeRF 模式，删除 SDF / Image / Volume 模式

### 3.1 模式枚举与公共工具（保留接口，清空实现分支）

1. `include/neural-graphics-primitives/common.h`
   - 可以先保留 `ETestbedMode` 中所有枚举值，减少 API 变动。
   - 后续在实现中只处理 `Nerf` 和 `None`，其他模式不再在逻辑分支中出现。

### 3.2 Testbed 只支持 NeRF 模式

1. `src/testbed.cu` 中的 `Testbed::load_training_data`
   - 已经只调用 `load_nerf(path)`，注释掉的 SDF/Image/Volume 分支可以完全删除。

2. `Testbed::set_mode`
   - 保留 `Nerf` 分支逻辑。
   - 对非 `Nerf`/`None` 模式：
     - 可以直接 `throw std::runtime_error{"Invalid testbed mode"}`，或不再提供设置入口。

3. 训练和网络维度函数
   - 删除以下函数的实现与声明：
     - `train_sdf`
     - `train_image`
     - `train_volume`
     - `training_prep_sdf`
     - `training_prep_image`
     - `training_prep_volume`
     - `network_dims_sdf`
     - `network_dims_image`
     - `network_dims_volume`
   - 在 `network_dims()` 等函数中：
     - 只保留 `ETestbedMode::Nerf` 分支调用 `network_dims_nerf()`。
   - 在 `train_and_render` 或相关 switch/mode 分支中：
     - 删除所有 SDF/Image/Volume 分支（训练和渲染）。

4. SDF/Image/Volume 相关渲染和 IoU 等逻辑
   - 若不再需要这些功能（例如 IoU 评估、mesh 导出等）：
     - 删除相关函数，如：
       - `compute_and_save_marching_cubes_mesh`
       - SDF 渲染选项与 IoU 在线计算部分。
     - 或保留仅用于 NeRF 的 mesh 提取部分，根据实际需求选择。

5. `include/neural-graphics-primitives/testbed.h`
   - 删除以下结构体与成员：
     - `struct Sdf` 及 `m_sdf`
     - `struct Image` 及 `m_image`
     - `struct Volume` 及 `m_volume`
   - 删除所有仅在这些模式下使用的成员函数声明：
     - SDF 训练/渲染选项相关函数。
     - Image/Volume 加载与渲染相关函数。

### 3.3 移除 SDF / Image / Volume 依赖文件

1. 头文件
   - 删除（在确认无引用后）：
     - `include/neural-graphics-primitives/sdf.h`
     - `include/neural-graphics-primitives/sdf_device.cuh`
     - `include/neural-graphics-primitives/fused_kernels/trace_sdf.cuh`
   - 使用 `rg` 确认这些文件只被 Testbed 中已删除的路径引用。

2. 源文件
   - `src/triangle_bvh.cu` 中如果有只服务于 SDF 模式的函数，可精简：
     - 保留 NeRF 仍可能用到的三角形 BVH 基础功能。
     - 删除只被 SDF 相关代码引用的分支（例如某些 `EMeshSdfMode` 的特殊逻辑）。

3. 配置目录
   - 删除：
     - `configs/image`
     - `configs/sdf`
     - `configs/volume`
   - 并确保 `Testbed::find_network_config` 只在 `configs/nerf` 下解析。

---

## 阶段 4：围绕 nerf‑synthetic 的针对性精简

### 4.1 `nerf_loader` 只保留 nerf‑synthetic 路径

1. 通读 `src/nerf_loader.cu`，找出不同数据格式分支
   - 识别以下特性路径：
     - 原生 NeRF Blender synthetic（`transforms_*.json` + PNG/EXR）。
     - LLFF、NSVF、其他结构化/稀疏数据集。
     - 各种 depth/mask/特殊相机模型等扩展。

2. 保留原始 NeRF synthetic 数据加载路径
   - 保留解析 `transforms_*.json`、图像加载和基本 metadata 填充的逻辑。
   - 对以下功能进行删除或降级：
     - NSVF “white=transparent” / “black=transparent” 特殊处理。
     - mask_color 功能。
     - depth 文件（如果 nerf‑synthetic 不需要）。
     - light_dirs、额外 latent 维度等仅特定数据集使用的扩展。

3. 精简 `NerfDataset` 结构（`nerf_loader.h`）
   - 保留：
     - `metadata`, `xforms`, `paths`
     - `render_aabb`, `render_aabb_to_local`, `up`, `offset`, `scale`, `aabb_scale`
     - `n_images`、`envmap_data`（视需求保留）
   - 删除或不再使用：
     - 只在非 synthetic 数据集中使用的字段（例如强依赖 depth/mask/额外维度的字段），视 `testbed_nerf` 依赖关系而定。
   - 保留必要的空间变换函数（`nerf_matrix_to_ngp` 等），因训练时会用到。

### 4.2 精简 `testbed_nerf.cu` 的训练选项

1. 聚焦核心 NeRF 训练流程
   - 保留以下关键功能：
     - 射线采样、density grid 更新、ray marching。
     - 基础损失函数与优化器设置（Adam）。
     - 必需的辅助内核，如 error map（若用于采样策略）。
   - 删除以下扩展功能（若确定不用）：
     - 某些特定论文相关的模式/flag（例如额外的正则或复杂损失组合）。
     - 额外 latent 维度逻辑（`n_extra_learnable_dims` 等），如果仅为扩展数据集而非 nerf‑synthetic 服务。

2. 约简训练配置参数
   - 将训练相关的配置（学习率、batch size 等）集中依赖 `configs/nerf/*.json`。
   - 若有完全 GUI/交互用的控制项（如某些 debug 可视化专用变量），可删除。

### 4.3 简化命令行接口：`main.cu`

1. 保留训练相关的命令行参数
   - `--scene`：数据路径。
   - `--config`/`--network`：网络配置路径。
   - `--snapshot`/`--load_snapshot`：可选快照加载。
   - `--no-train`：禁止自动训练启动（可选保留）。
   - `--no-gui`：保留但不再有实际 GUI 功能，其意义仅为“纯命令行模式”。

2. 删除纯 GUI/交互用途的参数
   - `--width`, `--height`：GUI 分辨率设置。
   - `--vr`：启动 VR 模式。
   - 其他仅在 GUI 代码中使用的 flags（通过搜索参数变量引用来确认）。

3. （可选）增加更简化入口
   - 创建一个更简洁的 CLI 流程（可选，不是必需）：
     - 固定训练模式为 NeRF。
     - 接收 `--scene` 和 `--config`，内部直接构造 `Testbed(ETestbedMode::Nerf, scene, config)`。
     - 运行到收敛或固定迭代数后退出。

---

## 阶段 5：vendor / 构建脚本 / 其他资源清理

### 5.1 vendor 目录精简（不改接口，只删无用文件）

1. `vendor/args`
   - 保留：
     - `args.hxx`（主头文件）。
   - 删除：
     - `examples/`, `test/`, `packaging/` 及其他构建脚本/CI 配置/README 等不参与编译的文件。

2. 其他 vendor
   - `vendor/tinyobjloader`, `vendor/stb_image`, `vendor/tinyexr`, `vendor/tinylogger` 等：
     - 用 `rg` 确认实际 include 的头/源文件。
     - 保留被 include 的最小子集。
     - 删除其 docs、examples、tests、打包脚本、CI 配置等。

### 5.2 构建脚本与 vcpkg

1. vcpkg 下载逻辑
   - 如希望进一步简化依赖管理：
     - 在 `CMakeLists.txt` 中移除 `include(cmake/setup_vcpkg.cmake)`。
     - 改为使用本地安装的依赖（例如用户自行准备 vcpkg 或系统包管理器提供的库）。
   - 若接受自动下载 vcpkg，则可暂时保持不动，它只影响构建，不影响训练逻辑。

2. CMake 精简
   - 确保 `add_library(ngp-baseline-nv ...)` 中只列出精简后的源文件。
   - 删除被移除文件对应的编译项（例如 SDF/GUI 相关源文件）。

### 5.3 其他资源与文档

1. RTC 缓存
   - `rtc/cache`：
     - 保留目录结构，用 `.gitignore` 忽略实际生成的缓存。
     - 可以清空仓库中已有缓存文件，减小体积。

2. 配置目录
   - 最终只保留：
     - `configs/nerf`。
   - 删除其他模式配置目录后，检查 `Testbed::find_network_config` 行为，确保只指向 `configs/nerf`。

3. README 更新
   - 更新 `README.md`，简要说明：
     - 项目目的：作为纯 nerf‑synthetic 训练 baseline。
     - 依赖环境、编译命令。
     - 数据准备方法（Blender NeRF 数据集结构）。
     - 典型训练命令行示例。
     - 明确说明：项目已移除 GUI/VR/Python/SDF/Image/Volume 等功能。

---

## 后续执行建议

- 推荐执行顺序：
  1. 阶段 2：先删除 GUI/VR/Python 相关内容（对训练路径影响最小）。
  2. 阶段 3：移除 SDF/Image/Volume 模式与相关文件。
  3. 阶段 4：针对 nerf‑synthetic 对 `nerf_loader` 和 `testbed_nerf` 做针对性精简。
  4. 阶段 5：清理 vendor、多余配置与构建脚本。
- 每完成一个阶段，建议重新编译并跑一次 nerf‑synthetic 训练，确保行为一致。

