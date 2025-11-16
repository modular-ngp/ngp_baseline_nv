# ngp-minimal 实现路线图 (Implementation Roadmap)

> **项目目标**: 在 `ngp-baseline-nv/ngp-minimal/` 目录下，从零开始实现一个**完全独立的、仅支持 NeRF-synthetic 数据集训练的最小版 instant-ngp**。

## 核心理解与设计原则

### 1. 需求理解

根据 `Log.md` 和 `ROADMAP_RETIRE_MINIMAL.md` 的详细分析，我理解您的核心需求为：

- **完全独立实现**: `ngp-minimal/` 不依赖 `../include` 或 `../src`，不链接 `ngp-baseline-nv` 静态库
- **按需搬迁策略**: 仅实现 NeRF-synthetic 训练路径实际用到的功能子集
- **保持接口一致性**: 使用与完整版相同的类名/函数名/调用关系，便于对比验证
- **去除所有冗余**: 不包含 GUI、VR、SDF、Volume、Image 等其他模式
- **交叉验证**: 每个阶段都与完整版对比，确保行为一致

### 2. 实施策略

**"按需搬迁 = 先画子图，再重建"**

1. 从入口出发锁定调用子图（NeRF-synthetic 训练路径）
2. 在 `ngp-minimal/` 中用相同接口名重建这一子图
3. 每一步都用完整版对照验证

## 实施阶段划分

本项目将分为 6 个主要阶段，每个阶段包含若干可验证的里程碑。

---

## Phase 1: 基线确认 & 训练路径调用图梳理 ✅

**目标**: 从完整版出发，锁定 NeRF-synthetic 训练的"使用子图"

### Milestone 1.1: 选定基线场景与配置 ✅

**已完成的分析**:
- 参考场景: `data/nerf-synthetic/lego` (必选), `data/nerf-synthetic/chair` (次选)
- 参考配置: `configs/nerf/base.json`
- 训练命令: `./ngp-baseline-nv-app --scene data/nerf-synthetic/lego --config configs/nerf/base.json --no-gui`

### Milestone 1.2: 核心调用链梳理 ✅

**已识别的主调用路径**:

```
main/wmain
  ↓
ngp::main_func(arguments)
  ├─ ArgumentParser 解析 CLI 参数
  ├─ Testbed testbed;
  ├─ testbed.load_training_data(scene_path)
  │    ↓
  │  set_mode(ETestbedMode::Nerf)
  │    ↓
  │  load_nerf(data_path)
  │    ↓
  │  ngp::load_nerf(json_paths) [nerf_loader.cu]
  │    ↓
  │  load_nerf_post() [初始化 density grid]
  │
  ├─ testbed.reload_network_from_file(config_path)
  ├─ testbed.m_train = !no_train_flag
  │
  └─ while (testbed.frame())
       ↓
     train_and_render(skip_rendering)
       ↓
     train(batch_size)
       ↓
     training_prep_nerf(batch_size) [更新 density grid]
       ↓
     train_nerf_step(batch_size)
       ├─ generate_training_samples_nerf() [采样]
       ├─ network->inference() [前向]
       ├─ compute_loss_kernel_train_nerf() [损失]
       └─ trainer->training_step() [反向+优化]
```

### Milestone 1.3: 数据结构字段使用分析 ✅

**已识别的核心数据结构**:

#### NerfDataset (必需字段)
- ✅ **训练必需**:
  - `std::vector<TrainingImageMetadata> metadata`
  - `GPUMemory<TrainingImageMetadata> metadata_gpu`
  - `std::vector<TrainingXForm> xforms`
  - `std::vector<std::string> paths`
  - `std::vector<GPUMemory<uint8_t>> pixelmemory`
  - `BoundingBox render_aabb`
  - `mat3 render_aabb_to_local`
  - `vec3 up, offset`
  - `float scale`
  - `int aabb_scale`
  - `size_t n_images`

- ❌ **可暂不实现**:
  - `raymemory, depthmemory` (非默认配置)
  - `sharpness_data, envmap_data` (高级功能)
  - `n_extra_learnable_dims, has_light_dirs` (扩展功能)
  - `wants_importance_sampling` (重要性采样，初期可简化)

#### Testbed::Nerf::Training (必需成员)
- ✅ **训练必需**:
  - `NerfDataset dataset`
  - `density_grid` 相关（occupancy grid）
  - `counters_rgb` (训练计数)
  - `loss_type, random_bg_color` 等基础训练配置

- ❌ **可暂不实现**:
  - `error_map` 及 CDF（重要性采样）
  - `train_envmap` 及环境贴图
  - `optimize_extrinsics/distortion/focal_length` (相机优化)

---

## Phase 2: ngp-minimal 工程骨架 + 独立 CMake + CLI

**目标**: 建立独立的可编译工程框架

### Milestone 2.1: 创建独立 CMake 配置

**任务清单**:
- [ ] 创建 `ngp-minimal/CMakeLists.txt`
  - 定义独立目标 `ngp-minimal-app`
  - 链接必要的第三方库（tiny-cuda-nn, CUDA, nlohmann_json, stb_image）
  - **不链接** `ngp-baseline-nv` 库
  - 设置正确的 CUDA 架构和编译选项

- [ ] 创建目录结构:
  ```
  ngp-minimal/
  ├── CMakeLists.txt
  ├── README.md
  ├── include/
  │   └── ngp-minimal/
  │       ├── common.h
  │       ├── common_device.cuh
  │       ├── gpu_memory.h
  │       ├── nerf_loader.h
  │       ├── nerf_network.h
  │       ├── testbed.h
  │       └── ...
  └── src/
      ├── main.cu
      ├── nerf_loader.cu
      ├── testbed.cu
      ├── testbed_nerf.cu
      └── ...
  ```

- [ ] 验证根目录 `CMakeLists.txt` 已添加 `add_subdirectory(ngp-minimal)`

**验证标准**:
- ✅ 成功构建生成 `ngp-minimal-app.exe`
- ✅ 无链接到 `ngp-baseline-nv` 库
- ✅ 编译无错误和警告

### Milestone 2.2: 实现最小 CLI 入口

**任务清单**:
- [ ] 实现 `ngp-minimal/src/main.cu`:
  - 定义 `wmain` (Windows) / `main` (Linux) 入口
  - 使用 `args.hxx` 解析命令行参数
  - 支持参数:
    - `--scene <path>`: NeRF-synthetic 数据路径
    - `--config <path>`: 网络配置 JSON
    - `--snapshot <path>`: 快照加载（可选）
    - `--no-train`: 禁用训练
    - `--help`: 帮助信息
  
- [ ] 定义 `CliOptions` 结构体
- [ ] 实现 `ngp::main_func(arguments)` 框架
- [ ] 占位实现 `app_main(CliOptions)` - 仅打印参数

**验证标准**:
- ✅ `ngp-minimal-app --help` 输出帮助信息
- ✅ `ngp-minimal-app --scene test --config test.json` 正确解析并打印参数

**预计时间**: 2-3 小时

---

## Phase 3: 数据层按需搬迁（NerfDataset + loader）

**目标**: 实现与完整版接口一致但精简的数据加载

### Milestone 3.1: 定义精简版数学和数据结构

**任务清单**:
- [ ] 创建 `ngp-minimal/include/ngp-minimal/common.h`:
  - 基础数学类型: `vec2, vec3, vec4, ivec2, mat3, mat4x3`
  - 枚举: `ETestbedMode, ETrainMode, ELossType, EImageDataType, EDepthDataType`
  - 常量: `NERF_SCALE` 等

- [ ] 创建 `ngp-minimal/include/ngp-minimal/common_device.cuh`:
  - 设备端数学函数
  - `BoundingBox` 类
  - `Ray` 结构
  - `Lens` 结构

- [ ] 创建 `ngp-minimal/include/ngp-minimal/gpu_memory.h`:
  - `GPUMemory<T>` 模板类（CUDA 内存管理）
  - `enlarge()`, `resize()`, `free_memory()` 等方法

- [ ] 创建 `ngp-minimal/include/ngp-minimal/nerf_loader.h`:
  - `struct TrainingImageMetadata`
    - `ivec2 resolution`
    - `vec2 focal_length, principal_point`
    - `Lens lens`
    - `vec4 rolling_shutter`
    - `void* pixels` (GPU 指针)
  
  - `struct TrainingXForm`
    - `mat4x3 start, end`
  
  - `struct NerfDataset` (精简版)
    - 仅包含 Milestone 1.3 中标记为"必需"的字段

**验证标准**:
- ✅ 所有头文件可以独立编译
- ✅ 无依赖 `../include` 的引用

### Milestone 3.2: 实现 NeRF-synthetic 数据加载器

**任务清单**:
- [ ] 实现 `ngp-minimal/src/nerf_loader.cu`:
  - `NerfDataset load_nerf(const std::vector<fs::path>& json_paths, float sharpen_amount = 0.f)`
  - 解析 Blender NeRF-synthetic 格式 JSON:
    - `transforms_train.json`
    - `transforms_test.json`
    - `transforms_val.json`
  - 提取字段:
    - `camera_angle_x` → focal_length
    - `frames[].file_path` → 图像路径
    - `frames[].transform_matrix` → camera-to-world 矩阵
  - 读取 PNG 图像（使用 stb_image）
  - 计算场景 AABB, offset, scale
  - 上传像素数据到 GPU (`GPUMemory<uint8_t>`)
  
- [ ] 实现辅助函数:
  - `read_image()`: 加载 PNG/JPG 到 CPU
  - `compute_aabb()`: 计算场景边界框
  - `nerf_matrix_to_ngp()`: 坐标系转换

**验证标准**:
- ✅ 成功加载 `data/nerf-synthetic/lego`
- ✅ 与完整版对比元数据:
  - `n_images` 一致
  - 每张图像的 `resolution, focal_length, principal_point` 数值误差 < 1e-5
  - `render_aabb, offset, scale, aabb_scale` 一致
  - `xforms` 矩阵数值误差 < 1e-5

**交叉验证工具**:
- [ ] 编写 Python 脚本 `scripts/compare_dataset.py`:
  - 从完整版导出 dataset 元数据到 JSON
  - 从 minimal 版导出 dataset 元数据到 JSON
  - 逐字段对比并报告差异

**预计时间**: 4-6 小时

---

## Phase 4: 网络/采样/渲染核心按需搬迁

**目标**: 实现 NeRF 网络和体渲染核心

### Milestone 4.1: NeRF 网络封装（基于 tiny-cuda-nn）

**任务清单**:
- [ ] 创建 `ngp-minimal/include/ngp-minimal/nerf_network.h`:
  - `template<typename T> class NerfNetwork`
  - 成员:
    - `std::shared_ptr<tcnn::Encoding<T>> m_encoding`
    - `std::shared_ptr<tcnn::Network<T>> m_network`
    - `std::shared_ptr<tcnn::Trainer<float, T, T>> m_trainer`
  - 方法:
    - `NerfNetwork(const nlohmann::json& config, const BoundingBox& aabb)`
    - `void inference(tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<float>& output, cudaStream_t stream)`
    - `void density(tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<float>& output, cudaStream_t stream)`
    - `void training_step(uint32_t batch_size, cudaStream_t stream)`

- [ ] 实现 `ngp-minimal/src/nerf_network.cu`:
  - 从 JSON 配置构建网络
  - HashGrid 编码 + MLP 架构
  - 与 tiny-cuda-nn 的 Encoding/Network/Trainer 集成

**验证标准**:
- ✅ 成功从 `configs/nerf/base.json` 加载网络配置
- ✅ 网络架构与完整版一致（层数、宽度、编码参数）

### Milestone 4.2: 体渲染与训练核心

**任务清单**:
- [ ] 创建 `ngp-minimal/include/ngp-minimal/nerf_kernels.cuh`:
  - Ray marching CUDA kernels
  - Volume rendering 公式实现
  - Loss computation kernels

- [ ] 实现 `ngp-minimal/src/testbed_nerf.cu`:
  - `generate_training_samples_nerf()`:
    - 从 dataset 随机采样 rays
    - 根据 density grid 生成采样点
  
  - `train_nerf_step()`:
    - 批量采样训练 rays
    - 网络前向推理（获取 density 和 color）
    - 体渲染累积 RGB
    - 计算 L2/Huber loss
    - 反向传播并更新网络参数
  
  - `update_density_grid_nerf()`:
    - 更新 occupancy grid
    - 用于加速 ray marching

**验证标准**:
- ✅ 固定随机种子和一小批样本
- ✅ 对比完整版与 minimal 版的前向输出（density, color）
- ✅ 单步训练后 loss 变化趋势合理
- ✅ 数值误差在可接受范围内（考虑浮点精度）

**预计时间**: 8-10 小时

---

## Phase 5: Testbed 与训练循环按需搬迁

**目标**: 实现完整的训练主循环

### Milestone 5.1: 精简版 Testbed 类

**任务清单**:
- [ ] 创建 `ngp-minimal/include/ngp-minimal/testbed.h`:
  - `class Testbed`（仅 NeRF 模式）
  - 成员:
    ```cpp
    struct Nerf {
        struct Training {
            NerfDataset dataset;
            
            // Density grid (occupancy grid)
            GPUMemory<uint8_t> density_grid;
            GPUMemory<float> density_grid_mean;
            uint32_t n_steps_between_updates = 16;
            
            // Training config
            ETrainMode train_mode = ETrainMode::Nerf;
            ELossType loss_type = ELossType::L2;
            bool random_bg_color = true;
            bool linear_colors = false;
            
            // Counters
            GPUMemory<uint32_t> counters_rgb;
        } training;
    } m_nerf;
    
    std::shared_ptr<NerfNetwork<precision_t>> m_network;
    std::shared_ptr<tcnn::Trainer<float, precision_t, precision_t>> m_trainer;
    
    uint64_t m_training_step = 0;
    float m_loss_scalar = 0.0f;
    bool m_train = true;
    uint32_t m_training_batch_size = 1<<18;
    
    ETestbedMode m_testbed_mode = ETestbedMode::Nerf;
    fs::path m_data_path;
    ```
  
  - 方法:
    ```cpp
    void load_training_data(const fs::path& path);
    void reload_network_from_file(const fs::path& config_path);
    void reload_network_from_json(const nlohmann::json& config);
    void load_snapshot(const fs::path& path);
    void save_snapshot(const fs::path& path);
    
    bool frame();  // 主循环入口
    void train(uint32_t batch_size, cudaStream_t stream);
    
    void set_mode(ETestbedMode mode);
    void load_nerf(const fs::path& data_path);
    void load_nerf_post();
    
    // Training
    void training_prep_nerf(uint32_t batch_size, cudaStream_t stream);
    void train_nerf_step(uint32_t batch_size, cudaStream_t stream);
    void update_density_grid_nerf(cudaStream_t stream);
    ```

- [ ] 实现 `ngp-minimal/src/testbed.cu`:
  - `Testbed` 构造/析构
  - `load_training_data()`: 调用 `load_nerf()`
  - `set_mode()`: 设置 NeRF 模式
  - `load_nerf()`: 调用 `ngp::load_nerf()` 加载数据
  - `load_nerf_post()`: 初始化 density grid
  - `reload_network_from_file()`: 解析 JSON 并构建网络
  - `frame()`: 训练主循环
    ```cpp
    bool Testbed::frame() {
        if (m_train) {
            train(m_training_batch_size, stream);
            ++m_training_step;
        }
        return true;  // 简化版不退出
    }
    ```
  - `train()`: 调用 NeRF 训练逻辑
    ```cpp
    void Testbed::train(uint32_t batch_size, cudaStream_t stream) {
        training_prep_nerf(batch_size, stream);
        train_nerf_step(batch_size, stream);
    }
    ```

### Milestone 5.2: 集成训练主循环到 CLI

**任务清单**:
- [ ] 在 `ngp-minimal/src/main.cu` 中实现完整流程:
  ```cpp
  int main_func(const std::vector<std::string>& arguments) {
      // 解析参数
      ArgumentParser parser{...};
      // ... 参数定义 ...
      
      Testbed testbed;
      
      if (scene_flag) {
          testbed.load_training_data(get(scene_flag));
      }
      
      if (snapshot_flag) {
          testbed.load_snapshot(get(snapshot_flag));
      } else if (network_config_flag) {
          testbed.reload_network_from_file(get(network_config_flag));
      }
      
      testbed.m_train = !no_train_flag;
      
      // 训练循环
      while (testbed.frame()) {
          if (testbed.m_training_step % 100 == 0) {
              tlog::info() << "iteration=" << testbed.m_training_step 
                          << " loss=" << testbed.m_loss_scalar;
          }
      }
      
      return 0;
  }
  ```

**验证标准**:
- ✅ 成功运行: `ngp-minimal-app --scene data/nerf-synthetic/lego --config configs/nerf/base.json`
- ✅ 训练循环正常执行
- ✅ Loss 打印格式与完整版一致
- ✅ 与完整版对比前 2000 步的 loss 曲线

**交叉验证**:
- [ ] 在相同配置下运行完整版和 minimal 版
- [ ] 记录前 5000 步的 loss 值
- [ ] 绘制 loss 曲线对比图
- [ ] 允许差异但趋势应一致

**预计时间**: 6-8 小时

---

## Phase 6: 冗余收缩、行为一致性评估

**目标**: 精简代码并全面验证

### Milestone 6.1: 代码审查与清理

**任务清单**:
- [ ] 遍历所有源文件，检查:
  - 未使用的字段/方法
  - 调试代码和注释
  - 死代码路径
  
- [ ] 删除冗余代码:
  - 未引用的辅助函数
  - 未使用的成员变量
  - 过时的注释

- [ ] 统一代码风格:
  - 使用 `.clang-format` 格式化
  - 添加必要的文档注释
  - 统一命名规范

**验证标准**:
- ✅ 所有代码通过静态分析（无警告）
- ✅ 训练功能无回退

### Milestone 6.2: 完整性测试

**任务清单**:
- [ ] 测试场景:
  - **必选**: `lego`
  - **次选**: `chair`
  - **可选**: `drums`, `hotdog`

- [ ] 对比项（完整版 vs Minimal 版）:
  - [ ] 训练 loss 曲线（前 5k 步）
  - [ ] 训练速度（iterations/秒）
  - [ ] GPU 内存占用
  - [ ] 编译时间
  - [ ] 可执行文件大小

- [ ] 可选：渲染质量对比
  - [ ] 如实现了简单渲染，导出测试视角图像
  - [ ] 计算 PSNR/SSIM

**验证标准**:
- ✅ Loss 下降趋势与完整版一致（允许数值差异 < 5%）
- ✅ 训练速度差异 < 10%
- ✅ GPU 内存占用减少或持平

### Milestone 6.3: 文档与复杂度对比

**任务清单**:
- [ ] 统计对比:
  ```
  指标                   完整版      Minimal版    减少比例
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  源文件数量              ~20         ~8          60%
  头文件数量              ~25         ~10         60%
  总代码行数              ~15000      ~3000       80%
  依赖库数量              ~15         ~5          67%
  编译时间（秒）          ~120        ~30         75%
  可执行文件大小（MB）    ~50         ~15         70%
  ```

- [ ] 编写 `ngp-minimal/README.md`:
  - 项目介绍
  - 编译指南
  - 使用示例
  - 与完整版的差异
  - 支持的功能列表
  - 不支持的功能列表

- [ ] 更新根目录 `Log.md`:
  - 记录实施过程
  - 遇到的问题与解决方案
  - 性能对比结果
  - 后续改进方向

**预计时间**: 3-4 小时

---

## 技术决策与约束

### 1. 编码/网络架构
- ✅ 使用与完整版一致的 HashGrid + MLP 架构
- ✅ 依赖 tiny-cuda-nn 提供 Encoding/Network/Trainer
- ✅ 保持网络配置 JSON 格式兼容

### 2. 简化策略（不实现的功能）

**完全不实现**:
- ❌ GUI、渲染窗口、交互式相机
- ❌ VR 支持
- ❌ 多 GPU 训练
- ❌ DLSS
- ❌ SDF、Volume、Image 模式
- ❌ 环境贴图训练
- ❌ 相机参数优化（extrinsics/distortion/focal）
- ❌ Marching Cubes 网格提取

**初期简化，后续可扩展**:
- ⚠️ Error map importance sampling（使用均匀采样）
- ⚠️ 深度监督（只用 RGB loss）
- ⚠️ 渲染输出（可选实现简单测试视角渲染）
- ⚠️ Snapshot 保存/加载（可后期添加）

**必须实现**:
- ✅ NeRF-synthetic (Blender) 数据加载
- ✅ HashGrid 位置编码
- ✅ MLP 网络
- ✅ 体渲染（volume rendering）
- ✅ RGB loss 计算
- ✅ Density grid / Occupancy grid
- ✅ 训练循环与优化器

### 3. 保留接口一致性

**类名一致**:
- `Testbed`
- `NerfDataset`
- `NerfNetwork`
- `TrainingImageMetadata`
- `TrainingXForm`

**函数名一致**:
- `load_training_data()`
- `reload_network_from_file()`
- `frame()`
- `train()`
- `train_nerf_step()`
- `generate_training_samples_nerf()`
- `update_density_grid_nerf()`

**好处**:
- 便于代码对比
- 易于调试（可逐函数对比）
- 后续扩展时可参考完整版

### 4. 依赖库管理

**必需依赖**:
- tiny-cuda-nn (通过 FetchContent)
- CUDA Toolkit
- nlohmann/json
- stb_image (vendor)
- args.hxx (vendor)
- tinylogger (vendor)

**可选依赖**:
- ZLIB (如需压缩)
- tinyexr (如需 EXR 支持)

---

## 验证策略

### 数据层验证
1. 导出完整版 dataset 元数据到 JSON
2. 导出 minimal 版 dataset 元数据到 JSON
3. Python 脚本逐字段对比

### 网络层验证
1. 固定网络初始化种子
2. 相同输入下对比前向输出
3. 允许浮点误差 < 1e-4

### 训练验证
1. 固定所有随机种子
2. 对比前 1000/2000/5000 步的 loss
3. 绘制 loss 曲线
4. 趋势一致性检查

### 视觉验证（可选）
1. 导出相同迭代步的测试视角图像
2. 计算 PSNR/SSIM
3. 目测对比

---

## 风险与应对

### 风险 1: tiny-cuda-nn 接口复杂
**应对**: 
- 先研究完整版使用方式
- 照搬接口调用方式
- 参考 tiny-cuda-nn 官方示例

### 风险 2: CUDA kernel 实现难度大
**应对**:
- 尽量复用 tiny-cuda-nn 的 fused kernel
- 简化自定义 kernel
- 初期可用简单实现，后续优化

### 风险 3: 训练结果不一致
**应对**:
- 分阶段验证
- 每个 milestone 做对比
- 及时发现并修正问题
- 记录所有差异及原因

### 风险 4: 依赖库版本兼容性
**应对**:
- 使用与完整版相同的 vcpkg manifest
- 锁定 tiny-cuda-nn 版本
- 定期测试编译

---

## 时间估算

| 阶段 | 预计时间 | 优先级 |
|------|---------|--------|
| Phase 1: 基线确认 | 已完成 | 高 |
| Phase 2: 工程骨架 | 2-3 小时 | 高 |
| Phase 3: 数据加载 | 4-6 小时 | 高 |
| Phase 4: 网络与训练 | 8-10 小时 | 高 |
| Phase 5: Testbed 集成 | 6-8 小时 | 高 |
| Phase 6: 清理与验证 | 3-4 小时 | 中 |
| **总计** | **23-31 小时** | - |

---

## 下一步行动

**立即开始**: Phase 2 - 工程骨架搭建

具体步骤:
1. 创建 `ngp-minimal/` 目录结构
2. 编写独立 CMakeLists.txt
3. 实现最小 CLI 入口
4. 验证编译通过

---

## 成功标准

项目成功的最终标准:

1. ✅ **独立性**: `ngp-minimal/` 完全独立编译，不依赖完整版
2. ✅ **功能性**: 成功训练 NeRF-synthetic 数据集
3. ✅ **一致性**: 训练结果与完整版基本一致（loss 趋势、收敛速度）
4. ✅ **简洁性**: 代码量减少 70%+ ，文件数量减少 60%+
5. ✅ **可维护性**: 代码清晰，文档完善，易于理解和扩展

---

## 附录: 关键代码结构参考

### A. Testbed 主循环伪代码

```cpp
// 完整版与 minimal 版保持一致的调用结构
int main_func(const std::vector<std::string>& args) {
    Testbed testbed;
    
    // 数据加载
    testbed.load_training_data(scene_path);
    
    // 网络配置
    testbed.reload_network_from_file(config_path);
    
    // 训练循环
    while (testbed.frame()) {
        // 自动调用 train() 和 update()
        log_training_progress();
    }
}

bool Testbed::frame() {
    if (m_train) {
        training_prep_nerf(batch_size, stream);
        train_nerf_step(batch_size, stream);
        ++m_training_step;
    }
    return true;
}
```

### B. 数据加载流程伪代码

```cpp
NerfDataset load_nerf(const std::vector<fs::path>& json_paths) {
    NerfDataset dataset;
    
    for (auto& json_path : json_paths) {
        // 解析 JSON
        auto transforms = parse_json(json_path);
        
        // 提取相机参数
        float focal_length = compute_focal_length(transforms);
        
        // 加载每一帧
        for (auto& frame : transforms["frames"]) {
            auto img = load_image(frame["file_path"]);
            auto xform = parse_transform_matrix(frame["transform_matrix"]);
            
            dataset.metadata.push_back({resolution, focal_length, ...});
            dataset.xforms.push_back(xform);
            dataset.pixelmemory.push_back(upload_to_gpu(img));
        }
    }
    
    // 计算 AABB
    dataset.render_aabb = compute_aabb(dataset.xforms);
    dataset.offset = compute_offset(dataset.render_aabb);
    dataset.scale = NERF_SCALE;
    
    return dataset;
}
```

### C. 训练步伪代码

```cpp
void Testbed::train_nerf_step(uint32_t batch_size, cudaStream_t stream) {
    // 1. 采样训练 rays
    auto rays = generate_training_samples_nerf(
        m_nerf.training.dataset,
        m_nerf.training.density_grid,
        batch_size
    );
    
    // 2. 网络前向
    auto network_output = m_network->inference(rays.coords, stream);
    
    // 3. 体渲染
    auto rgb_pred = volume_rendering(
        rays,
        network_output.density,
        network_output.color
    );
    
    // 4. 计算 loss
    auto loss = compute_loss(rgb_pred, rays.target_rgb);
    
    // 5. 反向传播
    m_trainer->backward_pass(loss, stream);
    
    // 6. 优化器更新
    m_trainer->optimizer_step(stream);
    
    // 7. 更新统计
    m_loss_scalar = loss.mean();
}
```

---

**文档版本**: v1.0  
**创建日期**: 2025-11-17  
**最后更新**: 2025-11-17  
**作者**: GitHub Copilot (根据 Log.md 和 ROADMAP_RETIRE_MINIMAL.md 分析生成)

