# ngp-minimal 实现日志

## 项目理解与目标

经过对整个项目和 ROADMAP_RETIRE_MINIMAL.md 的仔细阅读，我理解到您的核心需求是：

### 核心目标
在仓库根目录的 `ngp-minimal/` 目录中，从零开始实现一个**独立的、只支持 NeRF-synthetic 数据集训练的最小版 instant-ngp**。

### 关键约束
1. **完全独立实现**：`ngp-minimal/` 不依赖 `../include` 或 `../src`，不链接 `ngp-baseline-nv` 库
2. **按需搬迁策略**：仅实现 NeRF-synthetic 训练路径实际用到的功能子集
3. **保持接口一致性**：使用与完整版相同的类名/函数名/调用关系，便于对比验证
4. **去除所有冗余**：不包含 GUI、VR、SDF、Volume、Image 等其他模式
5. **交叉验证**：每个阶段都与完整版对比，确保行为一致

---

## 实施计划

根据 ROADMAP 的阶段划分，我将按以下步骤实施：

### Phase 1: 基线确认 & 训练路径调用图梳理 ✓（分析阶段）

**已完成的理解：**

#### 1.1 核心调用链（从 main 到训练）
```
main/wmain
  ↓
ngp::main_func(arguments)
  ↓ 解析 CLI 参数（--scene, --config, --snapshot, --no-train）
  ↓ 构造 Testbed
  ↓
Testbed::load_training_data(scene_path)
  ↓
Testbed::set_mode(ETestbedMode::Nerf)
  ↓
Testbed::load_nerf(data_path)
  ↓
ngp::load_nerf(json_paths)  // 在 nerf_loader.cu
  ↓
Testbed::load_nerf_post()   // 初始化 density grid
  ↓
Testbed::reload_network_from_file(config_path)
  ↓
训练主循环：
while (testbed.frame()) {
    Testbed::train_and_render(skip_rendering)
      ↓
    Testbed::train(batch_size)
      ↓
    Testbed::training_prep_nerf(batch_size)  // 更新 density grid
      ↓
    Testbed::train_nerf_step(batch_size)     // 核心训练步
        - generate_training_samples_nerf()    // 采样
        - network->inference()                 // 前向
        - compute_loss_kernel_train_nerf()    // 损失
        - trainer->training_step()             // 反向+优化
}
```

#### 1.2 核心数据结构（需要搬迁的子集）

**NerfDataset (nerf_loader.h)**
- ✅ 必需字段：
  - `std::vector<TrainingImageMetadata> metadata`
  - `GPUMemory<TrainingImageMetadata> metadata_gpu`
  - `std::vector<TrainingXForm> xforms`
  - `std::vector<std::string> paths`
  - `std::vector<GPUMemory<uint8_t>> pixelmemory`
  - `BoundingBox render_aabb`
  - `vec3 up, offset`
  - `float scale`
  - `int aabb_scale`
  - `size_t n_images`
  
- ❌ 可暂不实现：
  - `raymemory, depthmemory` (除非配置需要)
  - `sharpness_data, envmap_data` (高级功能)
  - `n_extra_learnable_dims, has_light_dirs` (扩展功能)

**Testbed::Nerf::Training (testbed.h)**
- ✅ 必需成员：
  - `NerfDataset dataset`
  - `density_grid` 相关（occupancy grid）
  - `counters_rgb` (训练计数)
  - `loss_type, random_bg_color` 等基础训练配置
  
- ❌ 可暂不实现：
  - `error_map` 及相关 CDF（重要性采样）
  - `train_envmap` 及环境贴图相关
  - `optimize_extrinsics/distortion/focal_length` (相机优化)

#### 1.3 依赖的数学/工具库
- ✅ 必需：
  - `vec2, vec3, vec4, mat3, mat4x3` 等基础数学类型
  - `BoundingBox`
  - `GPUMemory` 及内存管理
  - `Ray` 结构
  - JSON 解析（nlohmann/json）
  - 图像加载（stb_image/tinyexr）
  
- ✅ 核心第三方库：
  - `tiny-cuda-nn`（网络/编码/优化器）
  - CUDA 工具类

---

### Phase 2: ngp-minimal 工程骨架 + 独立 CMake + CLI

**目标：建立独立的可编译工程框架**

#### Milestone 2.1: 独立 CMake 配置
- [ ] 创建 `ngp-minimal/CMakeLists.txt`
  - 定义独立目标 `ngp-minimal-app`
  - 链接必要的第三方库（tiny-cuda-nn, CUDA, JSON, 图像库）
  - **不链接** `ngp-baseline-nv`
  
- [ ] 创建 `ngp-minimal/include/` 目录结构
- [ ] 创建 `ngp-minimal/src/` 目录结构
- [ ] 在根目录 CMakeLists.txt 中添加 `add_subdirectory(ngp-minimal)`

**验证：** 能够成功构建生成 `ngp-minimal-app.exe`

#### Milestone 2.2: 最小 CLI 入口
- [ ] 实现 `ngp-minimal/src/main.cu`
  - 使用 args.hxx 解析参数
  - 支持 `--scene`, `--config`, `--snapshot`, `--no-train`
  - 定义 `CliOptions` 结构
  - 调用 `app_main(CliOptions)`
  
**验证：** `ngp-minimal-app --help` 输出帮助信息

---

### Phase 3: 数据层按需搬迁（NerfDataset + loader）

**目标：实现与完整版接口一致但精简的数据加载**

#### Milestone 3.1: 定义精简版数据结构
- [ ] `ngp-minimal/include/common.h`
  - 基础数学类型：`vec2, vec3, vec4, mat3, mat4x3, ivec2`
  - `BoundingBox` 类
  - `Ray` 结构
  - `EImageDataType, EDepthDataType` 枚举
  
- [ ] `ngp-minimal/include/gpu_memory.h`
  - `GPUMemory<T>` 模板类（CUDA 内存管理）
  
- [ ] `ngp-minimal/include/nerf_loader.h`
  - `struct TrainingImageMetadata`
  - `struct TrainingXForm`
  - `struct NerfDataset`（精简版，仅包含训练必需字段）
  
#### Milestone 3.2: 实现 NeRF-synthetic 数据加载器
- [ ] `ngp-minimal/src/nerf_loader.cu`
  - `NerfDataset load_nerf_synthetic(const std::string& root_path)`
  - 解析 `transforms_train.json`, `transforms_test.json`, `transforms_val.json`
  - 读取 PNG 图像（使用 stb_image）
  - 计算 focal_length, principal_point
  - 构建 camera-to-world 变换矩阵
  - 设置 AABB, offset, scale
  - 上传像素数据到 GPU
  
**验证：**
- [ ] 加载 `data/nerf-synthetic/lego`
- [ ] 导出元数据（n_images, resolution, focal_length, aabb）
- [ ] 与完整版对比，确保数值一致

---

### Phase 4: 网络/采样/渲染核心按需搬迁

**目标：实现 NeRF 网络和体渲染核心**

#### Milestone 4.1: NeRF 网络封装
- [ ] `ngp-minimal/include/nerf_network.h`
  - `class NerfNetwork<T>`（基于 tiny-cuda-nn）
  - `forward()` 方法
  - `density()` 和 `density_and_color()` 接口
  
- [ ] `ngp-minimal/src/nerf_network.cu`
  - 网络配置解析（从 JSON）
  - HashGrid 编码 + MLP 网络组合
  - 与 tiny-cuda-nn 的 Encoding/Network 集成

#### Milestone 4.2: 体渲染与训练核心
- [ ] `ngp-minimal/include/nerf_kernels.cuh`
  - Ray marching 相关 CUDA kernel
  - Volume rendering kernel
  - Loss computation kernel
  
- [ ] `ngp-minimal/src/train_nerf.cu`
  - `generate_training_samples_nerf()`：从 dataset 采样 rays
  - `train_nerf_step()`：
    - 采样训练 batch
    - 前向传播（网络推理）
    - 体渲染累积颜色
    - 计算 RGB loss
    - 反向传播
  
**验证：**
- [ ] 固定一小批样本，对比前向输出
- [ ] 单步训练后 loss 变化合理

---

### Phase 5: Testbed 与训练循环按需搬迁

**目标：实现完整的训练主循环**

#### Milestone 5.1: 精简版 Testbed 类
- [ ] `ngp-minimal/include/testbed.h`
  - `class Testbed`（仅 NeRF 模式）
  - 成员：
    - `NerfDataset m_dataset`
    - `std::shared_ptr<NerfNetwork<T>> m_network`
    - `std::shared_ptr<Trainer<T>> m_trainer`
    - Density grid 相关成员
    - 训练统计：`m_training_step`, `m_loss_scalar`
  - 方法：
    - `load_training_data(path)`
    - `reload_network_from_file(config_path)`
    - `load_snapshot(path)` / `save_snapshot(path)`
    - `frame()` - 主循环入口
    - `train(batch_size)` - 训练一步
    
- [ ] `ngp-minimal/src/testbed.cu`
  - 实现 Testbed 基础方法
  
- [ ] `ngp-minimal/src/testbed_nerf.cu`
  - `training_prep_nerf()`：更新 density grid
  - `train_nerf_step()`：调用 Phase 4 的训练函数
  - `update_density_grid_nerf()`：occupancy grid 更新逻辑

#### Milestone 5.2: 训练主循环集成
- [ ] 在 `main.cu` 中集成：
  ```cpp
  Testbed testbed;
  testbed.load_training_data(opts.scene);
  testbed.reload_network_from_file(opts.config);
  testbed.m_train = !opts.no_train;
  
  while (testbed.frame()) {
      log("iteration={} loss={}", testbed.training_step(), testbed.loss_scalar());
  }
  ```

**验证：**
- [ ] 完整训练 `data/nerf-synthetic/lego` 2000 步
- [ ] 对比与完整版的 loss 曲线
- [ ] 检查训练时间、GPU 利用率

---

### Phase 6: 冗余收缩、行为一致性评估

**目标：精简代码并全面验证**

#### Milestone 6.1: 代码审查与清理
- [ ] 检查所有未使用的字段/方法
- [ ] 删除调试代码和注释掉的代码
- [ ] 统一代码风格

#### Milestone 6.2: 完整性测试
- [ ] 测试场景：
  - `lego` (必选)
  - `chair` (次选)
  - `drums` (可选)
  
- [ ] 对比项：
  - 训练 loss 曲线（前 5k 步）
  - PSNR 指标（如果实现了评估）
  - 训练速度（iter/s）
  - GPU 内存占用
  
#### Milestone 6.3: 文档与复杂度对比
- [ ] 更新本文档，记录：
  - 代码行数对比（Full vs Minimal）
  - 文件数量对比
  - 依赖库数量
  - 编译时间对比
  
- [ ] 编写 `ngp-minimal/README.md`
  - 使用说明
  - 编译指南
  - 训练示例命令
  - 与完整版的差异说明

---

## 实施优先级与时间估算

### 高优先级（核心路径，必须实现）
1. **Phase 2**: 工程骨架（1-2 小时）
2. **Phase 3**: 数据加载（3-4 小时）
3. **Phase 4**: 网络与训练核心（6-8 小时）
4. **Phase 5**: Testbed 集成（4-6 小时）

### 中优先级（验证与优化）
5. **Phase 6**: 清理与验证（2-3 小时）

### 低优先级（可选扩展）
- Snapshot 保存/加载（1-2 小时）
- 简单的渲染输出（测试视角图像）（2-3 小时）
- PSNR 评估（1 小时）

**总计预估：** 20-30 小时纯编码时间

---

## 技术决策

### 1. 编码/网络架构
- 使用与完整版一致的 HashGrid + MLP 架构
- 依赖 tiny-cuda-nn 库提供的 Encoding/Network/Trainer

### 2. 简化点
- **不实现**：GUI、渲染窗口、交互式相机
- **不实现**：VR、多 GPU、DLSS
- **不实现**：SDF、Volume、Image 模式
- **不实现**：Error map importance sampling（初期）
- **不实现**：Envmap 训练
- **不实现**：相机参数优化（extrinsics/distortion/focal）
- **初期简化**：只支持 Blender NeRF-synthetic 格式

### 3. 保留接口一致性的策略
- 类名：`Testbed`, `NerfDataset`, `NerfNetwork` 与原版一致
- 函数名：`load_training_data`, `frame`, `train_nerf_step` 等保持一致
- 调用顺序：严格按照原版的调用图
- **好处**：便于代码对比、迁移经验、后续扩展

---

## 验证策略

### 数据层验证
- 导出完整版与 minimal 版的 dataset 元数据到 JSON
- 使用 Python 脚本对比差异

### 训练验证
- 固定随机种子
- 对比前 100/500/1000/5000 步的 loss
- 允许浮点误差，但趋势应一致

### 视觉验证（如果实现渲染）
- 导出相同迭代步的测试视角图像
- 计算 PSNR/SSIM 或目测对比

---

## 风险与应对

### 风险 1: 对 tiny-cuda-nn 接口不熟悉
**应对**：先研究完整版如何使用，照搬接口调用方式

### 风险 2: CUDA kernel 实现复杂
**应对**：尽量复用 tiny-cuda-nn 的 fused kernel，简化自定义 kernel

### 风险 3: 训练结果与完整版不一致
**应对**：分阶段验证，每个 milestone 都做对比，及时发现问题

### 风险 4: 依赖库版本兼容性问题
**应对**：使用与完整版相同的 vcpkg manifest，保持依赖一致

---

## 下一步行动

现在我已经完成了需求理解和详细规划，**准备开始实施 Phase 2: 工程骨架搭建**。

具体将：
1. 创建 `ngp-minimal/` 目录结构
2. 编写独立的 CMakeLists.txt
3. 实现最小 CLI 入口（main.cu）
4. 验证编译通过

**是否开始执行？**

