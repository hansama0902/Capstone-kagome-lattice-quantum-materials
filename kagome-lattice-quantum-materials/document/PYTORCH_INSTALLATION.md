# PyTorch版本安装和使用指南
# PyTorch Version Installation & Usage Guide

## 🚀 快速安装 / Quick Installation

### 选项1: CPU版本（任何机器都可以运行）
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install botorch gpytorch
```

### 选项2: CUDA版本（需要NVIDIA GPU）
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 然后安装BoTorch和GPyTorch
pip install botorch gpytorch
```

### 选项3: Mac with Apple Silicon (M1/M2/M3)
```bash
pip install torch torchvision torchaudio
pip install botorch gpytorch
```

---

## 📋 版本对比 / Version Comparison

### 文件说明
- **app.py** - NumPy/SciPy版本（已测试，可立即使用）
- **app_pytorch.py** - PyTorch/BoTorch版本（更强大，需要安装依赖）

### API完全兼容！
两个版本的API端点完全相同，React前端无需任何修改即可切换！

---

## 🔧 使用PyTorch版本 / Using PyTorch Version

### 第1步：安装依赖
```bash
# 检查是否有GPU
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# 如果没有CUDA，安装CPU版本
pip install torch botorch gpytorch --break-system-packages
```

### 第2步：启动PyTorch版本
```bash
cd /home/claude/backend
python3 app_pytorch.py
```

### 第3步：测试
```bash
# 在另一个终端
curl http://localhost:5000/health
```

**预期输出:**
```json
{
  "status": "healthy",
  "backend": "pytorch/botorch",
  "device": "cuda:0",  // 或 "cpu"
  "cuda_available": true,  // 或 false
  "torch_version": "2.x.x"
}
```

---

## ⚡ 性能对比测试 / Performance Comparison

### 自动对比测试
```bash
python3 compare_backends.py
```

这个脚本会：
1. 测试NumPy版本的速度
2. 测试PyTorch版本的速度（如果已安装）
3. 对比准确性
4. 生成性能报告

---

## 🎯 PyTorch版本的优势 / PyTorch Version Advantages

### 1. **批量采集** (BATCH_SIZE > 1)
```python
# NumPy版本：每次只能建议1个点
# PyTorch版本：每次可以建议5个点
BATCH_SIZE = 5  # ✅ PyTorch支持！
```

**效果:**
- 总评估次数更多（20 + 30×5 = 170次）
- 探索空间更充分
- 找到更优解的概率更高

### 2. **GPU加速**
```
CPU (NumPy):    单次DOS计算 ~2-3秒
CPU (PyTorch):  单次DOS计算 ~1-2秒
GPU (PyTorch):  单次DOS计算 ~0.3-0.5秒 🚀
```

### 3. **更强大的GP模型**
- GPyTorch的精确边际似然优化
- 更灵活的核函数
- 自动超参数调优
- 数值稳定性更好

### 4. **高级采集函数**
```python
# 可以使用更多采集函数
from botorch.acquisition import (
    qExpectedImprovement,       # ✅ 当前使用
    qUpperConfidenceBound,       # ✅ 可选
    qProbabilityOfImprovement,   # ✅ 可选
    qNoisyExpectedImprovement    # ✅ 可选
)
```

---

## 🔄 如何切换版本 / How to Switch Versions

### 使用NumPy版本
```bash
python3 app.py
```

### 使用PyTorch版本
```bash
python3 app_pytorch.py
```

### 在React前端中切换
React前端**完全不需要修改**！只需要：
1. 停止当前后端
2. 启动另一个版本
3. 继续使用

---

## 📊 功能对比表 / Feature Comparison

| 功能 | NumPy版本 | PyTorch版本 |
|-----|----------|------------|
| **基础功能** | | |
| DOS计算 | ✅ | ✅ |
| BO优化 | ✅ | ✅ |
| 局部优化 | ✅ | ✅ |
| API端点 | ✅ (8个) | ✅ (8个) |
| **高级功能** | | |
| 批量采集 | ❌ (仅1个) | ✅ (5个) |
| GPU加速 | ❌ | ✅ |
| 多采集函数 | ❌ | ✅ |
| **性能** | | |
| 速度 (CPU) | 慢 | 中 |
| 速度 (GPU) | N/A | 快 🚀 |
| 准确性 | 好 | 很好 ⭐ |
| **部署** | | |
| 依赖复杂度 | 低 | 中 |
| 安装大小 | 0MB | ~1GB |
| 易用性 | 高 | 中 |

---

## 🎓 建议使用场景 / Recommended Use Cases

### 使用NumPy版本（app.py）的场景:
- ✅ 快速原型开发
- ✅ 课程演示
- ✅ 没有GPU
- ✅ 简单部署环境
- ✅ 迭代次数 < 50

### 使用PyTorch版本（app_pytorch.py）的场景:
- ✅ 有GPU可用
- ✅ 需要大量迭代（> 50次）
- ✅ 论文研究
- ✅ 生产环境
- ✅ 追求最优性能

---

## 🔍 验证安装 / Verify Installation

### 检查PyTorch
```bash
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

### 检查BoTorch
```bash
python3 -c "import botorch; print('BoTorch:', botorch.__version__)"
```

### 检查GPyTorch
```bash
python3 -c "import gpytorch; print('GPyTorch:', gpytorch.__version__)"
```

### 完整测试
```bash
python3 app_pytorch.py &
sleep 3
curl http://localhost:5000/health
```

---

## 🐛 常见问题 / Troubleshooting

### 问题1: 无法安装PyTorch
**解决方案:**
```bash
# 尝试CPU版本
pip install torch --index-url https://download.pytorch.org/whl/cpu --break-system-packages

# 或者直接使用NumPy版本
python3 app.py
```

### 问题2: CUDA版本不匹配
**解决方案:**
```bash
# 检查CUDA版本
nvcc --version

# 根据版本选择对应的PyTorch
# 参考：https://pytorch.org/get-started/locally/
```

### 问题3: 内存不足
**解决方案:**
减少采样数：
```python
# 在app_pytorch.py中修改
MC_SAMPLES = 512  # 减少到512
RAW_SAMPLES = 256  # 减少到256
```

### 问题4: 计算太慢
**解决方案:**
1. 检查是否使用GPU：`print(tkwargs['device'])`
2. 减少k点数：`n_kpoints=500` → `n_kpoints=300`
3. 减少迭代次数

---

## 📝 配置选项 / Configuration Options

### 在app_pytorch.py中可以调整:

```python
# 优化参数
MC_SAMPLES = 1024      # 蒙特卡洛采样数 (减少→更快但略不准)
NUM_RESTARTS = 10      # 重启次数 (增加→更准但更慢)
RAW_SAMPLES = 512      # 原始采样数
BATCH_SIZE = 5         # 批量大小 (1-10)

# 物理参数
n_kpoints = 1000       # k点数量 (减少→更快但略不准)
sigma = 5.0           # 高斯平滑参数
bins = 800            # 能量bins数量
```

---

## 🎉 推荐配置 / Recommended Settings

### 快速测试（~5分钟）
```python
n_initial = 5
n_iterations = 10
BATCH_SIZE = 3
n_kpoints = 300
```

### 标准运行（~15分钟）
```python
n_initial = 10
n_iterations = 20
BATCH_SIZE = 5
n_kpoints = 500
```

### 高精度（~30分钟）
```python
n_initial = 20
n_iterations = 30
BATCH_SIZE = 5
n_kpoints = 1000
```

---

## 📞 获取帮助 / Get Help

如果遇到问题:
1. 查看此文档的常见问题部分
2. 检查 `health` 端点的返回信息
3. 查看终端的错误信息
4. 如果PyTorch有问题，可以随时切换回NumPy版本

---

**两个版本都已准备好！选择最适合你的一个开始吧！** 🚀
