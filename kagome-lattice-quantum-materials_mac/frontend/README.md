# Kagome Lattice Optimization - React Frontend
# Kagome晶格优化 - React前端

## 🚀 快速开始 / Quick Start

### 安装依赖 / Install Dependencies
```bash
npm install
```

### 启动开发服务器 / Start Development Server
```bash
npm run dev
```

前端将在 `http://localhost:3000` 启动

**重要**: 确保后端服务器在 `http://localhost:5000` 运行！

---

## 📁 项目结构 / Project Structure

```
frontend/
├── src/
│   ├── api/
│   │   └── kagomeAPI.js          # 后端API客户端
│   ├── components/
│   │   ├── DOSVisualization.jsx  # DOS可视化
│   │   ├── DOSComparison.jsx     # DOS对比图
│   │   ├── ParameterControls.jsx # 参数控制
│   │   ├── OptimizationPanel.jsx # 优化面板
│   │   └── ResultsDisplay.jsx    # 结果展示
│   ├── App.jsx                   # 主应用组件
│   ├── main.jsx                  # React入口
│   └── index.css                 # 全局样式
├── package.json                  # 依赖配置
├── vite.config.js               # Vite配置
└── index.html                   # HTML模板
```

---

## 🎯 核心功能 / Core Features

### 1. 参数控制 / Parameter Controls
- 交互式滑块调整 t_a 和 t_b
- 实时参数显示
- 数值输入框

### 2. DOS可视化 / DOS Visualization
- 实时DOS曲线图
- Recharts图表库
- 响应式设计

### 3. 贝叶斯优化 / Bayesian Optimization
- 配置初始点数和迭代次数
- 实时进度显示
- 步进式优化控制

### 4. 结果展示 / Results Display
- 最佳候选点表格
- 目标函数值排序
- 参数误差计算
- 一键局部优化

### 5. DOS对比 / DOS Comparison
- 目标 vs 预测对比
- 误差度量显示
- 双线图表

---

## 🔧 技术栈 / Tech Stack

- **框架**: React 18 + Vite
- **UI组件**: Material-UI (MUI)
- **图表**: Recharts
- **HTTP客户端**: Axios
- **状态管理**: React Hooks (useState, useEffect)

---

## 📡 API集成 / API Integration

### 后端API端点
所有API调用通过 `src/api/kagomeAPI.js`：

```javascript
import kagomeAPI from './api/kagomeAPI';

// 使用示例
const health = await kagomeAPI.health();
const dos = await kagomeAPI.computeDOS(t_a, t_b);
const target = await kagomeAPI.generateTarget(t_a, t_b);
```

### API端点列表
- `health()` - 健康检查
- `computeDOS(t_a, t_b)` - 计算DOS
- `generateTarget(t_a, t_b)` - 生成目标
- `startOptimization(...)` - 开始优化
- `stepOptimization()` - 执行一步
- `getOptimizationStatus()` - 获取状态
- `localOptimize(...)` - 局部优化
- `getComparisonPlot(...)` - 生成对比图

---

## 🎨 组件说明 / Component Description

### ParameterControls
控制Hamiltonian参数：
- t_a: 最近邻跳跃积分
- t_b: 次近邻跳跃积分
- 范围: [-0.5, 0.5]
- 步进: 0.01

### DOSVisualization
单个DOS曲线显示：
- X轴: 能量 (eV)
- Y轴: 态密度 (任意单位)
- 800个数据点
- 平滑曲线

### DOSComparison
双DOS对比：
- 红线: 目标DOS
- 蓝线: 预测DOS
- 显示MSE误差
- 参数对比

### OptimizationPanel
优化控制：
- 配置初始点数 (3-20)
- 配置迭代次数 (5-50)
- 开始/停止/步进控制
- 实时进度条

### ResultsDisplay
结果展示表格：
- 前5名候选点
- 参数和目标函数值
- 与真实参数的误差
- 一键局部优化

---

## 🚀 开发工作流 / Development Workflow

### 启动完整系统

#### 终端1: 后端
```bash
cd backend
python app_pytorch.py
```

#### 终端2: 前端
```bash
cd frontend
npm run dev
```

#### 浏览器
打开 `http://localhost:3000`

---

## 🧪 使用流程 / Usage Flow

1. **生成目标** 
   - 调整参数 (t_a, t_b)
   - 点击 "Set as Target"

2. **查看目标DOS**
   - 在图表中显示

3. **开始优化**
   - 设置初始点数和迭代次数
   - 点击 "Start Optimization"

4. **查看结果**
   - 查看最佳候选点表格
   - 点击 "Refine" 进行局部优化

5. **对比DOS**
   - 自动显示目标 vs 预测对比

---

## 📊 性能优化 / Performance

- 使用Vite实现快速热更新
- Recharts图表性能优化
- 按需加载组件
- API请求防抖

---

## 🎨 自定义 / Customization

### 修改主题颜色
编辑 `src/App.jsx`:
```javascript
const theme = createTheme({
  palette: {
    primary: { main: '#你的颜色' },
    secondary: { main: '#你的颜色' },
  },
});
```

### 修改图表样式
编辑 `src/components/DOSVisualization.jsx`

### 添加新功能
在 `src/components/` 创建新组件

---

## 🐛 故障排除 / Troubleshooting

### 问题1: 无法连接后端
**解决**: 确保后端在5000端口运行
```bash
curl http://localhost:5000/health
```

### 问题2: CORS错误
**解决**: Vite配置已包含代理，应该不会有CORS问题

### 问题3: 图表不显示
**解决**: 检查DOS数据格式，确保有 `dos` 和 `bins` 字段

---

## 📦 构建生产版本 / Build for Production

```bash
npm run build
```

生成的文件在 `dist/` 目录

---

## 🎉 准备就绪！

启动前端后即可开始使用完整的Kagome晶格优化系统！

**Enjoy! 🚀**
