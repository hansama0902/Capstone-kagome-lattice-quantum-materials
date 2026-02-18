/**
 * Kagome Lattice Backend API Client
 * Kagome晶格后端API客户端
 */

import axios from 'axios';

const API_BASE = 'http://localhost:5001';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 120000, // 120 seconds for long computations
  headers: {
    'Content-Type': 'application/json',
  },
});

export const kagomeAPI = {
  /**
   * Health check endpoint
   * 健康检查端点
   */
  async health() {
    const response = await apiClient.get('/health');
    return response.data;
  },

  /**
   * Compute DOS for given parameters
   * 计算给定参数的DOS
   */
  async computeDOS(t_a, t_b, n_kpoints = 800) {
    const response = await apiClient.post('/api/compute_dos', {
      t_a,
      t_b,
      n_kpoints,
    });
    return response.data;
  },

  /**
   * Generate target DOS from known parameters
   * 从已知参数生成目标DOS
   */
  async generateTarget(t_a, t_b, n_kpoints = 800) {
    const response = await apiClient.post('/api/generate_target_dos', {
      t_a,
      t_b,
      n_kpoints,
    });
    return response.data;
  },

  /**
   * Start Bayesian Optimization
   * 开始贝叶斯优化
   */
  async startOptimization(dos_target, bins_target, n_initial = 5, n_iterations = 20, metric = 'mse') {
    const response = await apiClient.post('/api/start_optimization', {
      dos_target,
      bins_target,
      n_initial,
      n_iterations,
      metric,
    });
    return response.data;
  },

  /**
   * Execute one optimization step
   * 执行一步优化
   */
  async stepOptimization() {
    const response = await apiClient.post('/api/step_optimization');
    return response.data;
  },

  /**
   * Get current optimization status
   * 获取当前优化状态
   */
  async getOptimizationStatus() {
    const response = await apiClient.get('/api/get_optimization_status');
    return response.data;
  },

  /**
   * Perform local optimization
   * 执行局部优化
   */
  async localOptimize(initial_point, dos_target, bins_target, metric = 'mse') {
    const response = await apiClient.post('/api/local_optimize', {
      initial_point,
      dos_target,
      bins_target,
      metric,
    });
    return response.data;
  },

  /**
   * Generate DOS comparison plot
   * 生成DOS对比图
   */
  async getComparisonPlot(dos_target, bins_target, points, metric = 'mse') {
    const response = await apiClient.post('/api/plot_dos_comparison', {
      dos_target,
      bins_target,
      points,
      metric,
    });
    return response.data;
  },

  /**
   * Get parameter space visualization data
   * 获取参数空间可视化数据
   */
  async getParameterSpace(resolution = 50, iteration = null) {
    const response = await apiClient.post('/api/get_parameter_space', {
      resolution,
      iteration,  // Pass selected iteration number
    });
    return response.data;
  },

  /**
   * Generate multi-panel DOS comparison (Top 5)
   * 生成多面板DOS对比（Top 5候选点）
   */
  async getMultiDOSPlot(dos_target, bins_target, candidates) {
    const response = await apiClient.post('/api/plot_multi_dos', {
      dos_target,
      bins_target,
      candidates,
    });
    return response.data;
  },
};

export default kagomeAPI;
