"""
Flask Backend with PyTorch/BoTorch Implementation
基于PyTorch/BoTorch的Flask后端实现

This version uses:
- PyTorch for GPU acceleration
- BoTorch for advanced Bayesian Optimization
- GPyTorch for Gaussian Process models
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import warnings
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde

# PyTorch imports
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition import qExpectedImprovement
from botorch import fit_gpytorch_mll
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler

# Suppress warnings
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global configuration
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# Optimization parameters
dimension = 2
lower_bound = torch.tensor([-0.5, -0.5], **tkwargs)
upper_bound = torch.tensor([0.5, 0.5], **tkwargs)
standard_bounds = torch.zeros(2, dimension, **tkwargs)
standard_bounds[1] = 1.0

# BO parameters
MC_SAMPLES = 1024
NUM_RESTARTS = 10
RAW_SAMPLES = 512
BATCH_SIZE = 5

D_LATTICE = 0.133

# Optimization state
optimization_state = {
    'is_running': False,
    'current_iteration': 0,
    'total_iterations': 0,
    'train_x': None,
    'train_obj': None,
    'model': None,
    'mll': None,
    'sampler': None,
    'history': [],
    'dos_target': None,
    'bins_target': None,
    'iteration_snapshots': []  # Store state at each iteration
}


def kagome_hamiltonian(k, t_a=-1.0, t_b=-1.0, d=0.133):
    """
    Kagome lattice Hamiltonian (NumPy version for DOS computation)
    Kagome晶格哈密顿量（NumPy版本用于DOS计算）
    """
    a = d * 2
    kx, ky = k
    
    H = np.zeros((3, 3), dtype=complex)
    
    phase1 = np.exp(1j * (kx * a * np.sqrt(3)/2 - ky * a/2))
    
    H[0, 2] = t_a + t_b * phase1
    H[2, 0] = np.conj(H[0, 2])
    
    H[1, 0] = t_a + t_b * np.conj(phase1) * np.exp(1j * ky * a)
    H[0, 1] = np.conj(H[1, 0])
    
    H[2, 1] = t_a + t_b * np.exp(-1j * ky * a)
    H[1, 2] = np.conj(H[2, 1])
    
    return H


def compute_dos(t_a, t_b, d=0.133, bins=800, energy_range=(-0.15, 0.25), 
                sigma=5.0, n_kpoints=1000):
    """
    Compute Density of States
    计算态密度
    """
    n_per_dim = int(np.sqrt(n_kpoints))
    kx_vals = np.linspace(-np.pi/(d*np.sqrt(3)), np.pi/(d*np.sqrt(3)), n_per_dim)
    ky_vals = np.linspace(-np.pi/d, np.pi/d, n_per_dim)
    
    eigenvalues = []
    for kx in kx_vals:
        for ky in ky_vals:
            try:
                H = kagome_hamiltonian([kx, ky], t_a, t_b, d)
                eigvals = np.linalg.eigvalsh(H)
                
                # Check for NaN or Inf in eigenvalues
                if not (np.isnan(eigvals).any() or np.isinf(eigvals).any()):
                    eigenvalues.extend(eigvals)
            except Exception as e:
                # Skip this k-point if calculation fails
                continue
    
    if len(eigenvalues) == 0:
        # No valid eigenvalues, return zero DOS
        print(f"  Warning: No valid eigenvalues for t_a={t_a:.4f}, t_b={t_b:.4f}")
        counts = np.zeros(bins)
        bin_edges = np.linspace(energy_range[0], energy_range[1], bins + 1)
        return counts, bin_edges
    
    eigenvalues = np.array(eigenvalues)
    
    # Remove any NaN or Inf values
    eigenvalues = eigenvalues[~np.isnan(eigenvalues)]
    eigenvalues = eigenvalues[~np.isinf(eigenvalues)]
    
    if len(eigenvalues) == 0:
        # All eigenvalues were invalid
        print(f"  Warning: All eigenvalues invalid for t_a={t_a:.4f}, t_b={t_b:.4f}")
        counts = np.zeros(bins)
        bin_edges = np.linspace(energy_range[0], energy_range[1], bins + 1)
        return counts, bin_edges
    
    counts, bin_edges = np.histogram(eigenvalues, bins=bins, 
                                     range=energy_range, density=True)
    
    # Replace NaN with 0
    counts = np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Apply Gaussian smoothing
    counts = gaussian_filter1d(counts, sigma=sigma)
    
    # Final NaN check
    counts = np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
    
    return counts, bin_edges


def compute_dos_distance(dos1, dos2, metric='mse'):
    """
    Compute distance between two DOS
    计算两个DOS之间的距离
    
    Args:
        dos1: First DOS array
        dos2: Second DOS array
        metric: 'mse' or 'wasserstein'
    """
    # Normalize DOS
    dos1_norm = dos1 / (np.sum(dos1) + 1e-10)
    dos2_norm = dos2 / (np.sum(dos2) + 1e-10)
    
    if metric == 'mse':
        # Mean Squared Error
        return np.mean((dos1_norm - dos2_norm)**2)
    elif metric == 'wasserstein':
        # Wasserstein distance (simplified)
        # Using cumulative distribution
        cdf1 = np.cumsum(dos1_norm)
        cdf2 = np.cumsum(dos2_norm)
        return np.mean(np.abs(cdf1 - cdf2))
    else:
        return np.mean((dos1_norm - dos2_norm)**2)


def compute_train_obj(x, dos_target, bins_target, metric='mse'):
    """
    Compute optimization objective for PyTorch tensor
    为PyTorch张量计算优化目标
    
    Args:
        x: PyTorch tensor of parameters [batch_size, 2]
        dos_target: Target DOS (numpy array)
        bins_target: Energy bins (numpy array)
        metric: Distance metric to use
    
    Returns:
        PyTorch tensor of negative distances [batch_size, 1]
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    
    objectives = []
    
    for i in range(x.shape[0]):
        params = x[i].cpu().numpy()
        t_a, t_b = float(params[0]), float(params[1])
        
        # Check bounds
        if t_a < lower_bound[0].item() or t_a > upper_bound[0].item():
            objectives.append(-1000.0)
            continue
        if t_b < lower_bound[1].item() or t_b > upper_bound[1].item():
            objectives.append(-1000.0)
            continue
        
        try:
            # Compute DOS
            dos_current, _ = compute_dos(t_a, t_b, n_kpoints=200)
            
            # Check for NaN or Inf in DOS
            if np.isnan(dos_current).any():
                print(f"  Warning: NaN in DOS for t_a={t_a:.4f}, t_b={t_b:.4f}")
                objectives.append(-1000.0)
                continue
            
            if np.isinf(dos_current).any():
                print(f"  Warning: Inf in DOS for t_a={t_a:.4f}, t_b={t_b:.4f}")
                objectives.append(-1000.0)
                continue
            
            # Check if DOS is all zeros
            if np.sum(np.abs(dos_current)) < 1e-10:
                print(f"  Warning: Zero DOS for t_a={t_a:.4f}, t_b={t_b:.4f}")
                objectives.append(-1000.0)
                continue
            
            # Ensure same length
            if len(dos_current) != len(dos_target):
                # Interpolate to match target length
                from scipy.interpolate import interp1d
                x_old = np.linspace(0, 1, len(dos_current))
                x_new = np.linspace(0, 1, len(dos_target))
                f = interp1d(x_old, dos_current, kind='linear', fill_value='extrapolate')
                dos_current = f(x_new)
            
            # Compute distance
            distance = compute_dos_distance(dos_target, dos_current, metric=metric)
            
            # Check for NaN in distance
            if np.isnan(distance) or np.isinf(distance):
                print(f"  Warning: Invalid distance for t_a={t_a:.4f}, t_b={t_b:.4f}")
                objectives.append(-1000.0)
                continue
            
            # Return negative (for maximization)
            objectives.append(-distance)
            
        except Exception as e:
            print(f"  Error computing objective for t_a={t_a:.4f}, t_b={t_b:.4f}: {e}")
            objectives.append(-1000.0)
    
    # Convert to tensor and check for NaN
    obj_tensor = torch.tensor(objectives, dtype=tkwargs['dtype'], device=tkwargs['device']).unsqueeze(-1)
    
    # Replace any remaining NaN with penalty value
    obj_tensor = torch.where(torch.isnan(obj_tensor), 
                             torch.tensor(-1000.0, **tkwargs), 
                             obj_tensor)
    
    return obj_tensor


def initialize_model(train_x, train_obj):
    """
    Initialize Gaussian Process model with BoTorch
    使用BoTorch初始化高斯过程模型
    """
    model = SingleTaskGP(
        train_x, 
        train_obj, 
        outcome_transform=Standardize(m=train_obj.shape[-1])
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model


def optimize_acqf_and_get_observation(model, train_x, train_obj, sampler):
    """
    Optimize acquisition function and get new observation
    优化采集函数并获取新观测点
    """
    qei = qExpectedImprovement(
        model=model, 
        best_f=train_obj.max(), 
        sampler=sampler
    )
    
    candidates, _ = optimize_acqf(
        acq_function=qei,
        bounds=standard_bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )
    
    # Unnormalize candidates
    new_x = unnormalize(candidates.detach(), bounds=torch.stack([lower_bound, upper_bound]))
    
    return new_x


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'backend': 'pytorch/botorch',
        'device': str(tkwargs['device']),
        'cuda_available': torch.cuda.is_available(),
        'torch_version': torch.__version__
    })


@app.route('/api/compute_dos', methods=['POST'])
def api_compute_dos():
    """Compute DOS for given parameters"""
    data = request.json
    t_a = float(data.get('t_a', -1.0))
    t_b = float(data.get('t_b', -1.0))
    n_kpoints = int(data.get('n_kpoints', 800))
    
    dos, bins = compute_dos(t_a, t_b, n_kpoints=n_kpoints)
    
    return jsonify({
        'dos': dos.tolist(),
        'bins': bins.tolist(),
        'parameters': {'t_a': t_a, 't_b': t_b}
    })


@app.route('/api/generate_target_dos', methods=['POST'])
def api_generate_target():
    """Generate target DOS"""
    data = request.json
    t_a_target = float(data.get('t_a', -0.3))
    t_b_target = float(data.get('t_b', -0.2))
    n_kpoints = int(data.get('n_kpoints', 800))
    
    dos_target, bins_target = compute_dos(t_a_target, t_b_target, n_kpoints=n_kpoints)
    
    return jsonify({
        'dos': dos_target.tolist(),
        'bins': bins_target.tolist(),
        'true_parameters': {'t_a': t_a_target, 't_b': t_b_target}
    })


@app.route('/api/start_optimization', methods=['POST'])
def api_start_optimization():
    """Start Bayesian Optimization with PyTorch/BoTorch"""
    global optimization_state
    
    try:
        # Allow restart if optimization was completed
        if optimization_state['is_running']:
            if optimization_state['current_iteration'] >= optimization_state['total_iterations']:
                # Reset for new optimization
                print("DEBUG: Resetting completed optimization for restart")
                optimization_state['is_running'] = False
            else:
                return jsonify({
                    'error': 'Optimization already running',
                    'current_iteration': optimization_state['current_iteration'],
                    'total_iterations': optimization_state['total_iterations']
                }), 400
        
        data = request.json
        print(f"\n{'='*70}")
        print("DEBUG: Received optimization request")
        print(f"  Data keys: {list(data.keys())}")
        print(f"  dos_target length: {len(data.get('dos_target', []))}")
        print(f"  bins_target length: {len(data.get('bins_target', []))}")
        print(f"  n_initial: {data.get('n_initial')}")
        print(f"  n_iterations: {data.get('n_iterations')}")
        
        dos_target = np.array(data['dos_target'])
        bins_target = np.array(data['bins_target'])
        
        print(f"  dos_target shape: {dos_target.shape}")
        print(f"  dos_target type: {dos_target.dtype}")
        print(f"  dos_target has NaN: {np.isnan(dos_target).any()}")
        print(f"  dos_target has Inf: {np.isinf(dos_target).any()}")
        
        n_initial = int(data.get('n_initial', 10))
        n_iterations = int(data.get('n_iterations', 20))
        metric = data.get('metric', 'mse')  # 'mse' or 'wasserstein'
        
        print(f"  Parsed parameters OK")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n❌ ERROR in request parsing: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Request parsing error: {str(e)}'}), 400
    
    optimization_state['is_running'] = True
    optimization_state['current_iteration'] = 0
    optimization_state['total_iterations'] = n_iterations
    optimization_state['history'] = []
    optimization_state['dos_target'] = dos_target
    optimization_state['bins_target'] = bins_target
    optimization_state['metric'] = metric
    
    try:
        print("DEBUG: Generating initial points...")
        # Generate initial points using Sobol sampling
        train_x_qei = draw_sobol_samples(
            bounds=standard_bounds, 
            n=n_initial, 
            q=1
        ).squeeze(1)
        print(f"  Sobol samples shape: {train_x_qei.shape}")
        
        # Unnormalize
        train_x_qei = unnormalize(train_x_qei, bounds=torch.stack([lower_bound, upper_bound]))
        print(f"  Unnormalized samples: {train_x_qei}")
        
    except Exception as e:
        print(f"❌ ERROR in sampling: {e}")
        import traceback
        traceback.print_exc()
        optimization_state['is_running'] = False
        return jsonify({'error': f'Sampling error: {str(e)}'}), 500
    
    try:
        # Evaluate initial points
        print(f"DEBUG: Evaluating {n_initial} initial points...")
        print(f"  Using n_kpoints for DOS computation")
        train_obj_qei = compute_train_obj(train_x_qei, dos_target, bins_target, metric=metric)
        print(f"  Objectives computed: {train_obj_qei}")
        
    except Exception as e:
        print(f"❌ ERROR in objective computation: {e}")
        import traceback
        traceback.print_exc()
        optimization_state['is_running'] = False
        return jsonify({'error': f'Objective computation error: {str(e)}'}), 500
    
    try:
        # Add to history
        for i, (x, obj) in enumerate(zip(train_x_qei, train_obj_qei)):
            optimization_state['history'].append({
                'iteration': 0,
                'point': x.cpu().numpy().tolist(),
                'objective': obj.item(),
                'type': 'initial'
            })
        print("  History updated")
        
        # Initialize model
        print("DEBUG: Initializing GP model...")
        mll_qei, model_qei = initialize_model(train_x_qei, train_obj_qei)
        print("  Model initialized")
        
        # Fit GP
        print("DEBUG: Fitting GP...")
        fit_gpytorch_mll(mll_qei)
        print("  GP fitted")
        
        # Initialize sampler
        print("DEBUG: Initializing sampler...")
        qei_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        print("  Sampler initialized")
        
    except Exception as e:
        print(f"❌ ERROR in model initialization: {e}")
        import traceback
        traceback.print_exc()
        optimization_state['is_running'] = False
        return jsonify({'error': f'Model initialization error: {str(e)}'}), 500
    
    # Store state
    optimization_state['train_x'] = train_x_qei
    optimization_state['train_obj'] = train_obj_qei
    optimization_state['model'] = model_qei
    optimization_state['mll'] = mll_qei
    optimization_state['sampler'] = qei_sampler
    
    # Get best points
    sorted_indices = torch.argsort(train_obj_qei.squeeze(), descending=True)[:5]
    best_points = train_x_qei[sorted_indices].cpu().numpy().tolist()
    best_objectives = train_obj_qei[sorted_indices].cpu().numpy().tolist()
    
    # Save initial snapshot (Iteration 0)
    initial_snapshot = {
        'iteration': 0,
        'best_points': best_points,
        'best_objectives': best_objectives,
        'n_evaluated': len(train_x_qei),
        'train_x': train_x_qei.cpu().numpy().tolist(),
        'train_obj': train_obj_qei.cpu().numpy().tolist(),
        'new_points': [],  # No new points in initialization
    }
    optimization_state['iteration_snapshots'] = [initial_snapshot]  # Reset and add initial
    
    print("DEBUG: Optimization initialized successfully!")
    print(f"  Best objective: {best_objectives[0][0]:.6f}")
    print(f"  Saved initial snapshot (Iteration 0)")
    print(f"{'='*70}\n")
    
    return jsonify({
        'status': 'initialized',
        'initial_points': train_x_qei.cpu().numpy().tolist(),
        'initial_objectives': train_obj_qei.cpu().numpy().tolist(),
        'best_points': best_points,
        'best_objectives': best_objectives,
        'metric': metric
    })


@app.route('/api/step_optimization', methods=['POST'])
def api_step_optimization():
    """Execute one optimization step"""
    global optimization_state
    
    if not optimization_state['is_running']:
        return jsonify({
            'status': 'error',
            'error': 'Optimization not started',
            'message': 'Please call /api/start_optimization first'
        }), 400
    
    if optimization_state['current_iteration'] >= optimization_state['total_iterations']:
        optimization_state['is_running'] = False
        return jsonify({
            'status': 'completed',
            'message': 'All iterations completed',
            'current_iteration': optimization_state['current_iteration'],
            'total_iterations': optimization_state['total_iterations']
        })
    
    try:
        # Fit GP model
        fit_gpytorch_mll(optimization_state['mll'])
        
        # Optimize acquisition function
        new_x = optimize_acqf_and_get_observation(
            optimization_state['model'],
            optimization_state['train_x'],
            optimization_state['train_obj'],
            optimization_state['sampler']
        )
        
        # Evaluate new points
        print(f"Evaluating {new_x.shape[0]} new points...")
        new_obj = compute_train_obj(
            new_x,
            optimization_state['dos_target'],
            optimization_state['bins_target'],
            metric=optimization_state['metric']
        )
        
        # Update training data
        optimization_state['train_x'] = torch.cat([optimization_state['train_x'], new_x])
        optimization_state['train_obj'] = torch.cat([optimization_state['train_obj'], new_obj])
        
        # Remove duplicates
        x_np = optimization_state['train_x'].cpu().numpy()
        _, unique_indices = np.unique(x_np, axis=0, return_index=True)
        unique_indices = np.sort(unique_indices)
        unique_indices = torch.tensor(unique_indices, dtype=torch.long, 
                                     device=optimization_state['train_x'].device)
        optimization_state['train_x'] = optimization_state['train_x'][unique_indices]
        optimization_state['train_obj'] = optimization_state['train_obj'][unique_indices]
        
        # Reinitialize model
        mll, model = initialize_model(optimization_state['train_x'], 
                                      optimization_state['train_obj'])
        optimization_state['model'] = model
        optimization_state['mll'] = mll
        
        # Update iteration
        optimization_state['current_iteration'] += 1
        
        # Add to history
        for i, (x, obj) in enumerate(zip(new_x, new_obj)):
            optimization_state['history'].append({
                'iteration': optimization_state['current_iteration'],
                'point': x.cpu().numpy().tolist(),
                'objective': obj.item(),
                'type': 'bo_step'
            })
        
        # Get current best points
        sorted_indices = torch.argsort(optimization_state['train_obj'].squeeze(), 
                                      descending=True)[:5]
        best_points = optimization_state['train_x'][sorted_indices].cpu().numpy().tolist()
        best_objectives = optimization_state['train_obj'][sorted_indices].cpu().numpy().tolist()
        
        # Save iteration snapshot with complete data
        snapshot = {
            'iteration': optimization_state['current_iteration'],
            'best_points': best_points,
            'best_objectives': best_objectives,
            'n_evaluated': len(optimization_state['train_x']),
            'train_x': optimization_state['train_x'].cpu().numpy().tolist(),
            'train_obj': optimization_state['train_obj'].cpu().numpy().tolist(),
            'new_points': new_x.cpu().numpy().tolist(),  # Save new points for visualization
        }
        optimization_state['iteration_snapshots'].append(snapshot)
        
        # Also save new_x in current state for immediate visualization
        optimization_state['last_new_points'] = new_x.cpu().numpy()
        
        return jsonify({
            'status': 'running',
            'current_iteration': optimization_state['current_iteration'],
            'total_iterations': optimization_state['total_iterations'],
            'new_points': new_x.cpu().numpy().tolist(),
            'new_objectives': new_obj.cpu().numpy().tolist(),
            'best_points': best_points,
            'best_objectives': best_objectives
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        optimization_state['is_running'] = False
        return jsonify({'error': str(e)}), 500


@app.route('/api/get_optimization_status', methods=['GET'])
def api_get_status():
    """Get optimization status"""
    if not optimization_state['is_running'] and optimization_state['current_iteration'] == 0:
        return jsonify({'status': 'not_started'})
    
    if optimization_state['current_iteration'] >= optimization_state['total_iterations']:
        return jsonify({
            'status': 'completed',
            'history': optimization_state['history']
        })
    
    return jsonify({
        'status': 'running',
        'current_iteration': optimization_state['current_iteration'],
        'total_iterations': optimization_state['total_iterations'],
        'is_running': optimization_state['is_running']
    })


@app.route('/api/local_optimize', methods=['POST'])
def api_local_optimize():
    """Local optimization using scipy"""
    from scipy.optimize import minimize
    
    data = request.json
    initial_point = np.array(data['initial_point'])
    dos_target = np.array(data['dos_target'])
    bins_target = np.array(data['bins_target'])
    metric = data.get('metric', 'mse')
    
    bounds = [(lower_bound[i].item(), upper_bound[i].item()) for i in range(2)]
    
    def objective(x):
        x_tensor = torch.tensor([x], **tkwargs)
        obj = compute_train_obj(x_tensor, dos_target, bins_target, metric=metric)
        return -obj.item()
    
    result = minimize(objective, initial_point, method='Powell',
                     bounds=bounds, options={'maxiter': 50})
    
    return jsonify({
        'optimized_point': result.x.tolist(),
        'objective_value': -result.fun,
        'success': result.success,
        'message': 'Optimization completed'
    })


@app.route('/api/plot_dos_comparison', methods=['POST'])
def api_plot_comparison():
    """Generate DOS comparison plot"""
    data = request.json
    dos_target = np.array(data['dos_target'])
    bins_target = np.array(data['bins_target'])
    points = data['points']
    metric = data.get('metric', 'mse')
    
    n_plots = len(points)
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4), sharey=True)
    if n_plots == 1:
        axes = [axes]
    
    for i, (ax, point) in enumerate(zip(axes, points)):
        ax.plot(bins_target[:-1], dos_target, 'r-', linewidth=2, 
               label='Target', alpha=0.8)
        
        dos_pred, bins_pred = compute_dos(point[0], point[1], n_kpoints=800)
        distance = compute_dos_distance(dos_target, dos_pred, metric=metric)
        
        ax.plot(bins_pred[:-1], dos_pred, 'b-', linewidth=2, 
               label=f'Predicted (Err={distance:.4f})')
        
        ax.set_xlabel('Energy (eV)', fontsize=12)
        if i == 0:
            ax.set_ylabel('DOS (arb. units)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title(f't_a={point[0]:.3f}, t_b={point[1]:.3f}', fontsize=11)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return jsonify({'image': img_base64})


@app.route('/api/get_parameter_space', methods=['POST'])
def api_get_parameter_space():
    """
    Generate parameter space visualization plot
    生成参数空间可视化图
    """
    if not optimization_state['model']:
        return jsonify({'error': 'No model available. Run optimization first.'}), 400
    
    data = request.json
    resolution = int(data.get('resolution', 100))
    iteration_num = data.get('iteration')  # Can be None for current
    
    try:
        print(f"\nDEBUG: Generating parameter space plot (resolution={resolution}, iteration={iteration_num})")
        
        # Determine which data to use
        use_snapshot = False
        new_points = np.array([])
        
        if iteration_num is not None and iteration_num > 0:
            # Try to get historical snapshot
            if iteration_num <= len(optimization_state['iteration_snapshots']):
                snapshot = optimization_state['iteration_snapshots'][iteration_num - 1]
                train_x = np.array(snapshot.get('train_x', []))
                train_obj = np.array(snapshot.get('train_obj', []))
                top5_points = np.array(snapshot.get('best_points', []))
                new_points = np.array(snapshot.get('new_points', []))
                use_snapshot = True
                print(f"  Using historical snapshot: Iteration {iteration_num}")
                print(f"  Train points in snapshot: {len(train_x)}")
            else:
                print(f"  Iteration {iteration_num} not found in history")
                use_snapshot = False
        
        if not use_snapshot:
            # Use current state
            train_x = optimization_state['train_x'].cpu().numpy()
            train_obj = optimization_state['train_obj'].cpu().numpy()
            sorted_indices = torch.argsort(
                optimization_state['train_obj'].squeeze(), 
                descending=True
            )[:5]
            top5_points = optimization_state['train_x'][sorted_indices].cpu().numpy()
            new_points = optimization_state.get('last_new_points', np.array([]))
            iteration_num = optimization_state['current_iteration']
            print(f"  Using current state: Iteration {iteration_num}")
        
        # Create grid
        t_a_grid = np.linspace(-0.5, 0.5, resolution)
        t_b_grid = np.linspace(-0.5, 0.5, resolution)
        T_a, T_b = np.meshgrid(t_a_grid, t_b_grid)
        
        # Generate predictions
        if use_snapshot:
            # For historical iterations, use interpolation (GP model state not available)
            print("  Using interpolation for historical iteration")
            from scipy.interpolate import griddata
            
            # Interpolate mean
            points_2d = train_x[:, :2] if train_x.ndim > 1 else train_x
            values = train_obj.flatten() if train_obj.ndim > 1 else train_obj
            
            mean = griddata(points_2d, values, (T_a, T_b), 
                           method='linear', fill_value=values.min())
            
            # Estimate uncertainty based on distance to nearest point
            from scipy.spatial.distance import cdist
            grid_flat = np.c_[T_a.ravel(), T_b.ravel()]
            distances = cdist(grid_flat, points_2d)
            min_distances = distances.min(axis=1)
            std = min_distances.reshape(resolution, resolution) * 100  # Scale for visibility
        else:
            # For current iteration, use GP model
            print("  Using GP model for current iteration")
            grid_points = torch.tensor(
                np.c_[T_a.ravel(), T_b.ravel()],
                dtype=tkwargs['dtype'],
                device=tkwargs['device']
            )
            
            with torch.no_grad():
                try:
                    optimization_state['model'].eval()
                    posterior = optimization_state['model'].posterior(grid_points)
                    mean = posterior.mean.cpu().numpy().reshape(resolution, resolution)
                    variance = posterior.variance.cpu().numpy().reshape(resolution, resolution)
                    std = np.sqrt(variance)
                except Exception as e:
                    print(f"  Warning: GP prediction failed, using interpolation: {e}")
                    from scipy.interpolate import griddata
                    mean = griddata(train_x, train_obj.flatten(), 
                                   (T_a, T_b), method='linear', fill_value=-1000)
                    std = np.ones_like(mean) * 0.1
        
        print(f"  Mean range: [{mean.min():.6f}, {mean.max():.6f}]")
        print(f"  Std range: [{std.min():.6f}, {std.max():.6f}]")
        
        # Create matplotlib figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Mean
        ax1 = axes[0]
        contour1 = ax1.contourf(T_a, T_b, mean, levels=20, cmap='RdYlBu_r')
        plt.colorbar(contour1, ax=ax1, label='Mean Objective')
        ax1.contour(T_a, T_b, mean, levels=10, colors='black', alpha=0.2, linewidths=0.5)
        
        ax1.scatter(train_x[:, 0], train_x[:, 1],
                   c='blue', marker='o', s=60, 
                   edgecolors='white', linewidths=1.5,
                   label='Evaluated Points', zorder=3)
        
        # Plot new points (black triangles) if available
        if len(new_points) > 0:
            ax1.scatter(new_points[:, 0], new_points[:, 1],
                       c='black', marker='v', s=120,
                       edgecolors='yellow', linewidths=2,
                       label='New Points', zorder=4)
        
        ax1.scatter(top5_points[:, 0], top5_points[:, 1],
                   c='red', marker='D', s=120,
                   edgecolors='white', linewidths=2,
                   label='Top 5', zorder=5)
        
        ax1.scatter([-0.3], [-0.2],
                   c='gold', marker='*', s=400,
                   edgecolors='black', linewidths=2,
                   label='Target', zorder=6)
        
        ax1.set_xlabel('t_a (Nearest-neighbor)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('t_b (Next-nearest)', fontsize=13, fontweight='bold')
        ax1.set_title(f'Mean Prediction - Iteration {iteration_num}', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax1.set_xlim([-0.5, 0.5])
        ax1.set_ylim([-0.5, 0.5])
        ax1.grid(True, alpha=0.3)
        
        # Right: Uncertainty
        ax2 = axes[1]
        contour2 = ax2.contourf(T_a, T_b, std, levels=20, cmap='Greens')
        plt.colorbar(contour2, ax=ax2, label='Uncertainty (Std)')
        ax2.contour(T_a, T_b, std, levels=10, colors='black', alpha=0.2, linewidths=0.5)
        
        ax2.scatter(train_x[:, 0], train_x[:, 1],
                   c='blue', marker='o', s=60,
                   edgecolors='white', linewidths=1.5,
                   label='Evaluated Points', zorder=3)
        
        # Plot new points (black triangles) if available
        if len(new_points) > 0:
            ax2.scatter(new_points[:, 0], new_points[:, 1],
                       c='black', marker='v', s=120,
                       edgecolors='yellow', linewidths=2,
                       label='New Points', zorder=4)
        
        ax2.scatter(top5_points[:, 0], top5_points[:, 1],
                   c='red', marker='D', s=120,
                   edgecolors='white', linewidths=2,
                   label='Top 5', zorder=5)
        
        ax2.scatter([-0.3], [-0.2],
                   c='gold', marker='*', s=400,
                   edgecolors='black', linewidths=2,
                   label='Target', zorder=6)
        
        ax2.set_xlabel('t_a (Nearest-neighbor)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('t_b (Next-nearest)', fontsize=13, fontweight='bold')
        ax2.set_title(f'Uncertainty - Iteration {iteration_num}', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax2.set_xlim([-0.5, 0.5])
        ax2.set_ylim([-0.5, 0.5])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        print(f"  Parameter space plot generated successfully")
        
        return jsonify({
            'image': img_base64,
            'current_iteration': iteration_num,
            'n_points': len(train_x)
        })
        
    except Exception as e:
        print(f"❌ ERROR in parameter space generation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/get_iteration_history', methods=['GET'])
def api_get_iteration_history():
    """
    Get all iteration snapshots
    获取所有迭代快照
    """
    if not optimization_state['iteration_snapshots']:
        return jsonify({
            'iterations': [],
            'message': 'No iteration history available'
        })
    
    return jsonify({
        'iterations': optimization_state['iteration_snapshots'],
        'current_iteration': optimization_state['current_iteration'],
        'total_iterations': optimization_state['total_iterations']
    })


@app.route('/api/plot_multi_dos', methods=['POST'])
def api_plot_multi_dos():
    """
    Generate multi-panel DOS comparison (like original code)
    生成多面板DOS对比图（与原始代码相同）
    Shows Target, BO Suggested, and Final (after local opt) DOS
    """
    data = request.json
    dos_target = np.array(data['dos_target'])
    bins_target = np.array(data['bins_target'])
    candidates = data['candidates']  # List of top candidates
    
    n_plots = len(candidates)
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5), sharey=True)
    if n_plots == 1:
        axes = [axes]
    
    for i, (ax, candidate) in enumerate(zip(axes, candidates)):
        point = candidate['point']
        
        # Normalize target DOS for display
        # integral_target = np.trapz(dos_target, bins_target[:-1])

        try:
            integral_target = np.trapezoid(dos_target, bins_target[:-1])  # NumPy 2.0+
        except AttributeError:
            integral_target = np.trapz(dos_target, bins_target[:-1])      # NumPy 1.x
        
        # Plot target DOS (red)
        ax.plot(bins_target[:-1], dos_target, 'r-', linewidth=3, label='Target DOS')
        
        # Compute BO suggested DOS (black)
        dos_bo, bins_bo = compute_dos(point[0], point[1], n_kpoints=800)
        ax.plot(bins_bo[:-1], dos_bo * integral_target, 'k-', linewidth=3, 
               label='BO Suggested DOS')
        
        # If local optimization was performed, show final DOS (blue)
        if 'optimized_point' in candidate:
            opt_point = candidate['optimized_point']
            dos_final, bins_final = compute_dos(opt_point[0], opt_point[1], n_kpoints=800)
            ax.plot(bins_final[:-1], dos_final * integral_target, 'b-', linewidth=3,
                   label='Final DOS')
        
        ax.set_xlabel('Bias Voltage (eV)', fontsize=14)
        if i == 0:
            ax.set_ylabel('Normalized dI/dV (arb.)', fontsize=14)
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.15, 0.25])
        ax.tick_params(axis='both', labelsize=12)
        
        # Add rank title
        ax.set_title(f'Rank #{i+1}', fontsize=15, fontweight='bold')
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return jsonify({'image': img_base64})


if __name__ == '__main__':
    print("="*70)
    print("Kagome Lattice Optimization Backend (PyTorch/BoTorch)")
    print("Kagome晶格优化后端 (PyTorch/BoTorch)")
    print("="*70)
    print(f"Device: {tkwargs['device']}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch version: {torch.__version__}")
    print("="*70)
    app.run(debug=False, host='0.0.0.0', port=5001, threaded=True)
