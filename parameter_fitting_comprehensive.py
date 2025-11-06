"""
Comprehensive Parameter Fitting for Parametric Curve
Tests multiple optimization approaches to find best solution for:
  x = t*cos(theta) - exp(M*|t|)*sin(0.3*t)*sin(theta) + X
  y = 42 + t*sin(theta) + exp(M*|t|)*sin(0.3*t)*cos(theta)

Unknowns: theta (0-50 deg), M (-0.05 to 0.05), X (0-100)
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, differential_evolution, basinhopping
from scipy.spatial.distance import cdist
import time
from datetime import datetime
import json
import os

# Configuration
CSV_PATH = 'xy_data.csv'
BEST_RESULT_FILE = 'best_solution.txt'
LOG_FILE = 'optimization_log.txt'

# Parameter bounds
THETA_MIN_DEG = 0.001
THETA_MAX_DEG = 50.0
M_MIN = -0.05 + 1e-6
M_MAX = 0.05 - 1e-6
X_MIN = 0.0 + 1e-6
X_MAX = 100.0 - 1e-6

# Load data
print("Loading data...")
df = pd.read_csv(CSV_PATH)
xy_data = df[['x', 'y']].to_numpy()
print(f"Loaded {len(xy_data)} data points")

# For L1 evaluation: uniform t grid
T_GRID = np.linspace(6.0, 60.0, 1000)


def model_curve(t, theta_rad, M, X):
    """Generate curve points for given parameters and t values"""
    exp_term = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    x = t * c - exp_term * s + X
    y = 42.0 + t * s + exp_term * c
    return np.column_stack([x, y])


def compute_l1_distance(theta_rad, M, X):
    """
    Compute L1 distance between uniformly sampled curve points and data.
    This is the primary evaluation metric.
    """
    # Generate curve points at uniform t grid
    curve_xy = model_curve(T_GRID, theta_rad, M, X)
    
    # For each curve point, find nearest data point (L1 distance)
    distances = cdist(curve_xy, xy_data, metric='cityblock')
    min_distances = np.min(distances, axis=1)
    
    # Return mean L1 distance
    return np.mean(min_distances)


def transform_to_uv(xy_points, theta_rad, X):
    """Transform (x,y) to (u,v) space after translation and rotation"""
    x = xy_points[:, 0]
    y = xy_points[:, 1]
    x_prime = x - X
    y_prime = y - 42.0
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    u = x_prime * c + y_prime * s
    v = -x_prime * s + y_prime * c
    return u, v


# ============================================================================
# Residual Functions (Different Approaches)
# ============================================================================

def residuals_uv_space(p):
    """Residual in transformed (u,v) space"""
    theta_rad, M, X = p
    u, v = transform_to_uv(xy_data, theta_rad, X)
    pred_v = np.exp(M * u) * np.sin(0.3 * u)
    res = v - pred_v
    # Penalty for u outside [6, 60]
    u_penalty = np.sum(np.clip(6.0 - u, 0, None)**2) + np.sum(np.clip(u - 60.0, 0, None)**2)
    return np.concatenate([res, 0.1 * np.sqrt(u_penalty) * np.ones(1)])


def residuals_direct_xy(p):
    """Residual in original (x,y) space"""
    theta_rad, M, X = p
    # For each data point, find best matching t
    u, v = transform_to_uv(xy_data, theta_rad, X)
    # u should be close to t, so use u as t estimate
    t_est = np.clip(u, 6.0, 60.0)
    curve_pred = model_curve(t_est, theta_rad, M, X)
    res = xy_data - curve_pred
    return res.flatten()


def residuals_hybrid(p):
    """Hybrid: combine uv space and direct xy residuals"""
    theta_rad, M, X = p
    u, v = transform_to_uv(xy_data, theta_rad, X)
    pred_v = np.exp(M * u) * np.sin(0.3 * u)
    res_uv = v - pred_v
    
    # Also include direct xy residual
    t_est = np.clip(u, 6.0, 60.0)
    curve_pred = model_curve(t_est, theta_rad, M, X)
    res_xy = (xy_data - curve_pred).flatten()
    
    # Weight uv space more heavily
    return np.concatenate([res_uv * 2.0, res_xy * 0.5])


# ============================================================================
# Optimization Methods
# ============================================================================

def bounds_array():
    """Get bounds as array for scipy optimizers"""
    return [
        (np.deg2rad(THETA_MIN_DEG), np.deg2rad(THETA_MAX_DEG)),
        (M_MIN, M_MAX),
        (X_MIN, X_MAX)
    ]


def optimize_least_squares(residual_func, initial_guess, method='lm', loss='linear'):
    """Optimize using scipy least_squares"""
    try:
        # 'lm' method doesn't support bounds, only 'trf' and 'dogbox' do
        kwargs = {
            'fun': residual_func,
            'x0': initial_guess,
            'method': method,
            'loss': loss,
            'max_nfev': 5000,
            'verbose': 0
        }
        
        # Only add bounds for methods that support them
        if method in ['trf', 'dogbox']:
            kwargs['bounds'] = (
                [np.deg2rad(THETA_MIN_DEG), M_MIN, X_MIN],
                [np.deg2rad(THETA_MAX_DEG), M_MAX, X_MAX]
            )
        
        result = least_squares(**kwargs)
        
        # Clip parameters to bounds if method doesn't support bounds (e.g., 'lm')
        if method == 'lm':
            params = result.x.copy()
            params[0] = np.clip(params[0], np.deg2rad(THETA_MIN_DEG), np.deg2rad(THETA_MAX_DEG))
            params[1] = np.clip(params[1], M_MIN, M_MAX)
            params[2] = np.clip(params[2], X_MIN, X_MAX)
            return params, result.cost, result.success
        
        return result.x, result.cost, result.success
    except Exception as e:
        print(f"  Error in least_squares: {e}")
        return None, None, False


def optimize_differential_evolution(residual_func):
    """Optimize using differential evolution (global optimizer)"""
    try:
        def objective(p):
            return np.sum(residual_func(p)**2)
        
        result = differential_evolution(
            objective,
            bounds_array(),
            seed=42,
            maxiter=1000,
            popsize=15,
            atol=1e-8,
            tol=1e-8,
            workers=1
        )
        return result.x, result.fun, result.success
    except Exception as e:
        print(f"  Error in differential_evolution: {e}")
        return None, None, False


def optimize_basin_hopping(residual_func, initial_guess):
    """Optimize using basin hopping"""
    try:
        def objective(p):
            return np.sum(residual_func(p)**2)
        
        result = basinhopping(
            objective,
            initial_guess,
            niter=100,
            T=1.0,
            stepsize=0.1,
            minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bounds_array()},
            seed=42
        )
        return result.x, result.fun, result.success
    except Exception as e:
        print(f"  Error in basin_hopping: {e}")
        return None, None, False


# ============================================================================
# Main Optimization Loop
# ============================================================================

class BestSolutionTracker:
    def __init__(self):
        self.best_l1 = float('inf')
        self.best_params = None
        self.best_method = None
        self.best_residual = None
        self.best_timestamp = None
    
    def evaluate(self, params, method_name, residual_cost):
        """Evaluate solution and update best if needed"""
        if params is None:
            return False
        
        theta_rad, M, X = params
        
        # Check bounds
        if not (np.deg2rad(THETA_MIN_DEG) <= theta_rad <= np.deg2rad(THETA_MAX_DEG)):
            return False
        if not (M_MIN <= M <= M_MAX):
            return False
        if not (X_MIN <= X <= X_MAX):
            return False
        
        # Compute L1 distance (primary metric)
        l1_dist = compute_l1_distance(theta_rad, M, X)
        
        if l1_dist < self.best_l1:
            self.best_l1 = l1_dist
            self.best_params = params.copy()
            self.best_method = method_name
            self.best_residual = residual_cost
            self.best_timestamp = datetime.now().isoformat()
            return True
        
        return False
    
    def save_best(self):
        """Save best solution to file"""
        if self.best_params is None:
            return
        
        theta_rad, M, X = self.best_params
        theta_deg = np.rad2deg(theta_rad)
        
        # Generate LaTeX string
        latex = (f"\\left(t*\\cos({theta_rad:.10f})-e^{{{M:.10f}\\left|t\\right|}}\\cdot\\sin(0.3t)\\sin({theta_rad:.10f})+{X:.10f},"
                 f"42+t*\\sin({theta_rad:.10f})+e^{{{M:.10f}\\left|t\\right|}}\\cdot\\sin(0.3t)\\cos({theta_rad:.10f})\\right)")
        
        # Create detailed report
        report = f"""
================================================================================
BEST SOLUTION FOUND
================================================================================
Timestamp: {self.best_timestamp}
Method: {self.best_method}

Parameters:
  theta = {theta_deg:.10f} degrees ({theta_rad:.10f} radians)
  M     = {M:.10f}
  X     = {X:.10f}

Evaluation Metrics:
  L1 Distance (mean): {self.best_l1:.10f}
  Residual Cost:     {self.best_residual:.10e}

LaTeX Submission String:
{latex}

For Desmos Calculator:
Copy the LaTeX string above and paste it into:
https://www.desmos.com/calculator/rfj91yrxob

================================================================================
"""

        # Save to file
        with open(BEST_RESULT_FILE, 'w') as f:
            f.write(report)
        
        print("\n" + "="*80)
        print("BEST SOLUTION UPDATED!")
        print("="*80)
        print(f"L1 Distance: {self.best_l1:.10f}")
        print(f"Method: {self.best_method}")
        print(f"theta: {theta_deg:.10f} deg, M: {M:.10f}, X: {X:.10f}")
        print(f"Saved to: {BEST_RESULT_FILE}")
        print("="*80 + "\n")


def generate_initial_guesses(n=20):
    """Generate diverse initial guesses"""
    guesses = []
    
    # Systematic grid
    theta_grid = np.linspace(5, 45, 5)
    M_grid = np.linspace(-0.04, 0.04, 5)
    X_grid = np.linspace(10, 90, 5)
    
    for theta_deg in theta_grid:
        for M in M_grid:
            for X in X_grid:
                guesses.append([np.deg2rad(theta_deg), M, X])
    
    # Random samples
    np.random.seed(42)
    for _ in range(n):
        theta_deg = np.random.uniform(THETA_MIN_DEG, THETA_MAX_DEG)
        M = np.random.uniform(M_MIN, M_MAX)
        X = np.random.uniform(X_MIN, X_MAX)
        guesses.append([np.deg2rad(theta_deg), M, X])
    
    return guesses


def run_comprehensive_optimization(max_iterations=None, quick_test=False):
    """Run comprehensive optimization with multiple methods"""
    tracker = BestSolutionTracker()
    
    # Initial guesses (fewer for quick test)
    n_guesses = 10 if quick_test else 30
    initial_guesses = generate_initial_guesses(n_guesses)
    
    # Residual functions to test
    residual_functions = [
        ("residuals_uv_space", residuals_uv_space),
        ("residuals_direct_xy", residuals_direct_xy),
        ("residuals_hybrid", residuals_hybrid),
    ]
    
    iteration = 0
    start_time = time.time()
    last_save_time = start_time
    
    print("\n" + "="*80)
    print("STARTING COMPREHENSIVE PARAMETER OPTIMIZATION")
    print("="*80)
    print(f"Mode: {'QUICK TEST' if quick_test else 'FULL OPTIMIZATION'}")
    print(f"Total initial guesses: {len(initial_guesses)}")
    print(f"Residual functions: {len(residual_functions)}")
    print("="*80 + "\n")
    
    # Open log file
    log_handle = open(LOG_FILE, 'w') if LOG_FILE else None
    
    try:
        # Test 1: Least squares with different methods and losses
        print("\n[1] Testing least_squares with different configurations...")
        for res_name, residual_func in residual_functions:
            for method in ['lm', 'trf', 'dogbox']:
                # 'lm' method only supports 'linear' loss
                losses = ['linear'] if method == 'lm' else ['linear', 'huber', 'soft_l1']
                for loss in losses:
                    n_guesses_test = 5 if quick_test else 10
                    for i, p0 in enumerate(initial_guesses[:n_guesses_test]):  # Limit for speed
                        iteration += 1
                        method_name = f"{res_name}_least_squares_{method}_{loss}_init{i}"
                        
                        params, cost, success = optimize_least_squares(
                            residual_func, p0, method=method, loss=loss
                        )
                        
                        if success and params is not None:
                            updated = tracker.evaluate(params, method_name, cost)
                            if updated:
                                tracker.save_best()
                                if log_handle:
                                    log_handle.write(f"{datetime.now().isoformat()}: {method_name}\n")
                                    log_handle.flush()
                        
                        if max_iterations and iteration >= max_iterations:
                            break
                        if iteration % 10 == 0:
                            elapsed = time.time() - start_time
                            print(f"  Progress: {iteration} iterations, {elapsed:.1f}s elapsed, "
                                  f"Best L1: {tracker.best_l1:.8f}")
                        
                        # Periodic save every 30 seconds
                        if time.time() - last_save_time > 30:
                            tracker.save_best()
                            last_save_time = time.time()
                    if max_iterations and iteration >= max_iterations:
                        break
                if max_iterations and iteration >= max_iterations:
                    break
            if max_iterations and iteration >= max_iterations:
                break
        
        # Test 2: Differential Evolution (global optimizer)
        print("\n[2] Testing differential_evolution...")
        for res_name, residual_func in residual_functions:
            iteration += 1
            method_name = f"{res_name}_differential_evolution"
            
            params, cost, success = optimize_differential_evolution(residual_func)
            
            if success and params is not None:
                updated = tracker.evaluate(params, method_name, cost)
                if updated:
                    tracker.save_best()
                    if log_handle:
                        log_handle.write(f"{datetime.now().isoformat()}: {method_name}\n")
                        log_handle.flush()
            
            if max_iterations and iteration >= max_iterations:
                break
        
        # Test 3: Basin hopping with different starting points
        print("\n[3] Testing basin_hopping...")
        n_guesses_bh = 5 if quick_test else 15
        for res_name, residual_func in residual_functions:
            for i, p0 in enumerate(initial_guesses[:n_guesses_bh]):
                iteration += 1
                method_name = f"{res_name}_basin_hopping_init{i}"
                
                params, cost, success = optimize_basin_hopping(residual_func, p0)
                
                if success and params is not None:
                    updated = tracker.evaluate(params, method_name, cost)
                    if updated:
                        tracker.save_best()
                        if log_handle:
                            log_handle.write(f"{datetime.now().isoformat()}: {method_name}\n")
                            log_handle.flush()
                
                if max_iterations and iteration >= max_iterations:
                    break
            if max_iterations and iteration >= max_iterations:
                break
        
        # Test 4: Refinement: Use best solution as starting point for more iterations
        print("\n[4] Refining best solution with additional optimizations...")
        if tracker.best_params is not None:
            for res_name, residual_func in residual_functions:
                for method in ['lm', 'trf']:
                    iteration += 1
                    method_name = f"refine_{res_name}_{method}"
                    
                    params, cost, success = optimize_least_squares(
                        residual_func, tracker.best_params, method=method, loss='linear'
                    )
                    
                    if success and params is not None:
                        updated = tracker.evaluate(params, method_name, cost)
                        if updated:
                            tracker.save_best()
                            if log_handle:
                                log_handle.write(f"{datetime.now().isoformat()}: {method_name}\n")
                                log_handle.flush()
                    
                    if max_iterations and iteration >= max_iterations:
                        break
                if max_iterations and iteration >= max_iterations:
                    break
        
        # Run extended search if no max_iterations limit and not quick test
        if max_iterations is None and not quick_test:
            print("\n[5] Running extended search with best parameters...")
            # Generate more random starting points near best solution
            best_params = tracker.best_params
            for _ in range(50):
                iteration += 1
                # Perturb best solution
                p0 = best_params.copy()
                p0[0] += np.random.uniform(-0.1, 0.1)  # theta
                p0[1] += np.random.uniform(-0.001, 0.001)  # M
                p0[2] += np.random.uniform(-5, 5)  # X
                p0[0] = np.clip(p0[0], np.deg2rad(THETA_MIN_DEG), np.deg2rad(THETA_MAX_DEG))
                p0[1] = np.clip(p0[1], M_MIN, M_MAX)
                p0[2] = np.clip(p0[2], X_MIN, X_MAX)
                
                for res_name, residual_func in residual_functions[:1]:  # Just use uv_space
                    method_name = f"extended_{res_name}_init{iteration}"
                    
                    params, cost, success = optimize_least_squares(
                        residual_func, p0, method='lm', loss='linear'
                    )
                    
                    if success and params is not None:
                        updated = tracker.evaluate(params, method_name, cost)
                        if updated:
                            tracker.save_best()
                            if log_handle:
                                log_handle.write(f"{datetime.now().isoformat()}: {method_name}\n")
                                log_handle.flush()
                    
                    if iteration % 20 == 0:
                        elapsed = time.time() - start_time
                        print(f"  Extended search: {iteration} iterations, {elapsed:.1f}s elapsed, "
                              f"Best L1: {tracker.best_l1:.10f}")
    
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
        print("Saving current best solution...")
        tracker.save_best()
        if log_handle:
            log_handle.write(f"{datetime.now().isoformat()}: Interrupted by user\n")
            log_handle.close()
        elapsed = time.time() - start_time
        print(f"\nInterrupted after {elapsed:.1f} seconds")
        print(f"Best solution saved to: {BEST_RESULT_FILE}")
        return tracker
    except Exception as e:
        print(f"\n\nError during optimization: {e}")
        print("Saving current best solution...")
        tracker.save_best()
        if log_handle:
            log_handle.write(f"{datetime.now().isoformat()}: Error - {str(e)}\n")
            log_handle.close()
        elapsed = time.time() - start_time
        print(f"\nError after {elapsed:.1f} seconds")
        print(f"Best solution saved to: {BEST_RESULT_FILE}")
        return tracker
    finally:
        if log_handle:
            log_handle.close()
    
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Total iterations: {iteration}")
    print(f"Total time: {elapsed:.1f} seconds")
    print(f"Best L1 distance: {tracker.best_l1:.10f}")
    print(f"Best method: {tracker.best_method}")
    print("="*80 + "\n")
    
    # Final save
    tracker.save_best()
    
    return tracker


if __name__ == "__main__":
    import sys
    
    print("="*80)
    print("PARAMETER FITTING - COMPREHENSIVE OPTIMIZATION")
    print("="*80)
    print("\nThis script will test multiple optimization approaches.")
    print("Best solution will be saved to:", BEST_RESULT_FILE)
    
    # Check for quick test mode
    quick_test = '--quick' in sys.argv or '--test' in sys.argv
    if quick_test:
        print("\nMode: QUICK TEST (limited iterations)")
        print("Run without --quick for full optimization")
    else:
        print("\nMode: FULL OPTIMIZATION (can run for hours)")
        print("Add --quick flag for quick test mode")
    
    print("\nStarting optimization...\n")
    
    # Run optimization
    # Set max_iterations=None for full run, or a number to limit
    max_iter = None if not quick_test else 50
    tracker = run_comprehensive_optimization(max_iterations=max_iter, quick_test=quick_test)
    
    print(f"\nFinal best solution saved to: {BEST_RESULT_FILE}")
    print("Check the file for the LaTeX submission string!")

