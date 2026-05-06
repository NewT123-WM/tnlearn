import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed

def _run_single_bfgs(loss_func, n_params, seed):
    """
    Runs a single BFGS optimization with a random starting point.
    A random seed is used to ensure that each parallel process
    generates a different random starting point.
    """
    # 使用传入的seed来初始化此进程的随机数生成器
    np.random.seed(seed)
    x0 = np.random.uniform(low=-1, high=1, size=n_params)
    result = minimize(
        loss_func,
        x0,
        method='BFGS',
        options={
            'maxiter': 200,
            'gtol': 1e-10,
            'eps': 1e-12,
            'disp': False
        }
    )
    return result

def parallel_multi_start_bfgs(loss_func, n_starts=10, n_params=10):
    """
    Runs BFGS from multiple random starting points in parallel.
    """
    # 为每个并行作业生成一个唯一的随机种子
    seeds = np.random.randint(np.iinfo(np.int32).max, size=n_starts)
    
    results = Parallel(n_jobs=n_starts)(
        delayed(_run_single_bfgs)(loss_func, n_params, seed) for seed in seeds
    )

    best_result = None
    for result in results:
        if best_result is None or result.fun < best_result.fun:
            best_result = result
            
    return best_result
