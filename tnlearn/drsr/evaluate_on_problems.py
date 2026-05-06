import numpy as np
MAX_NPARAMS = 10
params = [1.0]*MAX_NPARAMS
from scipy.optimize import minimize

 # 全局变量定义保留小数点的位数
DECIMAL_PLACES = 3

def multi_start_bfgs(loss_func, n_starts=5, n_params=10):
    best_result = None
    for i in range(n_starts):
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
        if best_result is None or result.fun < best_result.fun:
            best_result = result
    return best_result

def evaluate(data: dict , equation, verbose: bool = False) -> float:
        """ 从大模型的输出program中直接获取的"""
        if verbose:
            print('我运行了!')

        inputs, outputs = data['inputs'], data['outputs']
        X = inputs
        X_rounded = np.round(X, DECIMAL_PLACES)
        # Optimize parameters based on data
        
        def loss(params):
            y_pred = equation(*X.T, params)
            return np.mean((y_pred - outputs) ** 2)

        loss_partial = lambda params: loss(params)
        result = multi_start_bfgs(loss_partial, n_params=MAX_NPARAMS)

        # Return evaluation score
        optimized_params = result.x
        loss = result.fun
        if np.isnan(loss) or np.isinf(loss):
            return None
        else:
            # 计算并输出优化后的方程在输入数据上的预测结果
            optimized_predictions = equation(*X.T, optimized_params)
            # 计算残差（实际值 - 预测值）
            res = outputs - optimized_predictions
            #计算output的var
            var_outputs = np.var(outputs)
            nmse = loss/var_outputs if var_outputs != 0 else np.inf
            # 计算R²指标
            r2 = 1 - nmse
            
            # 打印R²指标和NMSE
            if verbose:
                print(f'R² 指标: {r2:.6f}')
                print(f'NMSE 指标: {nmse:.6f}')
            
            # 使用全局变量保留预测结果和残差的小数位数
            res_rounded = np.round(res, DECIMAL_PLACES)
            outputs_rounded = np.round(outputs, DECIMAL_PLACES)  # 计算输出的四舍五入值
            # 直接返回数据矩阵，使用X_rounded作为输入数据
            result_data = np.column_stack((X_rounded, outputs_rounded, res_rounded))
            result_data = np.round(result_data, DECIMAL_PLACES)  # 保留小数点位数
            # 兼容：返回分数、结果矩阵、BFGS 优化得到的参数
            return -loss, result_data, optimized_params
