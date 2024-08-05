from Poly_tensor_regressor import PolynomialTensorRegression
import numpy as np

decomp_rank = 3  # Decomposition rank
poly_order = 3  # Polynomial order
net_dims = (64, 32)  # Network layer dimensions
reg_lambda = 0.01  # Regularization strength

np.random.seed(42)

X = np.random.rand(100, 3, 2)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
y = np.random.rand(100)
reg_lambda_w = 0.1
reg_lambda_c = 0.1
losses = []

neuron = PolynomialTensorRegression(decomp_rank, 
                                    poly_order, 
                                    method='cp', 
                                    reg_lambda_w=0.01, 
                                    reg_lambda_c=0.01) 

neuron.fit(X,y)
print(neuron.neuron)