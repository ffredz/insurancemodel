from src.gbm import LogisticLoss, SquaredLoss

import numpy as np 

def test_log_loss():
    L = LogisticLoss()
    rng = np.random.default_rng(0)
    z = rng.normal(0,3,1000)
    y = rng.binomial(1,0.3,1000).astype(float)
    eps = 1e-5
    num_grad = (L.loss(y, z+eps) - L.loss(y, z-eps)) / (2*eps)
    num_hess = (L.grad(y, z+eps) - L.grad(y, z-eps)) / (2*eps)
    assert np.allclose(num_grad, L.grad(y, z), atol=1e-6)
    assert np.allclose(num_hess, L.hess(y, z), atol=1e-6)
    assert abs(L.grad(y, np.full(len(y), L.init_score(y))).sum()) < 1e-8

if __name__ == "__main__":
    test_log_loss()
    print("OK")

