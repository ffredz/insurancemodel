import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

from src.gbm import Tree, LogisticLoss, SquaredLoss


def make_regression_data(n=1500, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = 3 * X[:, 3] + 1.0 * (X[:, 1] > 0) + rng.normal(size=n)
    return X, y



@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_tree_matches_sklearn_squared_loss(depth):
    X, y = make_regression_data()
    loss = SquaredLoss()
    z = np.zeros(len(y))                                 
    g, h = loss.grad(y, z), loss.hess(y, z)               

    mine = Tree(max_depth=depth, lam=0.0, gamma=0.0, min_child_weight=0.0).fit(X, g, h).predict(X)
    sk = DecisionTreeRegressor(max_depth=depth).fit(X, y).predict(X)

    assert np.allclose(mine, sk, atol=1e-9)


def test_depth_zero_is_single_leaf():
    X, y = make_regression_data()
    g, h = -y, np.ones(len(y))
    lam = 2.0
    pred = Tree(max_depth=0, lam=lam).fit(X, g, h).predict(X)
    assert np.allclose(pred, -g.sum() / (h.sum() + lam))


def test_huge_gamma_gives_single_leaf():
    X, y = make_regression_data()
    g, h = -y, np.ones(len(y))
    pred = Tree(max_depth=4, lam=1.0, gamma=1e12, min_child_weight=0.0).fit(X, g, h).predict(X)
    assert len(np.unique(pred)) == 1


def test_min_child_weight_is_respected():
    X, y = make_regression_data()
    g, h = -y, np.ones(len(y))                          
    mcw = 50
    pred = Tree(max_depth=4, lam=0.0, gamma=0.0, min_child_weight=mcw).fit(X, g, h).predict(X)
    _, counts = np.unique(pred, return_counts=True)        
    assert counts.min() >= mcw


xgb = pytest.importorskip("xgboost")


def make_classification_data(n=3000, d=6, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    p = 1 / (1 + np.exp(-(1.5 * X[:, 2] - X[:, 4])))
    y = (rng.random(n) < p).astype(float)
    z0 = rng.normal(0, 1, n)      
    return X, y, z0


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("lam, gamma, mcw", [
    (1.0, 0.0, 0.0),
    (1.0, 0.0, 1.0),
    (5.0, 0.0, 0.5),
    (200.0, 0.0, 0.0),      
])
def test_tree_matches_xgboost_logistic(depth, lam, gamma, mcw):
    X, y, z0 = make_classification_data()
    loss = LogisticLoss()
    g, h = loss.grad(y, z0), loss.hess(y, z0)

    mine = Tree(max_depth=depth, lam=lam, gamma=gamma, min_child_weight=mcw).fit(X, g, h).predict(X)

    m = xgb.XGBClassifier(
        n_estimators=1, max_depth=depth, learning_rate=1.0,
        reg_lambda=lam, gamma=gamma, min_child_weight=mcw,
        tree_method="exact", n_jobs=1,
    )
    m.fit(X, y, base_margin=z0)
    ref = m.predict(X, output_margin=True, base_margin=z0) - z0   

    assert np.allclose(mine, ref, atol=1e-4)


def test_gamma_pruning_matches_xgboost():
   
    depth, lam, gamma, mcw = 3, 1.0, 50.0, 0.0
    X, y, z0 = make_classification_data()
    loss = LogisticLoss()
    g, h = loss.grad(y, z0), loss.hess(y, z0)

    mine = Tree(max_depth=depth, lam=lam, gamma=gamma, min_child_weight=mcw).fit(X, g, h).predict(X)

    m = xgb.XGBClassifier(n_estimators=1, max_depth=depth, learning_rate=1.0, reg_lambda=lam,
                          gamma=gamma, min_child_weight=mcw, tree_method="exact", n_jobs=1)
    m.fit(X, y, base_margin=z0)
    ref = m.predict(X, output_margin=True, base_margin=z0) - z0

    n_leaves = len(np.unique(np.round(mine, 8)))
    assert n_leaves < 2 ** depth        
    assert np.allclose(mine, ref, atol=1e-4)


print("done")