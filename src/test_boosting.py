import numpy as np
import pandas as pd
import pytest

from src.gbm import GradientBoosting, LogisticLoss

xgb = pytest.importorskip("xgboost")


def make_data(seed=0, n=3000, d=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    p = 1 / (1 + np.exp(-(-1.0 + 1.5 * X[:, 2] - X[:, 4])))
    y = (rng.random(n) < p).astype(float)
    return X, y


# ---- mot ekte xgboost over flere runder (NaN-frie data) ----------------------------
@pytest.mark.parametrize("n_estimators", [1, 3, 10])
@pytest.mark.parametrize("learning_rate", [1.0, 0.3])
@pytest.mark.parametrize("depth", [2, 3])
def test_matches_xgboost(n_estimators, learning_rate, depth):
    X, y = make_data()
    mine = GradientBoosting(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=depth,
                            lam=1.0, gamma=0.0, min_child_weight=1.0).fit(X, y)
    m = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=depth,
                          reg_lambda=1.0, gamma=0.0, min_child_weight=1.0,
                          tree_method="exact", base_score=y.mean(), n_jobs=1).fit(X, y)
    assert np.allclose(mine.predict_raw(X), m.predict(X, output_margin=True), atol=1e-4)
    assert np.allclose(mine.predict_proba(X), m.predict_proba(X), atol=1e-4)


# ---- grensesnitt og bokføring ------------------------------------------------------
def test_predict_proba_shape_and_range():
    X, y = make_data()
    P = GradientBoosting(n_estimators=5, learning_rate=0.3, max_depth=3).fit(X, y).predict_proba(X)
    assert P.shape == (len(y), 2)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert (P >= 0).all() and (P <= 1).all()


def test_zero_trees_gives_base_rate():
    X, y = make_data()
    P = GradientBoosting(n_estimators=0).fit(X, y).predict_proba(X)
    assert np.allclose(P[:, 1], y.mean())


def test_training_loss_decreases():
    X, y = make_data()
    b = GradientBoosting(n_estimators=20, learning_rate=0.3, max_depth=3).fit(X, y)
    assert np.all(np.diff(b.train_loss_) <= 1e-12)


def test_truncated_prediction_equals_smaller_model():
    X, y = make_data()
    big = GradientBoosting(n_estimators=10, learning_rate=0.3, max_depth=3).fit(X, y)
    small = GradientBoosting(n_estimators=3, learning_rate=0.3, max_depth=3).fit(X, y)
    assert np.allclose(big.predict_raw(X, n_trees=3), small.predict_raw(X))


def test_validation_tracking_matches_full_prediction():
    X, y = make_data(0)
    Xv, yv = make_data(1)
    b = GradientBoosting(n_estimators=8, learning_rate=0.3, max_depth=3).fit(X, y, eval_set=(Xv, yv))
    full = LogisticLoss().loss(yv, b.predict_raw(Xv)).mean()
    assert np.isclose(b.val_loss_[-1], full)


def test_early_stopping():
    X, y = make_data(0, n=4000)
    Xv, yv = make_data(99, n=2000)
    b = GradientBoosting(n_estimators=300, learning_rate=0.3, max_depth=3,
                         early_stopping_rounds=10).fit(X, y, eval_set=(Xv, yv))
    assert len(b.trees_) < 300
    assert len(b.val_loss_) == len(b.trees_)
    assert b.best_iteration_ == int(np.argmin(b.val_loss_))


def test_pandas_input_with_scrambled_index():
    X, y = make_data()
    Xdf = pd.DataFrame(X).sample(frac=1, random_state=0)           # stokket indeks
    ydf = pd.Series(y[Xdf.index.to_numpy()], index=Xdf.index)
    kw = dict(n_estimators=5, learning_rate=0.3, max_depth=3)
    a = GradientBoosting(**kw).fit(Xdf, ydf)
    b = GradientBoosting(**kw).fit(Xdf.to_numpy(), ydf.to_numpy())
    assert np.allclose(a.predict_raw(Xdf), b.predict_raw(Xdf.to_numpy()))