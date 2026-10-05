"""The one test every backtest needs: nothing computed for day t may change when data after
t is deleted. Runs on synthetic prices, so no network is needed.  Run: python -m pytest -q"""
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config  # noqa: E402
from engine import pipeline, portfolio  # noqa: E402


def _synthetic(T=1700, seed=7):
    rng = np.random.default_rng(seed)
    names = ["E1", "E2", "E3", "B1", "B2", "G1"]
    cls = pd.Series(["equity", "equity", "equity", "bond", "bond", "gold"], index=names)
    f = np.cumsum(rng.normal(0, 0.01, T))
    r = rng.normal(0.0002, 0.008, (T, len(names))) + np.outer(np.diff(np.r_[0, f]), [1, 1, 1, -0.2, -0.2, 0])
    px = pd.DataFrame(100 * np.exp(np.cumsum(r, 0)), index=pd.bdate_range("2015-01-01", periods=T), columns=names)
    px.iloc[:300, 2] = np.nan                       # an asset that lists later
    return px, cls


def _cfg(mode):
    return replace(Config(), min_train=300, hmm_min_obs=300, train_window=600, mode=mode, core_bond="B1")


@pytest.mark.parametrize("mode", ["core", "core_tilt", "absolute"])
def test_no_lookahead(mode):
    cfg = _cfg(mode)
    px, cls = _synthetic()
    full = pipeline.run(px, cls, cfg)
    for cut in (1150, 1400, 1620):
        part = pipeline.run(px.iloc[:cut + 1], cls, cfg)
        t = px.index[cut]
        for k in ("p_turb", "pred", "target", "core"):
            a, b = full[k].loc[t], part[k].loc[t]
            assert np.allclose(np.asarray(a, float), np.asarray(b, float), atol=1e-9, equal_nan=True), (k, t)
        for k, v in full["features"].items():
            assert np.allclose(v.loc[t].to_numpy(float), part["features"][k].loc[t].to_numpy(float),
                               atol=1e-9, equal_nan=True), (k, t)


@pytest.mark.parametrize("mode", ["core", "core_tilt", "absolute"])
def test_weights_feasible(mode):
    cfg = _cfg(mode)
    px, cls = _synthetic(seed=3)
    out = pipeline.run(px, cls, cfg)
    w, core = out["target"], out["core"]
    assert (w.to_numpy() >= -1e-12).all()
    assert (w.sum(axis=1) <= 1 + 1e-9).all()
    caps = np.maximum(cfg.max_weight, core.to_numpy()) if mode != "absolute" else cfg.max_weight
    assert (w.to_numpy() <= caps + 1e-9).all()
    for k, cap in cfg.class_caps.items():
        assert (w.loc[:, cls.eq(k)].sum(axis=1) <= max(cap, 0) + 1e-9).all(), k
    if mode == "core_tilt":
        # fully invested like the core, and the tilt stays inside the tracking-error budget
        assert np.allclose(w.sum(axis=1), core.sum(axis=1), atol=1e-9)
        rets, live = out["rets"], out["pred"].notna().any(axis=1).to_numpy()
        assert live.sum() > 500
        d = (w - core).to_numpy()
        from engine.portfolio import _cov
        R = rets.to_numpy(float)
        for t in np.flatnonzero(live)[::50]:
            idx = np.flatnonzero(np.isfinite(out["pred"].iloc[t].to_numpy()))
            S = _cov(R, t, idx, cfg)
            te = np.sqrt(d[t, idx] @ S @ d[t, idx] * 252)
            assert te <= cfg.te_cap + 1e-6, te
        assert np.abs(d[live]).sum(axis=1).mean() > 1e-3   # the model is actually tilting


def test_projection():
    rng = np.random.default_rng(0)
    for _ in range(500):
        n = int(rng.integers(2, 12))
        v, caps, B = rng.normal(size=n), rng.uniform(0.05, 0.5, n), rng.uniform(0.3, 1.0)
        w = portfolio.project(v, caps, B)
        assert (w >= 0).all() and (w <= caps + 1e-12).all() and w.sum() <= B + 1e-9
        B2 = rng.uniform(0, caps.sum())
        w2 = portfolio.project_sum(v, caps, B2)
        assert (w2 >= 0).all() and (w2 <= caps + 1e-12).all() and abs(w2.sum() - B2) < 1e-9


def test_core_is_the_6040_benchmark():
    cfg = _cfg("core_tilt")
    px, cls = _synthetic()
    tradable = px.notna()
    core = portfolio.core_weights(tradable, cls, cfg)
    late = core.iloc[-1]
    assert np.isclose(late[["E1", "E2", "E3"]].sum(), 0.6) and np.isclose(late["B1"], 0.4)
    assert np.isclose(core.iloc[10][["E1", "E2"]].sum(), 0.6) and core.iloc[10]["E3"] == 0.0
