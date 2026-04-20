"""
src/models/tabular.py — Tabular-only model implementations for Cancer Research.

This file is the reference implementation for the tabular_only model category.
It contains three models in increasing complexity:

  DecisionTreeRegressor   ← single tree; interpretable, good baseline (reference example)
  RandomForestRegressor   ← bagged ensemble; better generalisation
  GradientBoostedTrees    ← sklearn GBT; strong baseline before XGBoost/CatBoost

All three share the same external API, which is what the framework requires:
  - __init__(cfg)             reads all hyperparameters from DotDict config
  - fit(X, y)                 sklearn-style training (called by _train_tree in train.py)
  - predict(X)                returns float32 (N,) array
  - __call__(image, tabular)  protocol bridge so the registry type-check passes

Registration (already done in src/models/__init__.py):
  _TABULAR_REGISTRY = {
      ...
      "decision_tree":    lambda cfg: DecisionTreeRegressor(cfg),
      "random_forest":    lambda cfg: RandomForestRegressor(cfg),
      "gradient_boosted": lambda cfg: GradientBoostedTrees(cfg),
  }

Example YAML config (configs/experiments/decision_tree_baseline.yaml):
  experiment_name: dt_baseline

  modalities:
    tabular: true
    image: false

  model:
    category: tabular_only
    type: decision_tree
    max_depth: 5           # None = fully grown (overfits); start with 3–6
    min_samples_split: 10  # minimum samples to split a node
    min_samples_leaf: 5    # minimum samples in each leaf
    criterion: squared_error   # loss: squared_error | friedman_mse | absolute_error
    random_state: 42

  dataset:
    data_root: data
    tabular_file: clinical.csv
    patient_id_col: PATIENT_ID
    target_col: survival_months
    val_ratio: 0.15
    seed: 42
    batch_size: 256   # large batch — tree models drain the loader in one pass anyway

  training:
    save_dir: logs/runs/checkpoints
    risk_threshold: 24.0   # months — used to compute AUROC / AUPRC in utils

  wandb:
    project: TMB-prediciton
    run_name: dt_baseline_depth5

⚠ Shared code — see STANDARDS §5 before modifying.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.preprocessing import RobustScaler

from src.config import DotDict

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal base class — shared boilerplate for all tabular models
# ---------------------------------------------------------------------------

class _TabularBase:
    """Shared scaffold for sklearn-compatible tabular regressors.

    Subclasses must implement:
        _build_estimator(cfg) → sklearn estimator
        _category_label       → str (for log messages)

    This base class provides:
        fit(X, y)                 — delegates to self._est.fit
        predict(X)               — delegates to self._est.predict, returns float32
        __call__(image, tabular) — protocol bridge for CancerModelProtocol
        feature_importances_     — property (if the estimator supports it)
    """

    _category_label: str = "tabular"

    def __init__(self, cfg: DotDict) -> None:
        self._cfg = cfg
        self._est = self._build_estimator(cfg)
        self._fitted = False
        log.info("Initialised %s", self.__class__.__name__)

    def _build_estimator(self, cfg: DotDict) -> Any:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # sklearn-style API — called by _train_tree in train.py
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_TabularBase":
        """Fit the model on the full training matrix.

        Args:
            X: float32 array of shape (N, F) — tabular features.
            y: float32 array of shape (N,)   — regression targets.

        Returns:
            self (sklearn convention — allows chaining).
        """
        log.info(
            "Fitting %s on X=%s, y=%s",
            self.__class__.__name__, X.shape, y.shape,
        )
        self._est.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regression targets for *X*.

        Args:
            X: float32 array of shape (N, F).

        Returns:
            float32 array of shape (N,).

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._fitted:
            raise RuntimeError(
                f"{self.__class__.__name__}.predict() called before fit(). "
                "Call fit(X, y) first, or load a checkpoint."
            )
        preds = self._est.predict(X)
        return preds.astype(np.float32)

    # ------------------------------------------------------------------
    # CancerModelProtocol bridge
    # ------------------------------------------------------------------

    def __call__(
        self,
        image:   Any | None,    # ignored — tabular_only model
        tabular: np.ndarray | None,
    ) -> np.ndarray:
        """Protocol-compatible forward pass.

        Allows the framework to call model(image, tabular) uniformly
        without special-casing tree models in the evaluation loop.

        Args:
            image:   Ignored. Included to satisfy CancerModelProtocol.
            tabular: (B, F) float32 array.

        Returns:
            (B,) float32 predictions.
        """
        if tabular is None:
            raise ValueError(
                f"{self.__class__.__name__} is a tabular_only model — "
                "tabular input cannot be None."
            )
        # Accept both numpy arrays and torch tensors
        if hasattr(tabular, "numpy"):
            tabular = tabular.numpy()
        return self.predict(tabular)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def feature_importances_(self) -> np.ndarray | None:
        """Return Gini / impurity-based feature importances (if available).

        Shape: (F,) float64, sums to 1.0.
        Returns None for estimators that do not expose this attribute.
        """
        return getattr(self._est, "feature_importances_", None)

    def print_feature_importances(
        self,
        feature_names: list[str] | None = None,
        top_n: int = 20,
    ) -> None:
        """Log the top-N most important features to stdout.

        Args:
            feature_names: Column names from the tabular CSV (optional).
            top_n:         How many to print.
        """
        imps = self.feature_importances_
        if imps is None:
            log.warning("%s does not expose feature importances.", self.__class__.__name__)
            imps = self._est[-1].coef_

        order = np.argsort(imps)[::-1][:top_n]
        log.info("Top-%d feature importances for %s:", top_n, self.__class__.__name__)
        for rank, idx in enumerate(order, start=1):
            name = feature_names[idx] if feature_names and idx < len(feature_names) else f"feature_{idx}"
            log.info("  %2d. %-30s %.4f", rank, name, imps[idx])


# ---------------------------------------------------------------------------
# 1. Decision Tree — reference implementation
# ---------------------------------------------------------------------------

class DecisionTreeRegressor(_TabularBase):
    """Single decision tree for regression.

    This is the reference model implementation for the Zerone framework.
    It is highly interpretable — every prediction can be traced to a specific
    sequence of feature thresholds — which makes it useful for:
      - Sanity-checking the tabular pipeline (should overfit easily)
      - Feature importance analysis before training heavier models
      - Generating a fast clinical-rule baseline

    Limitations:
      - High variance: sensitive to small changes in training data.
      - Requires regularisation via max_depth / min_samples_* to avoid
        memorising the training set.
      - Typically outperformed by ensemble methods (RandomForest, GBT).

    Config keys (under cfg.model.*):
        max_depth          int | null   5        None = fully grown
        min_samples_split  int          10       min samples to split a node
        min_samples_leaf   int          5        min samples in a leaf
        criterion          str          "squared_error"
                                        options: squared_error | friedman_mse
                                                 absolute_error | poisson
        random_state       int          42
    """

    _category_label = "decision_tree"

    def _build_estimator(self, cfg: DotDict) -> Any:
        from sklearn.tree import DecisionTreeRegressor as _SKLearnDT  # type: ignore[import]

        model_cfg = cfg.get("model") or DotDict({})

        # Read hyperparameters from config — no hardcoded values
        max_depth         = model_cfg.get("max_depth",         5)
        min_samples_split = model_cfg.get("min_samples_split", 10)
        min_samples_leaf  = model_cfg.get("min_samples_leaf",  5)
        criterion         = model_cfg.get("criterion",         "squared_error")
        random_state      = model_cfg.get("random_state",      42)

        log.info(
            "DecisionTree config — max_depth=%s | min_samples_split=%d | "
            "min_samples_leaf=%d | criterion=%s",
            max_depth, min_samples_split, min_samples_leaf, criterion,
        )

        return _SKLearnDT(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=random_state,
        )

    def export_rules(self, feature_names: list[str] | None = None, max_depth: int = 3) -> str:
        """Export the tree structure as a human-readable text rule list.

        Useful for clinical reporting: each leaf represents a patient subgroup
        with a predicted survival value.

        Args:
            feature_names: Column names from the tabular CSV.
            max_depth:     Maximum depth to print (deeper trees become unreadable).

        Returns:
            Multi-line string of decision rules, or an error message if the
            tree has not been fitted yet.
        """
        if not self._fitted:
            return "Tree not yet fitted — call fit() first."

        try:
            from sklearn.tree import export_text  # type: ignore[import]
            return export_text(
                self._est,
                feature_names=feature_names,
                max_depth=max_depth,
            )
        except Exception as exc:
            return f"Could not export rules: {exc}"


# ---------------------------------------------------------------------------
# 2. Random Forest — stronger baseline, minimal config change
# ---------------------------------------------------------------------------

class RandomForestRegressor(_TabularBase):
    """Bagged ensemble of decision trees.

    Reduces the variance of a single tree by averaging over many trees trained
    on bootstrap samples. The out-of-bag (OOB) error is a free validation
    estimate without needing a held-out split.

    Config keys (under cfg.model.*):
        n_estimators       int   200
        max_depth          int | null   null   (None = fully grown per tree)
        min_samples_split  int   5
        min_samples_leaf   int   2
        max_features       str | float  "sqrt"   features per split
        oob_score          bool  true
        n_jobs             int   -1    (-1 = all CPU cores)
        random_state       int   42
    """

    _category_label = "random_forest"

    def _build_estimator(self, cfg: DotDict) -> Any:
        from sklearn.ensemble import RandomForestRegressor as _SKLearnRF  # type: ignore[import]

        model_cfg = cfg.get("model") or DotDict({})

        n_estimators      = model_cfg.get("n_estimators",      200)
        max_depth         = model_cfg.get("max_depth",         None)
        min_samples_split = model_cfg.get("min_samples_split", 5)
        min_samples_leaf  = model_cfg.get("min_samples_leaf",  2)
        max_features      = model_cfg.get("max_features",      "sqrt")
        oob_score         = model_cfg.get("oob_score",         True)
        n_jobs            = model_cfg.get("n_jobs",            -1)
        random_state      = model_cfg.get("random_state",      42)

        log.info(
            "RandomForest config — n_estimators=%d | max_depth=%s | "
            "max_features=%s | oob_score=%s",
            n_estimators, max_depth, max_features, oob_score,
        )

        return _SKLearnRF(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressor":
        super().fit(X, y)
        if getattr(self._est, "oob_score_", None) is not None:
            log.info("RandomForest OOB R² = %.4f", self._est.oob_score_)
        return self


# ---------------------------------------------------------------------------
# 3. Gradient Boosted Trees — strong sklearn baseline
# ---------------------------------------------------------------------------

class GradientBoostedTrees(_TabularBase):
    """sklearn HistGradientBoostingRegressor — fast, handles NaN natively.

    Uses histogram-based gradient boosting (similar to LightGBM internals).
    Key advantage over the vanilla GradientBoostingRegressor: it supports
    missing values directly without imputation, which is useful for sparse
    genomics data.

    Config keys (under cfg.model.*):
        max_iter           int    300    number of boosting rounds
        max_depth          int    4
        learning_rate      float  0.05
        min_samples_leaf   int    20
        l2_regularization  float  0.1
        early_stopping     bool   true   stop if val score doesn't improve
        n_iter_no_change   int    20
        random_state       int    42
    """

    _category_label = "gradient_boosted"

    def _build_estimator(self, cfg: DotDict) -> Any:
        from sklearn.ensemble import HistGradientBoostingRegressor as _HGBR  # type: ignore[import]

        model_cfg = cfg.get("model") or DotDict({})

        max_iter          = model_cfg.get("max_iter",          300)
        max_depth         = model_cfg.get("max_depth",         4)
        learning_rate     = model_cfg.get("learning_rate",     0.05)
        min_samples_leaf  = model_cfg.get("min_samples_leaf",  20)
        l2_reg            = model_cfg.get("l2_regularization", 0.1)
        early_stopping    = model_cfg.get("early_stopping",    True)
        n_iter_no_change  = model_cfg.get("n_iter_no_change",  20)
        random_state      = model_cfg.get("random_state",      42)

        log.info(
            "GradientBoostedTrees config — max_iter=%d | max_depth=%s | "
            "lr=%g | early_stopping=%s",
            max_iter, max_depth, learning_rate, early_stopping,
        )

        return _HGBR(
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_reg,
            early_stopping=early_stopping,
            n_iter_no_change=n_iter_no_change,
            random_state=random_state,
        )

class LassoRegressor(_TabularBase):
    _category_label = "LassoRegression"

    def _build_estimator(self, cfg: DotDict) -> Any:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PowerTransformer, RobustScaler
        from sklearn.linear_model import Lasso

        model_cfg = cfg.get("model") or DotDict({})

        alpha        = model_cfg.get("alpha",        1.0)
        max_iter     = model_cfg.get("max_iter",     1000)
        selection    = model_cfg.get("selection",    "cyclic")
        tol          = model_cfg.get("tol",          1e-4)
        random_state = model_cfg.get("random_state", 42)

        log.info(
            "LassoRegressor config — alpha=%s | max_iter=%s | selection=%s | tol=%s | random_state=%s",
            alpha,
            max_iter,
            selection,
            tol,
            random_state,
        )

        return Pipeline([
            ("power", PowerTransformer(standardize=False)),  # e.g. Yeo-Johnson by default
            ("scaler", RobustScaler()),
            ("lasso", Lasso(
                alpha=alpha,
                max_iter=max_iter,
                selection=selection,
                tol=tol,
                random_state=random_state,
            ))
        ])

