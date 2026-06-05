"""Statistical correctness tests for the model-cascade engine.

These tests use synthetic, perfectly-calibrated proxy/oracle functions (no LLM
calls) to verify that the accuracy / precision / recall guarantees hold with
the claimed coverage 1 - delta across many trials.
"""

import numpy as np
import pytest

from docetl.operations.utils.cascade import (
    CascadeSpec,
    CategoricalCascade,
    GuaranteeNotSupportedError,
    _seq_test_above,
    _seq_test_below,
)


# ---------------------------------------------------------------------------
# Sequential betting-test sanity checks.
#
# Contract: returns True while *undecided* (keep sampling); returns False once
# the betting supermartingale crosses the boundary (decision reached).
# ---------------------------------------------------------------------------
def test_seq_test_above_decides_on_strong_evidence():
    # All ones, low target -> decisively above -> decision reached (False).
    obs = np.ones(80)
    assert _seq_test_above(obs, 0.5, alpha=0.05) is False


def test_seq_test_below_decides_on_strong_evidence():
    # All zeros, mid target -> decisively below -> decision reached (False).
    obs = np.zeros(80)
    assert _seq_test_below(obs, 0.5, alpha=0.05) is False


def test_seq_test_stays_undecided_when_ambiguous():
    # A 50/50 sequence gives no evidence the mean exceeds 0.9 -> undecided (True).
    rng = np.random.default_rng(0)
    obs = (rng.random(60) < 0.5).astype(float)
    assert _seq_test_above(obs, 0.9, alpha=0.05) is True


# ---------------------------------------------------------------------------
# Helpers to build synthetic, calibrated proxy/oracle functions
# ---------------------------------------------------------------------------
def _make_accuracy_world(n, rng):
    """Binary oracle truth; a calibrated proxy whose probability of being
    correct equals its reported confidence."""
    truth = (rng.random(n) < 0.5).astype(int)
    conf = rng.uniform(0.5, 1.0, size=n)
    correct = rng.random(n) < conf
    proxy_label = np.where(correct, truth, 1 - truth)

    def proxy_predict(item):
        i = int(item)
        return int(proxy_label[i]), float(conf[i])

    def oracle_predict(item):
        return int(truth[int(item)])

    return truth, proxy_predict, oracle_predict


def _make_binary_world(n, rng):
    """Calibrated scores: P(positive | score=s) = s, so a proxy reporting
    confidence in its own label is well calibrated."""
    score = rng.uniform(0.0, 1.0, size=n)  # P(positive)
    truth = (rng.random(n) < score).astype(int)
    proxy_label = (score > 0.5).astype(int)
    conf = np.maximum(score, 1 - score)  # confidence in the predicted label

    def proxy_predict(item):
        i = int(item)
        return bool(proxy_label[i]), float(conf[i])

    def oracle_predict(item):
        return bool(truth[int(item)])

    return truth, proxy_predict, oracle_predict


# ---------------------------------------------------------------------------
# Guarantee coverage tests
# ---------------------------------------------------------------------------
def test_accuracy_guarantee_coverage():
    target, delta, n, trials = 0.85, 0.1, 800, 50
    successes = 0
    for trial in range(trials):
        rng = np.random.default_rng(trial)
        truth, proxy_predict, oracle_predict = _make_accuracy_world(n, rng)
        spec = CascadeSpec(
            proxy_model="proxy", guarantee="accuracy",
            target=target, delta=delta, seed=10_000 + trial,
        )
        result = CategoricalCascade(spec, proxy_predict, oracle_predict).run(list(range(n)))
        labels = np.array(result.labels)
        accuracy = float(np.mean(labels == truth))
        if accuracy >= target:
            successes += 1
    coverage = successes / trials
    assert coverage >= 1 - delta, f"accuracy coverage {coverage} < {1 - delta}"


def test_precision_guarantee_coverage():
    target, delta, n, trials = 0.9, 0.1, 600, 40
    successes = 0
    for trial in range(trials):
        rng = np.random.default_rng(1000 + trial)
        truth, proxy_predict, oracle_predict = _make_binary_world(n, rng)
        spec = CascadeSpec(
            proxy_model="proxy", guarantee="precision",
            target=target, delta=delta, label_budget=300, seed=20_000 + trial,
        )
        result = CategoricalCascade(spec, proxy_predict, oracle_predict).run(list(range(n)))
        pos = result.positive_indices
        if not pos:
            successes += 1  # empty positive set trivially satisfies precision
            continue
        precision = float(np.mean(truth[pos] == 1))
        if precision >= target:
            successes += 1
    coverage = successes / trials
    assert coverage >= 1 - delta, f"precision coverage {coverage} < {1 - delta}"


def test_recall_guarantee_coverage():
    target, delta, n, trials = 0.9, 0.1, 600, 40
    successes = 0
    for trial in range(trials):
        rng = np.random.default_rng(2000 + trial)
        truth, proxy_predict, oracle_predict = _make_binary_world(n, rng)
        n_positive = int(truth.sum())
        if n_positive == 0:
            continue
        spec = CascadeSpec(
            proxy_model="proxy", guarantee="recall",
            target=target, delta=delta, label_budget=300, seed=30_000 + trial,
        )
        result = CategoricalCascade(spec, proxy_predict, oracle_predict).run(list(range(n)))
        pos = set(result.positive_indices)
        found = sum(1 for i in range(n) if truth[i] == 1 and i in pos)
        recall = found / n_positive
        if recall >= target:
            successes += 1
    coverage = successes / trials
    assert coverage >= 1 - delta, f"recall coverage {coverage} < {1 - delta}"


# ---------------------------------------------------------------------------
# Behavioural sanity checks
# ---------------------------------------------------------------------------
def test_perfect_proxy_escalates_little():
    """A proxy that always matches the oracle (max confidence) should let the
    cascade trust most items, escalating only the calibration sample."""
    n = 500
    truth = np.arange(n) % 3  # 3-way labels

    def proxy_predict(item):
        return int(truth[int(item)]), 1.0

    def oracle_predict(item):
        return int(truth[int(item)])

    spec = CascadeSpec(proxy_model="proxy", guarantee="accuracy", target=0.9, delta=0.1, seed=0)
    result = CategoricalCascade(spec, proxy_predict, oracle_predict).run(list(range(n)))
    assert np.array_equal(np.array(result.labels), truth)
    assert result.stats.escalation_rate < 0.5


def test_output_matches_oracle_where_escalated():
    n = 300
    rng = np.random.default_rng(7)
    truth, proxy_predict, oracle_predict = _make_accuracy_world(n, rng)
    spec = CascadeSpec(proxy_model="proxy", guarantee="accuracy", target=0.95, delta=0.1, seed=3)
    result = CategoricalCascade(spec, proxy_predict, oracle_predict).run(list(range(n)))
    for i, escalated in enumerate(result.escalated):
        if escalated:
            assert result.labels[i] == truth[i]


def test_empty_input():
    spec = CascadeSpec(proxy_model="proxy", guarantee="recall", target=0.9, delta=0.1)
    result = CategoricalCascade(spec, lambda x: (True, 1.0), lambda x: True).run([])
    assert result.labels == []
    assert result.escalated == []
    assert result.stats.n_items == 0


def test_rejects_unknown_guarantee():
    with pytest.raises(GuaranteeNotSupportedError):
        CascadeSpec_invalid = CascadeSpec(proxy_model="p", guarantee="f1", target=0.9, delta=0.1)
        CategoricalCascade(CascadeSpec_invalid, lambda x: (True, 1.0), lambda x: True)


def test_rejects_bad_target_delta():
    with pytest.raises(ValueError):
        CategoricalCascade(
            CascadeSpec(proxy_model="p", guarantee="recall", target=1.5, delta=0.1),
            lambda x: (True, 1.0), lambda x: True,
        )
    with pytest.raises(ValueError):
        CategoricalCascade(
            CascadeSpec(proxy_model="p", guarantee="recall", target=0.9, delta=0.0),
            lambda x: (True, 1.0), lambda x: True,
        )
