"""Model-cascade machinery with statistical guarantees.

Routes each item in a batch to a cheap *proxy* model or an expensive *oracle*
model, while guaranteeing a target accuracy / precision / recall holds with
probability ``1 - delta`` for any finite sample size.

This is a dependency-free reimplementation of the procedures from BARGAIN
(UC Berkeley EPIC lab, https://github.com/ucbepic/BARGAIN). The statistical
core -- betting confidence sequences, without-replacement adaptive sampling,
and the threshold searches for the accuracy / precision / recall guarantees --
is ported from that work. It is adapted here to operate over pluggable
proxy / oracle *callables* so DocETL operators (filter, map-with-enum,
resolve, equijoin) can share a single guarantee-bearing engine.

The engine is intentionally free of any DocETL imports so it can be unit
tested against synthetic proxy/oracle functions without making LLM calls.

Usage
-----
    spec = CascadeSpec(proxy_model="gpt-4o-mini", guarantee="recall",
                       target=0.95, delta=0.05, label_budget=300)
    cascade = CategoricalCascade(spec, proxy_predict, oracle_predict)
    result = cascade.run(items)
    result.labels          # final label per item (oracle where escalated)
    result.escalated       # bool per item: was the oracle used?
    result.stats           # CascadeStats (proxy/oracle calls, threshold, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Optional

import numpy as np

Guarantee = str  # "accuracy" | "precision" | "recall"

# Callable supplied by an operator. ``proxy_predict`` returns the proxy's
# label plus its confidence in that label (probability in [0, 1], e.g. the
# softmax probability of the decoded answer token). ``oracle_predict`` returns
# the trusted label.
ProxyPredict = Callable[[Any], "tuple[Hashable, float]"]
OraclePredict = Callable[[Any], Hashable]


# ---------------------------------------------------------------------------
# Betting confidence sequences (ported verbatim from BARGAIN/bounds).
#
# These test, with a without-replacement betting martingale, whether the true
# mean of a {0,1} sequence is above / below a target ``m`` at confidence level
# ``alpha``. They underpin every guarantee below; the math is kept identical to
# the reference implementation.
# ---------------------------------------------------------------------------
def _sigma_squared(i: int, obs: np.ndarray) -> float:
    mu_hat = (1 / 2 + obs[:i].sum()) / (i + 1)
    return (1 / 4 + ((obs[:i] - mu_hat) ** 2).sum()) / (i + 1)


def _get_lambda(alpha: float, obs: np.ndarray, i: int, fixed_sample_size: bool) -> float:
    if fixed_sample_size:
        return np.sqrt(2 * np.log(2 / alpha) / (len(obs) * _sigma_squared(i - 1, obs)))
    return np.sqrt(
        2 * np.log(2 / alpha) / (i * np.log(i + 1) * _sigma_squared(i - 1, obs))
    )


def _k_plus(
    obs, target, alpha, trunc_scale, theta,
    without_replacement=False, N=0, fixed_sample_size=True,
):
    if without_replacement:
        assert N > 0
        t = np.arange(1, len(obs) + 1)
        S_t = np.cumsum(obs)
        S_tminus1 = np.append(0, S_t[0 : (len(obs) - 1)])
        m_wor_i = (N * target - S_tminus1) / (N - (t - 1))
    else:
        m_wor_i = np.repeat(target, len(obs))

    k_t = 1
    for i in range(1, len(obs) + 1):
        lam = _get_lambda(alpha, obs, i, fixed_sample_size)
        lam = np.minimum(lam, trunc_scale / m_wor_i[i - 1])
        k_t *= 1 + lam * (obs[i - 1] - m_wor_i[i - 1])
        if theta * k_t >= 1 / alpha:
            return False
    return True


def _k_minus(
    obs, target, alpha, trunc_scale, theta,
    without_replacement=False, N=0, fixed_sample_size=True,
):
    if without_replacement:
        assert N > 0
        m_wor_i = np.concatenate(
            [
                np.array([target]),
                (N * target - np.cumsum(obs[:-1])) / (N - (np.arange(len(obs) - 1))),
            ]
        )
    else:
        m_wor_i = np.repeat(target, len(obs))

    k_t = 1
    for i in range(1, len(obs) + 1):
        lam = _get_lambda(alpha, obs, i, fixed_sample_size)
        lam = np.minimum(lam, trunc_scale / (1 - m_wor_i[-1]))
        k_t *= 1 - lam * (obs[i - 1] - m_wor_i[i - 1])
        if (1 - theta) * k_t >= 1 / alpha:
            return False
    return True


def _mean_is_in_conf(
    obs, target, alpha, theta, trunc_scale,
    fixed_sample_size=True, without_replacement=False, N=0,
):
    kt_plus = _k_plus(
        obs, target, alpha, trunc_scale, theta, without_replacement, N, fixed_sample_size
    )
    if not kt_plus or theta == 1:
        return kt_plus
    return _k_minus(
        obs, target, alpha, trunc_scale, theta, without_replacement, N, fixed_sample_size
    )


def _seq_test_above(
    obs, m, alpha, fixed_sample_size=True, without_replacement=False, N=0
):
    """Sequential betting test for ``mean(obs) > m``.

    Returns ``True`` while the (1-alpha) confidence sequence is still
    *undecided* (the caller should keep sampling), and ``False`` once the
    betting supermartingale crosses the ``1/alpha`` boundary -- i.e. once we
    are (1-alpha)-confident the true mean exceeds ``m``. This inverted-looking
    contract matches how it is consumed inside the adaptive sampling loop.
    """
    assert not without_replacement or (without_replacement and N > 0)
    return _mean_is_in_conf(
        obs, m, alpha, 1, 3 / 4, fixed_sample_size, without_replacement, N
    )


def _seq_test_below(
    obs, m, alpha, fixed_sample_size=True, without_replacement=False, N=0
):
    """Sequential betting test for ``mean(obs) < m``.

    Returns ``True`` while undecided (keep sampling), ``False`` once we are
    (1-alpha)-confident the true mean is below ``m``. See :func:`_seq_test_above`.
    """
    assert not without_replacement or (without_replacement and N > 0)
    return _mean_is_in_conf(
        obs, m, alpha, 0, 3 / 4, fixed_sample_size, without_replacement, N
    )


# ---------------------------------------------------------------------------
# Without-replacement sampler (ported from BARGAIN/sampler).
#
# Draws items (by sorted position) below a threshold without replacement,
# remembering what was already drawn so repeated calls keep extending the
# sample rather than re-drawing.
# ---------------------------------------------------------------------------
class WoRSampler:
    def __init__(self, n: int):
        self.indxs = np.random.permutation(n)
        self.sampled_at_threshs: dict = {}
        self.all_sampled = np.array([])

    def sample(self, thresh: int, k: int):
        if thresh not in self.sampled_at_threshs:
            self.sampled_at_threshs[thresh] = 0

        t = self.sampled_at_threshs[thresh]
        curr_indxs = self.indxs[self.indxs <= thresh]
        if t >= len(curr_indxs):
            return np.array([], dtype=int), 0, True
        sample = curr_indxs[t : t + k]
        t += len(sample)
        sampled_all = t >= len(curr_indxs)
        self.sampled_at_threshs[thresh] = t
        prv_sampled = len(self.all_sampled)
        self.all_sampled = np.union1d(sample, self.all_sampled)
        budget_used = len(self.all_sampled) - prv_sampled
        return sample, budget_used, sampled_all


# ---------------------------------------------------------------------------
# Pluggable proxy / oracle wrappers with per-item caching, mirroring the
# accessor shape the ported procedures expect.
# ---------------------------------------------------------------------------
class _Proxy:
    def __init__(self, predict_fn: ProxyPredict):
        self._predict_fn = predict_fn
        self.preds: dict[int, tuple[Hashable, float]] = {}

    def reset(self):
        self.preds = {}

    def get_preds_and_scores(self, idxs, records):
        labels, scores = [], []
        for i, rec in zip(idxs, records):
            i = int(i)
            if i not in self.preds:
                self.preds[i] = self._predict_fn(rec)
            label, score = self.preds[i]
            labels.append(label)
            scores.append(float(score))
        return np.array(labels), np.array(scores, dtype=float)

    def n_calls(self) -> int:
        return len(self.preds)


class _Oracle:
    """Wraps the oracle callable. When ``positive_label`` is set (precision /
    recall modes) ``get_pred`` returns binary {0,1} labels; otherwise it
    returns the raw oracle label (accuracy mode)."""

    def __init__(self, predict_fn: OraclePredict, positive_label: Optional[Hashable] = None):
        self._predict_fn = predict_fn
        self._positive_label = positive_label
        self.preds: dict[int, Hashable] = {}

    def reset(self):
        self.preds = {}

    def _raw(self, idx: int, record):
        idx = int(idx)
        if idx not in self.preds:
            self.preds[idx] = self._predict_fn(record)
        return self.preds[idx]

    def get_pred(self, records, idxs):
        out = []
        for rec, i in zip(records, idxs):
            label = self._raw(i, rec)
            if self._positive_label is not None:
                out.append(1 if label == self._positive_label else 0)
            else:
                out.append(label)
        return np.array(out)

    def is_answer_correct(self, idxs, records, proxy_preds):
        out = []
        for i, rec, p in zip(idxs, records, proxy_preds):
            out.append(1.0 if self._raw(i, rec) == p else 0.0)
        return np.array(out, dtype=float)

    def get_number_preds(self) -> int:
        return len(self.preds)


# ---------------------------------------------------------------------------
# Public spec / result types.
# ---------------------------------------------------------------------------
@dataclass
class CascadeSpec:
    proxy_model: str
    guarantee: Guarantee  # "accuracy" | "precision" | "recall"
    target: float
    delta: float = 0.05
    label_budget: int = 400  # oracle calls for threshold learning (precision/recall)
    positive_label: Hashable = True  # which label counts as "positive" (P/R)
    negative_label: Hashable = False  # label assigned to non-positive items (P/R)
    n_thresholds: int = 20  # candidate thresholds considered (accuracy/precision)
    seed: Optional[int] = 0


@dataclass
class CascadeStats:
    n_items: int
    proxy_calls: int
    oracle_calls: int
    escalation_rate: float
    guarantee: Guarantee
    target: float
    delta: float


@dataclass
class CascadeResult:
    labels: list  # final label per item, in input order
    escalated: list  # bool per item: was the oracle used?
    stats: CascadeStats
    # For precision / recall guarantees, the set of input indices the engine
    # estimates as positive (the records to keep / the matching pairs).
    positive_indices: list = field(default_factory=list)


class GuaranteeNotSupportedError(ValueError):
    pass


class CategoricalCascade:
    """Guarantee-bearing proxy/oracle cascade over a list of items whose LLM
    output is a single categorical label.

    See module docstring for the proxy/oracle callable contract.
    """

    def __init__(
        self,
        spec: CascadeSpec,
        proxy_predict: ProxyPredict,
        oracle_predict: OraclePredict,
    ):
        if spec.guarantee not in ("accuracy", "precision", "recall"):
            raise GuaranteeNotSupportedError(
                f"unknown guarantee {spec.guarantee!r}; "
                "expected 'accuracy', 'precision' or 'recall'"
            )
        if not (0 < spec.target < 1):
            raise ValueError("target must be in (0, 1)")
        if not (0 < spec.delta < 1):
            raise ValueError("delta must be in (0, 1)")
        self.spec = spec
        self.proxy = _Proxy(proxy_predict)
        positive_label = (
            spec.positive_label if spec.guarantee in ("precision", "recall") else None
        )
        self.oracle = _Oracle(oracle_predict, positive_label=positive_label)

    def run(self, items: list) -> CascadeResult:
        if self.spec.seed is not None:
            np.random.seed(self.spec.seed)
        self.proxy.reset()
        self.oracle.reset()
        if len(items) == 0:
            return CascadeResult(
                labels=[],
                escalated=[],
                stats=CascadeStats(0, 0, 0, 0.0, self.spec.guarantee, self.spec.target, self.spec.delta),
                positive_indices=[],
            )
        if self.spec.guarantee == "accuracy":
            return self._run_accuracy(items)
        if self.spec.guarantee == "precision":
            return self._run_positive_set(items, mode="precision")
        return self._run_positive_set(items, mode="recall")

    def _stats(self, n_items: int) -> CascadeStats:
        oracle_calls = self.oracle.get_number_preds()
        return CascadeStats(
            n_items=n_items,
            proxy_calls=self.proxy.n_calls(),
            oracle_calls=oracle_calls,
            escalation_rate=oracle_calls / n_items if n_items else 0.0,
            guarantee=self.spec.guarantee,
            target=self.spec.target,
            delta=self.spec.delta,
        )

    # ------------------------------------------------------------------
    # Accuracy guarantee (ported from BARGAIN_A): output matches the oracle
    # on >= target fraction of items, w.p. >= 1 - delta.
    # ------------------------------------------------------------------
    def _check_worth_trying(self, sample_indx, sample_is_correct, t, target):
        if len(sample_indx) < 50:
            return True
        mask_at_t = sample_indx <= t
        samples_at_thresh = sample_is_correct[mask_at_t]
        if np.mean(samples_at_thresh) - np.std(samples_at_thresh) < target:
            return False
        return True

    def _sample_till_confident_above_target(
        self, all_data_indexes, all_preds, confidence, target, total_sampled, curr_thresh, data_records
    ):
        sample_step = 10
        sampled_is_correct = np.array([])
        sampled_index = np.array([]).astype(int)

        while self._check_worth_trying(sampled_index, sampled_is_correct, curr_thresh, target):
            sampled_indexes, budget_used, sampled_all = self.sampler.sample(curr_thresh, sample_step)

            sampled_data_indexes = all_data_indexes[sampled_indexes]
            proxy_preds = all_preds[sampled_indexes]
            sampled_is_correct = np.concatenate(
                [
                    sampled_is_correct,
                    self.oracle.is_answer_correct(
                        sampled_data_indexes, data_records[sampled_indexes], proxy_preds
                    ),
                ]
            )
            sampled_index = np.concatenate([sampled_index, sampled_indexes])
            total_sampled += budget_used

            if sampled_all:
                return not np.mean(sampled_is_correct) < target, sampled_index, total_sampled

            samples_at_thresh = sampled_is_correct[sampled_index <= curr_thresh]
            N = curr_thresh + 1
            if np.mean(samples_at_thresh) < target:
                conf_has_target = _seq_test_below(
                    samples_at_thresh, target, alpha=confidence, without_replacement=True, N=N, fixed_sample_size=False
                )
                is_below_target = True
            else:
                conf_has_target = _seq_test_above(
                    samples_at_thresh, target, alpha=confidence, without_replacement=True, N=N, fixed_sample_size=False
                )
                is_below_target = False
            if not conf_has_target:
                return not is_below_target, sampled_index, total_sampled

        return False, sampled_index, total_sampled

    def _run_accuracy(self, items: list) -> CascadeResult:
        data_records = np.array(items, dtype=object)
        data_idxs = np.arange(len(items))
        self.sampler = WoRSampler(len(data_idxs))
        thresh_step = max(len(data_idxs) // self.spec.n_thresholds, 1)

        proxy_preds, proxy_scores = self.proxy.get_preds_and_scores(data_idxs, data_records[data_idxs])

        sort_indx = np.argsort(proxy_scores)[::-1]
        proxy_preds = proxy_preds[sort_indx]
        proxy_scores = proxy_scores[sort_indx]
        data_idxs = data_idxs[sort_indx]
        data_records = data_records[sort_indx]

        sample_indexes: Any = []
        total_sampled = 0
        best_thresh = 0
        for curr_thresh in range(thresh_step - 1, len(data_idxs), thresh_step):
            if curr_thresh == len(data_idxs) - 1:
                new_target = self.spec.target
            else:
                n_from_proxy = curr_thresh + 1
                n_from_oracle = len(data_idxs) - n_from_proxy
                new_target = (
                    self.spec.target * (n_from_oracle + n_from_proxy) - n_from_oracle
                ) / n_from_proxy
                if new_target <= 0:
                    continue

            is_confident, sampled_index, total_sampled = self._sample_till_confident_above_target(
                data_idxs, proxy_preds, self.spec.delta, new_target, total_sampled, curr_thresh, data_records
            )
            sample_indexes = np.concatenate([sample_indexes, sampled_index])
            if not is_confident:
                break
            best_thresh = curr_thresh

        proxy_indxs = np.setdiff1d(
            data_idxs[:best_thresh], data_idxs[np.array(sample_indexes).astype(int)]
        )
        oracle_indexes = np.setdiff1d(data_idxs, proxy_indxs)

        id_to_pos = {int(id_): i for i, id_ in enumerate(data_idxs)}
        oracle_pos = [id_to_pos[int(oid)] for oid in oracle_indexes]
        oracle_outputs = self.oracle.get_pred(data_records[oracle_pos], oracle_indexes)

        proxy_out_preds, _ = self.proxy.get_preds_and_scores(proxy_indxs, data_records[[id_to_pos[int(p)] for p in proxy_indxs]])

        indexes_data_indx = np.concatenate([oracle_indexes, proxy_indxs])
        output = np.concatenate([oracle_outputs, proxy_out_preds])
        used_oracle = np.array([True] * len(oracle_indexes) + [False] * len(proxy_indxs))

        order = np.argsort(indexes_data_indx)
        output = output[order]
        used_oracle = used_oracle[order]

        return CascadeResult(
            labels=output.tolist(),
            escalated=used_oracle.tolist(),
            stats=self._stats(len(items)),
        )

    # ------------------------------------------------------------------
    # Precision / recall guarantees: return the set of items estimated
    # positive. Precision ported from BARGAIN_P; recall from BARGAIN_R
    # (beta=0 uniform path, the documented guarantee-preserving setting).
    # ------------------------------------------------------------------
    def _run_positive_set(self, items: list, mode: str) -> CascadeResult:
        data_records = np.array(items, dtype=object)
        data_idxs = np.arange(len(items))
        labels, scores = self.proxy.get_preds_and_scores(data_idxs, data_records)
        bin_preds = (labels == self.spec.positive_label).astype(int)
        # estimated probability that the true label is positive
        x_probs = bin_preds * scores + (1 - bin_preds) * (1 - scores)

        if mode == "precision":
            positive_idxs = self._precision_positive_set(data_idxs, data_records, x_probs)
        else:
            positive_idxs = self._recall_positive_set(data_idxs, data_records, x_probs)

        positive_set = set(int(i) for i in positive_idxs)
        result_labels = [
            self.spec.positive_label if i in positive_set else self.spec.negative_label
            for i in range(len(items))
        ]
        escalated = [i in self.oracle.preds for i in range(len(items))]
        return CascadeResult(
            labels=result_labels,
            escalated=escalated,
            stats=self._stats(len(items)),
            positive_indices=sorted(positive_set),
        )

    def _sample_till_confident(
        self, budget, all_data_indexes, confidence, target, total_sampled, curr_thresh, data_records
    ):
        sample_step = 10
        sampled_label = np.array([])
        sampled_index = np.array([])

        while budget - total_sampled > 0:
            no_sample = min(sample_step, budget - total_sampled)
            sampled_indexes, budget_used, sampled_all = self.sampler.sample(curr_thresh, no_sample)
            sampled_data_indexes = all_data_indexes[sampled_indexes]

            sampled_label = np.concatenate(
                [sampled_label, self.oracle.get_pred(data_records[sampled_indexes], sampled_data_indexes)]
            )
            sampled_index = np.concatenate([sampled_index, sampled_indexes])
            total_sampled += budget_used

            if sampled_all:
                return not np.mean(sampled_label) < target, sampled_index, sampled_label, total_sampled

            N = curr_thresh + 1
            if np.mean(sampled_label) < target:
                conf_has_target = _seq_test_below(
                    np.array(sampled_label), target, alpha=confidence, without_replacement=True, N=N, fixed_sample_size=False
                )
                is_below_target = True
            else:
                conf_has_target = _seq_test_above(
                    np.array(sampled_label), target, alpha=confidence, without_replacement=True, N=N, fixed_sample_size=False
                )
                is_below_target = False
            if not conf_has_target:
                return not is_below_target, sampled_index, sampled_label, total_sampled

        return False, sampled_index, sampled_label, total_sampled

    def _precision_positive_set(self, data_idxs, data_records, x_probs):
        eta = 0 + 1
        self.sampler = WoRSampler(len(data_idxs))
        thresh_step = max(len(data_idxs) // self.spec.n_thresholds, 1)

        sort_indx = np.argsort(x_probs)[::-1]
        data_idxs = data_idxs[sort_indx]
        data_records = data_records[sort_indx]

        budget = self.spec.label_budget
        sample_indexes: Any = []
        total_sampled = 0
        best_thresh = 0
        tries_used = 0
        for curr_thresh in range(thresh_step - 1, len(data_idxs), thresh_step):
            is_confident, sampled_index, _, total_sampled = self._sample_till_confident(
                budget, data_idxs, self.spec.delta / eta, self.spec.target, total_sampled, curr_thresh, data_records
            )
            sample_indexes = np.concatenate([sample_indexes, sampled_index])
            if budget == total_sampled:
                break
            if not is_confident:
                tries_used += 1
                if tries_used >= eta:
                    break
            else:
                best_thresh = curr_thresh

        if budget - total_sampled > 0 and best_thresh < len(data_idxs) - 1:
            more_samples = []
            curr_to_label = best_thresh + 1
            while budget - total_sampled > 0 and curr_to_label < len(data_idxs):
                more_samples.append(curr_to_label)
                if curr_to_label not in sample_indexes:
                    total_sampled += 1
                curr_to_label += 1
            sample_indexes = np.concatenate([sample_indexes, np.array(more_samples)])

        set_ids = data_idxs[:best_thresh]
        sample_indexes = np.unique(np.array(sample_indexes).astype(int))
        all_sample_idxs = data_idxs[sample_indexes]
        all_sample_records = data_records[sample_indexes]
        all_sample_labels = self.oracle.get_pred(all_sample_records, all_sample_idxs)
        samp_inds = data_idxs[sample_indexes[all_sample_labels == 1]]
        return np.unique(np.concatenate([set_ids, samp_inds]))

    def _recall_positive_set(self, data_idxs, data_records, x_probs):
        # beta=0 uniform path from BARGAIN_R.
        sort_indx = np.argsort(x_probs)
        data_idxs = data_idxs[sort_indx]
        data_records = data_records[sort_indx]

        sampled_indexes = np.random.choice(len(data_idxs), self.spec.label_budget, replace=True)
        sample_data_indx = data_idxs[sampled_indexes]
        sampled_label = self.oracle.get_pred(data_records[sampled_indexes], sample_data_indx)
        pos_sampled_indexes = sampled_indexes[sampled_label == 1]
        threshs_to_try = np.sort(pos_sampled_indexes)

        best_t = None
        for t in threshs_to_try:
            samples_at_thresh = pos_sampled_indexes >= t
            if np.mean(samples_at_thresh) < self.spec.target:
                conf_has_target = True
            else:
                conf_has_target = _seq_test_above(
                    np.array(samples_at_thresh), self.spec.target, alpha=self.spec.delta
                )
            if conf_has_target:
                break
            else:
                best_t = t
        if best_t is None:
            best_t = 0

        indexes_assumed_positive = np.arange(best_t, len(data_idxs)).astype(int)
        set_ids = data_idxs[indexes_assumed_positive]
        samp_inds = data_idxs[sampled_indexes[sampled_label == 1]]
        return np.unique(np.concatenate([set_ids, samp_inds]))
