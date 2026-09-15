"""Suggestion engines for the label app.

All engines score cells in ONE shared embedding space W (whitened-PCA +
L2-normalized feature rows — the deduplication/vis convention). Three
complementary sources feed the UI badges and the queue rankings:

  SuggestEngine  per-label nearest-exemplar kNN score (positive similarity
                 minus explicit-negative similarity — PU: undecided is not
                 a negative)
  MLModelEngine  a per-label sklearn classifier fitted on that label's
                 explicit positives vs explicit negatives (active learning
                 once a label has enough of BOTH)
  review_items   leave-one-out consistency check over DECIDED cells —
                 flags decided cells whose embedding contradicts their own
                 decision (likely mislabels)

Score arrays are cached per label and invalidated selectively: writing to
label A must not throw away label B's cached scores on a 40k-cell project,
so bump() takes an optional label id (None = invalidate everything).
"""

import threading

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def _diverse_order(W, m, seed):
    """Greedy farthest-point ordering: m row indices spread over the space.

    Starting from a seeded random row, each step appends the row with the
    LOWEST maximum cosine similarity to the already chosen set — the classic
    coverage-maximizing batch for the cold start, where every labeled cell
    teaches the kNN something new.
    """
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    m = int(min(m, n))
    chosen = [int(rng.integers(n))]
    max_sim = W @ W[chosen[0]]
    for _ in range(m - 1):
        nxt = int(np.argmin(max_sim))
        chosen.append(nxt)
        max_sim = np.maximum(max_sim, W @ W[nxt])
    return chosen


class SuggestEngine:
    """Per-label nearest-exemplar scores in the L2-normalized space.

    For label L with positive rows P and explicit-negative rows G:
      pos_part(x) = mean of the top-k cosine similarities from x to P
      neg_part(x) = mean of the top-k cosine similarities from x to G
      score       = pos_part - neg_weight * neg_part
    k = min(knn_k, |P|); the default knn_k=1 makes this a pure
    nearest-exemplar score — with few, mutually-diverse positives a top-k
    MEAN would dilute the one truly similar exemplar (measured: top-4 mean
    ~0.01 while the nearest same-type neighbor sits at ~0.8), so the max is
    the right primary signal and larger k only smooths once many exemplars
    accumulate. A suggestion fires when pos_part >= threshold AND
    pos_part > neg_part (negatives act as a veto and a ranking term).

    Score arrays are cached per label keyed by the decision counts and
    invalidated — for ONE label or wholesale — by bump().
    """

    def __init__(self, W, knn_k, neg_weight):
        self.W = W
        self.n = W.shape[0]
        self.knn_k = int(knn_k)
        self.neg_weight = float(neg_weight)
        self._cache = {}       # label_id -> (version, (npos, nneg), parts)
        self.version = 0
        self._lock = threading.Lock()

    def bump(self, label_id=None):
        """Invalidate cached scores (one label, or all when label is None).

        Called after every annotation write with the affected label —
        other labels' decisions did not change, so their cached score
        arrays stay valid (a real win on tens-of-thousands-cell projects).
        """
        with self._lock:
            self.version += 1
            if label_id is None:
                self._cache.clear()
            else:
                self._cache.pop(label_id, None)

    @staticmethod
    def _topk_mean(sims, k):
        """Mean of the k largest similarities per row (vectorized)."""
        if sims.shape[1] > k:
            part = np.partition(sims, -k, axis=1)[:, -k:]
        else:
            part = sims
        return part.mean(axis=1)

    def parts(self, label_id, pos_rows, neg_rows):
        """(pos_part, neg_part) score arrays over ALL cells for one label."""
        with self._lock:
            hit = self._cache.get(label_id)
            if hit is not None and hit[0] == self.version and \
                    hit[1] == (len(pos_rows), len(neg_rows)):
                return hit[2]

        pos_part = np.full(self.n, -1.0, dtype=np.float32)
        neg_part = np.zeros(self.n, dtype=np.float32)
        if pos_rows:
            k = min(self.knn_k, len(pos_rows))
            sims = self.W @ self.W[pos_rows].T
            pos_part = self._topk_mean(sims, k)
        if neg_rows:
            k = min(self.knn_k, len(neg_rows))
            sims = self.W @ self.W[neg_rows].T
            neg_part = self._topk_mean(sims, k)
        with self._lock:
            self._cache[label_id] = (self.version,
                                     (len(pos_rows), len(neg_rows)),
                                     (pos_part, neg_part))
        return pos_part, neg_part


class MLModelEngine:
    """Per-label discriminative model over the embedding space.

    Where SuggestEngine ranks by similarity to exemplars, this engine fits a
    small sklearn classifier on a label's explicit positives (state 1) vs
    explicit negatives (state 0) — logistic regression by default, random
    forest as the non-linear alternative — and scores EVERY cell with
    predict_proba. A trained boundary separates lookalike-but-negative
    regions that pure nearest-exemplar similarity still scores high.

    A label only engages once it has at least min_pos positives AND min_neg
    negatives (below that a two-class fit would be noise — the kNN engine
    keeps serving those labels). Like SuggestEngine, score arrays are cached
    per label keyed by the decision counts and invalidated by bump(); the
    refit costs well under a second on the 50-d space, so models are never
    persisted to disk.
    """

    def __init__(self, W, kind, min_pos, min_neg, seed=42):
        self.W = W
        self.kind = kind              # "logistic" | "random_forest"
        self.min_pos = int(min_pos)
        self.min_neg = int(min_neg)
        self.seed = int(seed)
        self.version = 0
        self._cache = {}   # label_id -> (version, (npos, nneg), probs|None)
        self._lock = threading.Lock()

    def bump(self, label_id=None):
        """Invalidate cached probabilities (one label, or all when None)."""
        with self._lock:
            self.version += 1
            if label_id is None:
                self._cache.clear()
            else:
                self._cache.pop(label_id, None)

    def _fit(self, pos_rows, neg_rows):
        """Fit the configured classifier on explicit pos vs neg rows."""
        X = np.vstack([self.W[pos_rows], self.W[neg_rows]])
        y = np.r_[np.ones(len(pos_rows)), np.zeros(len(neg_rows))].astype(np.int64)
        if self.kind == "random_forest":
            # n_jobs=1: the server is a threaded Flask app; process pools
            # per request are not worth it on <= a few thousand rows.
            clf = RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2,
                class_weight="balanced", random_state=self.seed, n_jobs=1)
        else:
            clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                     random_state=self.seed)
        clf.fit(X, y)
        return clf

    def probs(self, label_id, pos_rows, neg_rows):
        """P(positive) over ALL cells, or None while below the gates."""
        with self._lock:
            hit = self._cache.get(label_id)
            if hit is not None and hit[0] == self.version and \
                    hit[1] == (len(pos_rows), len(neg_rows)):
                return hit[2]

        out = None
        if len(pos_rows) >= self.min_pos and len(neg_rows) >= self.min_neg:
            out = self._fit(pos_rows, neg_rows) \
                     .predict_proba(self.W)[:, 1].astype(np.float32)
        with self._lock:
            self._cache[label_id] = (self.version,
                                     (len(pos_rows), len(neg_rows)), out)
        return out


def review_items(W, row_cid, pos_rows, neg_rows):
    """Leave-one-out kNN consistency check over DECIDED cells of a label.

    The active-learning complement to the uncertainty ranking: instead of
    picking new informative cells, flag already-labeled cells whose
    embedding contradicts their own decision (likely mislabels — slips of
    the hand, or auto annotations accepted when the label had too few
    exemplars to be trustworthy). A positive is suspicious when its nearest
    OTHER positive (self excluded — the self-similarity of 1.0 would
    otherwise mask everything) is FARTHER than its nearest explicit
    negative; symmetrically for negatives (a suspicious negative is a
    likely MISSED positive). Evaluable requires >= 2 own-side and >= 1
    opposite-side exemplars — with fewer there is nothing to be
    inconsistent with.

    Returns dicts sorted by suspicion margin descending (only susp > 0):
      {cell_id, state, susp, own_sim, ev_sim, ev_cell_id}
    where ev_* points at the contradicting neighbor cell for the UI.
    """
    items = []
    if len(pos_rows) >= 2 and neg_rows:
        PP = W[pos_rows] @ W[pos_rows].T
        np.fill_diagonal(PP, -2.0)
        CM = W[pos_rows] @ W[neg_rows].T
        own, contra = PP.max(axis=1), CM.max(axis=1)
        for i, row in enumerate(pos_rows):
            susp = float(contra[i] - own[i])
            if susp > 0:
                items.append({"cell_id": row_cid[int(row)],
                              "state": 1,
                              "susp": round(susp, 4),
                              "own_sim": round(float(own[i]), 4),
                              "ev_sim": round(float(contra[i]), 4),
                              "ev_cell_id": row_cid[
                                  int(neg_rows[int(np.argmax(CM[i]))])]})
    if len(neg_rows) >= 2 and pos_rows:
        GG = W[neg_rows] @ W[neg_rows].T
        np.fill_diagonal(GG, -2.0)
        CM = W[neg_rows] @ W[pos_rows].T
        own, contra = GG.max(axis=1), CM.max(axis=1)
        for i, row in enumerate(neg_rows):
            susp = float(contra[i] - own[i])
            if susp > 0:
                items.append({"cell_id": row_cid[int(row)],
                              "state": 0,
                              "susp": round(susp, 4),
                              "own_sim": round(float(own[i]), 4),
                              "ev_sim": round(float(contra[i]), 4),
                              "ev_cell_id": row_cid[
                                  int(pos_rows[int(np.argmax(CM[i]))])]})
    items.sort(key=lambda it: -it["susp"])
    return items
