"""Suggestion engines for the label app.

All engines score cells in ONE shared embedding space W (whitened-PCA +
L2-normalized feature rows — the deduplication/vis convention):

  SuggestEngine    per-label nearest-exemplar kNN score (positive
                   similarity minus explicit-negative similarity — PU:
                   undecided is not a negative)
  fit_label_model  per-label logistic refit over the explicit exemplars
                   (the manual Refresh-model action) — negatives shape a
                   real decision boundary instead of only nudging a
                   similarity ranking
  review_items     leave-one-out consistency check over DECIDED cells —
                   flags decided cells whose embedding contradicts their own
                   decision (likely mislabels)

Score arrays are cached per label keyed by the decision counts and the
engine version; bump() invalidates them after every write, so the ranking
always reflects the current exemplars.
"""

import threading

import numpy as np
from sklearn.linear_model import LogisticRegression

# A per-label logistic refit needs BOTH sides in reasonable numbers — below
# that the label keeps the kNN score (sklearn cannot say anything sensible
# from one or two exemplars of a class).
MIN_FIT_POS = 4
MIN_FIT_NEG = 4


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
    accumulate. Explicit negatives PUSH every similar cell down the ranking
    (the score is what the UI shows and what the Collect queue sorts by);
    cells without a decision are never treated as negatives.

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
        result = (pos_part, neg_part)
        with self._lock:
            self._cache[label_id] = (self.version,
                                     (len(pos_rows), len(neg_rows)),
                                     result)
        return result


def fit_label_model(W, pos_rows, neg_rows, seed):
    """Fit one label's logistic scorer over its explicit exemplars, or None.

    The discriminative complement to the kNN score: a per-label logistic
    regression in the SAME whitened space — positives are class 1, explicit
    negatives class 0, undecided cells never enter training (the same PU
    principle as the kNN score). Fitting is MANUAL (the UI's Refresh-model
    button): negatives then define a decision boundary instead of only
    pulling a similarity down, which is what makes Apply − / shift+click
    negatives really bite. class_weight='balanced' keeps a small negative
    set from being drowned by many positives. Returns None when either side
    holds fewer than MIN_FIT exemplars — the caller keeps the kNN score.
    """
    if len(pos_rows) < MIN_FIT_POS or len(neg_rows) < MIN_FIT_NEG:
        return None
    X = np.vstack([W[pos_rows], W[neg_rows]])
    y = np.concatenate([np.ones(len(pos_rows)), np.zeros(len(neg_rows))])
    model = LogisticRegression(class_weight="balanced", max_iter=1000,
                               random_state=seed)
    model.fit(X, y)
    return model


def review_items(W, row_cid, pos_rows, neg_rows):
    """Leave-one-out kNN consistency check over DECIDED cells of a label.

    The active-learning complement to the uncertainty ranking: instead of
    picking new informative cells, flag already-labeled cells whose
    embedding contradicts their own decision (likely mislabels — slips of
    the hand, or a label annotated before it had trustworthy exemplars).
    A positive is suspicious when its nearest
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
