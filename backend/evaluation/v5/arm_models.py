"""Association Rule Mining (ARM) models and baselines for ELAN V5 (Milestone 3).

Implements:
1. Reproducible transaction-building pipeline (`build_order_transactions` and
   `build_session_cart_transactions`) with within-order product deduplication
   and cross-table chronological cart-order verification.
2. `OrderPopularityBaseline`: Ranks unseen products by training order support count.
3. `PairCooccurrenceBaseline`: Lightweight item-pair co-occurrence baseline using
   smoothed conditional probability P(y | x) without multi-item antecedents or Lift gates.
4. `AprioriAssociationRuleMiner`: Exact level-wise Apriori frequent itemset miner
   (up to itemset size k=3, supporting 1->1 and 2->1 rules) using vertical
   transaction ID sets (`frozenset[str]`) and downward-closure pruning.
5. `FPGrowthMiner`: Exact FP-Tree / conditional pattern base frequent itemset miner
   used to independently verify 100% mathematical itemset equivalence with Apriori.

Safety & Lifecycle Design:
- Explicit model lifecycle: `.fit(...)` is completely separate from `.recommend_for_basket(...)`.
- No global state mutation, no model fitting on import, and zero database reads during inference.
- Product IDs are strings; all sorting uses `(-round(score, 12), ..., product_id)` with
  `reverse=False` so numeric scores sort descending and string `product_id`s break ties
  in ascending lexicographical order.
"""

from __future__ import annotations

import itertools
import time
import tracemalloc
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class BasketTransaction:
    """Single deduplicated shopping basket transaction preserving split and item order."""

    transaction_id: str
    session_id: str
    user_id: str
    timestamp: str
    split: str
    ordered_products: tuple[str, ...]  # Deduplicated products in verified chronological cart order
    itemset: frozenset[str]            # Unordered set of distinct product_ids for itemset mining


@dataclass(frozen=True)
class AssociationRule:
    """Single directed association rule X => consequent with explicit count & relative metrics."""

    antecedent: tuple[str, ...]        # Sorted tuple of product_ids (len 1 or 2)
    consequent: str                    # Single target product_id not in antecedent
    support_count: int                 # Integer count(X union {y}) in training baskets
    support: float                     # Relative support = support_count / n_transactions
    antecedent_count: int              # Integer count(X) in training baskets
    antecedent_support: float          # count(X) / n_transactions
    consequent_count: int              # Integer count({y}) in training baskets
    consequent_support: float          # count({y}) / n_transactions
    confidence: float                  # count(X union {y}) / count(X)
    smoothed_confidence: float         # count(X union {y}) / (count(X) + alpha_smooth)
    lift: float                        # confidence / consequent_support (>1 positive, =1 independent, <1 negative)
    leverage: float                    # support - (antecedent_support * consequent_support)


def build_order_transactions(
    orders_rows: Sequence[dict[str, str]],
    order_items_rows: Sequence[dict[str, str]],
    interactions_rows: Sequence[dict[str, str]] | None = None,
    split_filter: str | None = None,
) -> tuple[list[BasketTransaction], dict[str, Any]]:
    """Build order-level purchase transactions and verify chronological cart ordering.

    Deduplicates repeated `product_id` entries within the same order while preserving
    original order boundaries and verifying whether `order_item_id` sequence matches
    the session's chronological `add_to_cart` sequence in `synthetic_interactions.csv`.
    """
    orders_by_id: dict[str, dict[str, str]] = {}
    for row in orders_rows:
        sp = row["split"].strip()
        if split_filter is not None and sp != split_filter:
            continue
        oid = row["order_id"].strip()
        if oid in orders_by_id:
            raise ValueError(f"Duplicate order_id in orders table: {oid}")
        orders_by_id[oid] = row

    items_by_order: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in order_items_rows:
        oid = row["order_id"].strip()
        if oid not in orders_by_id:
            if split_filter is None:
                raise KeyError(f"Orphan order_id in order_items: {oid}")
            continue
        if row["split"].strip() != orders_by_id[oid]["split"].strip():
            raise ValueError(f"Split mismatch on order_item {row['order_item_id']}")
        items_by_order[oid].append(row)

    # Optional cross-check against session add_to_cart sequence in interactions
    cart_order_by_session: dict[str, list[str]] = defaultdict(list)
    if interactions_rows is not None:
        sorted_evs = sorted(
            (r for r in interactions_rows if r["event_type"].strip() == "add_to_cart"),
            key=lambda r: (r["session_id"].strip(), int(r["sequence_number"])),
        )
        for ev in sorted_evs:
            sid = ev["session_id"].strip()
            pid = ev["product_id"].strip()
            if pid not in cart_order_by_session[sid]:
                cart_order_by_session[sid].append(pid)

    transactions: list[BasketTransaction] = []
    raw_line_items_count = 0
    deduplicated_items_count = 0
    duplicate_product_occurrences_removed = 0
    chronological_order_verified_count = 0
    chronological_order_mismatch_count = 0

    for oid in sorted(orders_by_id):
        o_row = orders_by_id[oid]
        sid = o_row["session_id"].strip()
        line_items = sorted(items_by_order.get(oid, []), key=lambda r: r["order_item_id"].strip())
        if not line_items:
            raise ValueError(f"Order {oid} has zero line items")

        raw_line_items_count += len(line_items)
        ordered_unique: list[str] = []
        seen_pids: set[str] = set()
        for li in line_items:
            pid = li["product_id"].strip()
            if pid in seen_pids:
                duplicate_product_occurrences_removed += 1
            else:
                seen_pids.add(pid)
                ordered_unique.append(pid)

        deduplicated_items_count += len(ordered_unique)

        if interactions_rows is not None:
            session_cart_seq = [
                p for p in cart_order_by_session.get(sid, []) if p in seen_pids
            ]
            if session_cart_seq == ordered_unique:
                chronological_order_verified_count += 1
            else:
                chronological_order_mismatch_count += 1

        transactions.append(
            BasketTransaction(
                transaction_id=oid,
                session_id=sid,
                user_id=o_row["user_id"].strip(),
                timestamp=o_row["order_ts"].strip(),
                split=o_row["split"].strip(),
                ordered_products=tuple(ordered_unique),
                itemset=frozenset(ordered_unique),
            )
        )

    multi_item_count = sum(1 for t in transactions if len(t.itemset) >= 2)
    stats = {
        "split_filter": split_filter or "all",
        "total_orders": len(transactions),
        "single_item_orders": len(transactions) - multi_item_count,
        "multi_item_orders": multi_item_count,
        "multi_item_order_ratio": round(multi_item_count / max(1, len(transactions)), 4),
        "raw_line_items_count": raw_line_items_count,
        "deduplicated_product_occurrences": deduplicated_items_count,
        "duplicate_product_occurrences_removed": duplicate_product_occurrences_removed,
        "mean_distinct_products_per_order": round(
            deduplicated_items_count / max(1, len(transactions)), 4
        ),
        "chronological_cart_order_verified_orders": chronological_order_verified_count,
        "chronological_cart_order_mismatches": chronological_order_mismatch_count,
    }
    return transactions, stats


def build_session_cart_transactions(
    interactions_rows: Sequence[dict[str, str]],
    split_filter: str = "train",
) -> list[BasketTransaction]:
    """Build session-level add_to_cart baskets for the isolated secondary experiment."""
    evs = [
        r
        for r in interactions_rows
        if r["event_type"].strip() == "add_to_cart"
        and (split_filter is None or r["split"].strip() == split_filter)
    ]
    evs.sort(key=lambda r: (r["session_id"].strip(), int(r["sequence_number"])))

    by_session: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in evs:
        by_session[r["session_id"].strip()].append(r)

    baskets: list[BasketTransaction] = []
    for sid in sorted(by_session):
        rows = by_session[sid]
        ordered_unique: list[str] = []
        seen: set[str] = set()
        for r in rows:
            pid = r["product_id"].strip()
            if pid not in seen:
                seen.add(pid)
                ordered_unique.append(pid)
        baskets.append(
            BasketTransaction(
                transaction_id=sid,
                session_id=sid,
                user_id=rows[0]["user_id"].strip(),
                timestamp=rows[-1]["event_ts"].strip(),
                split=rows[0]["split"].strip(),
                ordered_products=tuple(ordered_unique),
                itemset=frozenset(ordered_unique),
            )
        )
    return baskets


class OrderPopularityBaseline:
    """Baseline ranking unseen products by their training order frequency."""

    def __init__(self) -> None:
        self.catalog_product_ids: tuple[str, ...] = ()
        self.support_counts: dict[str, int] = {}
        self.relative_supports: dict[str, float] = {}
        self.n_transactions: int = 0
        self.fit_time_seconds: float = 0.0
        self._is_fitted: bool = False

    def fit(
        self,
        train_transactions: Sequence[BasketTransaction],
        catalog_product_ids: Sequence[str],
    ) -> "OrderPopularityBaseline":
        t0 = time.perf_counter()
        for t in train_transactions:
            if t.split != "train":
                raise ValueError(
                    f"Data leakage error: non-train transaction ({t.split}) in OrderPopularityBaseline.fit"
                )
        self.catalog_product_ids = tuple(sorted(set(catalog_product_ids)))
        self.n_transactions = len(train_transactions)
        counts: Counter[str] = Counter()
        for t in train_transactions:
            for pid in t.itemset:
                counts[pid] += 1

        self.support_counts = {pid: int(counts.get(pid, 0)) for pid in self.catalog_product_ids}
        denom = float(max(1, self.n_transactions))
        self.relative_supports = {
            pid: self.support_counts[pid] / denom for pid in self.catalog_product_ids
        }
        self.fit_time_seconds = time.perf_counter() - t0
        self._is_fitted = True
        return self

    def recommend_for_basket(
        self,
        basket_product_ids: Sequence[str],
        top_k: int = 5,
    ) -> dict[str, Any]:
        if not self._is_fitted:
            raise RuntimeError("OrderPopularityBaseline must be fitted before calling recommend_for_basket")
        observed = set(basket_product_ids)
        candidates = [pid for pid in self.catalog_product_ids if pid not in observed]
        candidates.sort(
            key=lambda pid: (-round(self.relative_supports.get(pid, 0.0), 12), pid)
        )
        selected = candidates[:top_k]
        return {
            "input_basket_ids": list(basket_product_ids),
            "model": "order_popularity_baseline",
            "rule_matched_count": 0,
            "fallback_count": len(selected),
            "fallback_used": True,
            "recommendations": [
                {
                    "product_id": pid,
                    "score": round(self.relative_supports.get(pid, 0.0), 6),
                    "support_count": self.support_counts.get(pid, 0),
                    "support": round(self.relative_supports.get(pid, 0.0), 6),
                    "source": "popularity_baseline",
                }
                for pid in selected
            ],
        }


class PairCooccurrenceBaseline:
    """Lightweight item-pair co-occurrence baseline without Lift threshold or 2-item antecedents.

    Scores each candidate y not in B_obs by aggregating smoothed conditional probability:
        score(y | B_obs) = sum_{x in B_obs} count({x, y}) / (count({x}) + smoothing_alpha)
    """

    def __init__(self, smoothing_alpha: float = 2.0) -> None:
        self.smoothing_alpha = float(smoothing_alpha)
        self.catalog_product_ids: tuple[str, ...] = ()
        self.item_counts: dict[str, int] = {}
        self.pair_counts: dict[str, dict[str, int]] = {}
        self.popularity_fallback: OrderPopularityBaseline = OrderPopularityBaseline()
        self.fit_time_seconds: float = 0.0
        self.peak_memory_bytes: int = 0
        self._is_fitted: bool = False

    def fit(
        self,
        train_transactions: Sequence[BasketTransaction],
        catalog_product_ids: Sequence[str],
    ) -> "PairCooccurrenceBaseline":
        tracemalloc.start()
        t0 = time.perf_counter()
        for t in train_transactions:
            if t.split != "train":
                raise ValueError(
                    f"Data leakage error: non-train transaction ({t.split}) in PairCooccurrenceBaseline.fit"
                )
        self.catalog_product_ids = tuple(sorted(set(catalog_product_ids)))
        self.popularity_fallback.fit(train_transactions, self.catalog_product_ids)

        i_counts: Counter[str] = Counter()
        p_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for t in train_transactions:
            items = sorted(t.itemset)
            for x in items:
                i_counts[x] += 1
            for i_idx in range(len(items)):
                x = items[i_idx]
                for j_idx in range(i_idx + 1, len(items)):
                    y = items[j_idx]
                    p_counts[x][y] += 1
                    p_counts[y][x] += 1

        self.item_counts = dict(i_counts)
        self.pair_counts = {k: dict(v) for k, v in p_counts.items()}
        self.fit_time_seconds = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_memory_bytes = int(peak_bytes)
        self._is_fitted = True
        return self

    def recommend_for_basket(
        self,
        basket_product_ids: Sequence[str],
        top_k: int = 5,
        allow_fallback: bool = True,
    ) -> dict[str, Any]:
        if not self._is_fitted:
            raise RuntimeError("PairCooccurrenceBaseline must be fitted before calling recommend_for_basket")
        observed_list = list(dict.fromkeys(basket_product_ids))
        observed_set = set(observed_list)

        candidate_scores: dict[str, float] = defaultdict(float)
        candidate_co_counts: dict[str, int] = defaultdict(int)
        for x in observed_list:
            cnt_x = self.item_counts.get(x, 0)
            if cnt_x <= 0:
                continue
            denom = float(cnt_x) + self.smoothing_alpha
            for y, co_cnt in self.pair_counts.get(x, {}).items():
                if y in observed_set:
                    continue
                candidate_scores[y] += float(co_cnt) / denom
                candidate_co_counts[y] += int(co_cnt)

        pop_supports = self.popularity_fallback.relative_supports
        matched_candidates = sorted(
            candidate_scores.keys(),
            key=lambda pid: (
                -round(candidate_scores[pid], 12),
                -candidate_co_counts[pid],
                -round(pop_supports.get(pid, 0.0), 12),
                pid,
            ),
        )

        recs: list[dict[str, Any]] = []
        for pid in matched_candidates[:top_k]:
            recs.append(
                {
                    "product_id": pid,
                    "score": round(candidate_scores[pid], 6),
                    "cooccurrence_count": candidate_co_counts[pid],
                    "source": "pair_cooccurrence",
                }
            )

        rule_matched_count = len(recs)
        fallback_count = 0
        if allow_fallback and len(recs) < top_k:
            already_chosen = observed_set | {r["product_id"] for r in recs}
            rem_candidates = [
                pid for pid in self.catalog_product_ids if pid not in already_chosen
            ]
            rem_candidates.sort(
                key=lambda pid: (-round(pop_supports.get(pid, 0.0), 12), pid)
            )
            for pid in rem_candidates[: top_k - len(recs)]:
                fallback_count += 1
                recs.append(
                    {
                        "product_id": pid,
                        "score": round(pop_supports.get(pid, 0.0), 6),
                        "cooccurrence_count": 0,
                        "source": "popularity_fallback",
                    }
                )

        return {
            "input_basket_ids": observed_list,
            "model": "pair_cooccurrence_baseline",
            "rule_matched_count": rule_matched_count,
            "fallback_count": fallback_count,
            "fallback_used": fallback_count > 0,
            "recommendations": recs,
        }


class AprioriAssociationRuleMiner:
    """Exact level-wise Apriori frequent itemset and association rule miner using vertical TID sets.

    Supports itemset sizes up to `max_itemset_len` (default 3, yielding 1->1 and 2->1 rules).
    Enforces strict positive association (`lift > 1.0` and `lift >= min_lift`).
    """

    def __init__(
        self,
        min_support_count: int = 2,
        min_confidence: float = 0.10,
        min_lift: float = 1.25,
        max_itemset_len: int = 3,
        alpha_smooth: float = 2.0,
        beta_lift: float = 0.5,
        agg_mode: str = "sum",
        require_multi_antecedent_synergy: bool = True,
    ) -> None:
        if min_support_count < 1:
            raise ValueError("min_support_count must be >= 1")
        if min_lift <= 1.0:
            raise ValueError(
                "min_lift must be strictly > 1.0 to guarantee positive statistical association "
                "(lift=1.0 is independence; lift<1.0 is negative association)"
            )
        if agg_mode not in ("sum", "max"):
            raise ValueError(f"Unsupported agg_mode: {agg_mode}")

        self.min_support_count = int(min_support_count)
        self.min_confidence = float(min_confidence)
        self.min_lift = float(min_lift)
        self.max_itemset_len = int(max_itemset_len)
        self.alpha_smooth = float(alpha_smooth)
        self.beta_lift = float(beta_lift)
        self.agg_mode = str(agg_mode)
        self.require_multi_antecedent_synergy = bool(require_multi_antecedent_synergy)

        self.catalog_product_ids: tuple[str, ...] = ()
        self.n_transactions: int = 0
        self.min_support_relative: float = 0.0
        self.frequent_itemsets_by_len: dict[int, dict[tuple[str, ...], int]] = {}
        self.rules: list[AssociationRule] = []
        self.rules_by_antecedent: dict[tuple[str, ...], list[AssociationRule]] = {}
        self.popularity_fallback: OrderPopularityBaseline = OrderPopularityBaseline()
        self.fit_time_seconds: float = 0.0
        self.peak_memory_bytes: int = 0
        self._is_fitted: bool = False

    def fit(
        self,
        train_transactions: Sequence[BasketTransaction],
        catalog_product_ids: Sequence[str],
    ) -> "AprioriAssociationRuleMiner":
        tracemalloc.start()
        t0 = time.perf_counter()

        for t in train_transactions:
            if t.split != "train":
                raise ValueError(
                    f"Data leakage error: non-train transaction ({t.split}) in AprioriAssociationRuleMiner.fit"
                )

        self.catalog_product_ids = tuple(sorted(set(catalog_product_ids)))
        self.n_transactions = len(train_transactions)
        if self.n_transactions == 0:
            raise ValueError("train_transactions cannot be empty")
        self.min_support_relative = float(self.min_support_count) / float(self.n_transactions)
        self.popularity_fallback.fit(train_transactions, self.catalog_product_ids)

        # Build vertical TID sets for 1-itemsets
        tid_sets_1: dict[str, set[int]] = defaultdict(set)
        for idx, t in enumerate(train_transactions):
            for pid in t.itemset:
                tid_sets_1[pid].add(idx)

        l1_tids: dict[tuple[str, ...], frozenset[int]] = {
            (pid,): frozenset(tids)
            for pid, tids in sorted(tid_sets_1.items())
            if len(tids) >= self.min_support_count
        }
        self.frequent_itemsets_by_len = {
            1: {k: len(v) for k, v in l1_tids.items()}
        }

        current_level_tids = l1_tids
        all_tids_lookup: dict[tuple[str, ...], int] = dict(self.frequent_itemsets_by_len[1])

        # Level-wise Apriori candidate generation with downward-closure pruning
        for k in range(2, self.max_itemset_len + 1):
            prev_keys = sorted(current_level_tids.keys())
            prev_key_set = set(prev_keys)
            next_level_tids: dict[tuple[str, ...], frozenset[int]] = {}

            n_prev = len(prev_keys)
            for i in range(n_prev):
                a = prev_keys[i]
                for j in range(i + 1, n_prev):
                    b = prev_keys[j]
                    # Prefix join: first k-2 items must match
                    if a[: k - 2] != b[: k - 2]:
                        break
                    cand = tuple(sorted(set(a) | set(b)))
                    if len(cand) != k:
                        continue
                    # Apriori downward-closure check: all (k-1)-subsets must be frequent
                    if k > 2:
                        all_subsets_frequent = all(
                            sub in prev_key_set
                            for sub in itertools.combinations(cand, k - 1)
                        )
                        if not all_subsets_frequent:
                            continue

                    inter_tids = current_level_tids[a] & current_level_tids[b]
                    if len(inter_tids) >= self.min_support_count:
                        next_level_tids[cand] = inter_tids

            self.frequent_itemsets_by_len[k] = {
                cand: len(tids) for cand, tids in sorted(next_level_tids.items())
            }
            all_tids_lookup.update(self.frequent_itemsets_by_len[k])
            current_level_tids = next_level_tids
            if not current_level_tids:
                break

        # Generate 1->1 rules first so we can check 2->1 multi-antecedent synergy
        n_tx = float(self.n_transactions)
        single_rule_conf: dict[tuple[str, str], float] = {}
        candidate_rules: list[AssociationRule] = []

        for k in range(2, self.max_itemset_len + 1):
            freq_k = self.frequent_itemsets_by_len.get(k, {})
            for itemset_tuple, supp_cnt in freq_k.items():
                supp_rel = float(supp_cnt) / n_tx
                for consequent in itemset_tuple:
                    antecedent = tuple(x for x in itemset_tuple if x != consequent)
                    ant_cnt = all_tids_lookup[antecedent]
                    cons_cnt = all_tids_lookup[(consequent,)]
                    ant_supp = float(ant_cnt) / n_tx
                    cons_supp = float(cons_cnt) / n_tx

                    conf = float(supp_cnt) / float(ant_cnt)
                    smoothed_conf = float(supp_cnt) / (float(ant_cnt) + self.alpha_smooth)
                    lift = conf / cons_supp
                    leverage = supp_rel - (ant_supp * cons_supp)

                    if len(antecedent) == 1:
                        single_rule_conf[(antecedent[0], consequent)] = conf

                    if conf < self.min_confidence:
                        continue
                    # Strictly enforce positive association (lift > 1.0 and lift >= min_lift)
                    if lift <= 1.0 or lift < self.min_lift:
                        continue

                    if len(antecedent) >= 2 and self.require_multi_antecedent_synergy:
                        max_sub_conf = max(
                            single_rule_conf.get(
                                (sub_item, consequent),
                                float(all_tids_lookup.get(tuple(sorted((sub_item, consequent))), 0))
                                / float(all_tids_lookup[(sub_item,)]),
                            )
                            for sub_item in antecedent
                        )
                        if conf <= max_sub_conf + 1e-12:
                            continue

                    candidate_rules.append(
                        AssociationRule(
                            antecedent=antecedent,
                            consequent=consequent,
                            support_count=int(supp_cnt),
                            support=round(supp_rel, 6),
                            antecedent_count=int(ant_cnt),
                            antecedent_support=round(ant_supp, 6),
                            consequent_count=int(cons_cnt),
                            consequent_support=round(cons_supp, 6),
                            confidence=round(conf, 6),
                            smoothed_confidence=round(smoothed_conf, 6),
                            lift=round(lift, 6),
                            leverage=round(leverage, 6),
                        )
                    )

        # Sort rules deterministically
        candidate_rules.sort(
            key=lambda r: (
                -round(r.smoothed_confidence * (r.lift ** self.beta_lift), 12),
                -round(r.lift, 12),
                -r.support_count,
                r.antecedent,
                r.consequent,
            )
        )
        self.rules = candidate_rules
        by_ant: dict[tuple[str, ...], list[AssociationRule]] = defaultdict(list)
        for r in self.rules:
            by_ant[r.antecedent].append(r)
        self.rules_by_antecedent = dict(by_ant)

        self.fit_time_seconds = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_memory_bytes = int(peak_bytes)
        self._is_fitted = True
        return self

    def _rule_relevance_score(self, rule: AssociationRule) -> float:
        synergy_mult = 1.0 + 0.25 * (len(rule.antecedent) - 1)
        return float(rule.smoothed_confidence) * (float(rule.lift) ** self.beta_lift) * synergy_mult

    def recommend_for_basket(
        self,
        basket_product_ids: Sequence[str],
        top_k: int = 5,
        allow_fallback: bool = True,
    ) -> dict[str, Any]:
        """Return Top-K complementary products for `basket_product_ids`."""
        if not self._is_fitted:
            raise RuntimeError(
                "AprioriAssociationRuleMiner must be fitted before calling recommend_for_basket"
            )
        observed_list = list(dict.fromkeys(basket_product_ids))
        observed_set = set(observed_list)
        sorted_observed = sorted(observed_set)

        # Enumerate all non-empty subsets of size 1..(max_itemset_len - 1) present in basket
        matched_rules_by_consequent: dict[str, list[AssociationRule]] = defaultdict(list)
        max_ant_len = min(len(sorted_observed), self.max_itemset_len - 1)
        for ant_len in range(1, max_ant_len + 1):
            for ant_tuple in itertools.combinations(sorted_observed, ant_len):
                for rule in self.rules_by_antecedent.get(ant_tuple, ()):
                    if rule.consequent not in observed_set:
                        matched_rules_by_consequent[rule.consequent].append(rule)

        scored_consequents: dict[str, dict[str, Any]] = {}
        for cons_pid, c_rules in matched_rules_by_consequent.items():
            rule_scores = [self._rule_relevance_score(r) for r in c_rules]
            agg_score = sum(rule_scores) if self.agg_mode == "sum" else max(rule_scores)
            best_rule = max(
                c_rules,
                key=lambda r: (
                    round(self._rule_relevance_score(r), 12),
                    round(r.lift, 12),
                    r.support_count,
                ),
            )
            scored_consequents[cons_pid] = {
                "product_id": cons_pid,
                "score": agg_score,
                "max_lift": max(r.lift for r in c_rules),
                "max_confidence": max(r.confidence for r in c_rules),
                "max_support_count": max(r.support_count for r in c_rules),
                "matched_rules_count": len(c_rules),
                "best_antecedent": list(best_rule.antecedent),
                "best_rule_confidence": best_rule.confidence,
                "best_rule_lift": best_rule.lift,
                "best_rule_support_count": best_rule.support_count,
                "best_rule_support": best_rule.support,
            }

        pop_supports = self.popularity_fallback.relative_supports
        ranked_rule_pids = sorted(
            scored_consequents.keys(),
            key=lambda pid: (
                -round(scored_consequents[pid]["score"], 12),
                -round(scored_consequents[pid]["max_lift"], 12),
                -scored_consequents[pid]["max_support_count"],
                -round(pop_supports.get(pid, 0.0), 12),
                pid,
            ),
        )

        recommendations: list[dict[str, Any]] = []
        for pid in ranked_rule_pids[:top_k]:
            info = scored_consequents[pid]
            ant_str = " + ".join(info["best_antecedent"])
            recommendations.append(
                {
                    "product_id": pid,
                    "score": round(info["score"], 6),
                    "matched_antecedent": info["best_antecedent"],
                    "matched_rules_count": info["matched_rules_count"],
                    "confidence": info["best_rule_confidence"],
                    "lift": info["best_rule_lift"],
                    "support_count": info["best_rule_support_count"],
                    "support": info["best_rule_support"],
                    "source": "association_rule",
                    "reason": (
                        f"frequently bought with {ant_str} "
                        f"(conf={info['best_rule_confidence']*100:.1f}%, lift={info['best_rule_lift']:.2f}x)"
                    ),
                }
            )

        rule_matched_count = len(recommendations)
        fallback_count = 0
        if allow_fallback and len(recommendations) < top_k:
            excluded = observed_set | {r["product_id"] for r in recommendations}
            fallback_cands = [
                pid for pid in self.catalog_product_ids if pid not in excluded
            ]
            fallback_cands.sort(
                key=lambda pid: (-round(pop_supports.get(pid, 0.0), 12), pid)
            )
            for pid in fallback_cands[: top_k - len(recommendations)]:
                fallback_count += 1
                recommendations.append(
                    {
                        "product_id": pid,
                        "score": round(pop_supports.get(pid, 0.0), 6),
                        "matched_antecedent": [],
                        "matched_rules_count": 0,
                        "confidence": 0.0,
                        "lift": 1.0,
                        "support_count": self.popularity_fallback.support_counts.get(pid, 0),
                        "support": round(pop_supports.get(pid, 0.0), 6),
                        "source": "popularity_fallback",
                        "reason": "popular in training orders (fallback)",
                    }
                )

        return {
            "input_basket_ids": observed_list,
            "model": "apriori_arm_v5" if allow_fallback else "apriori_arm_pure_v5",
            "rule_matched_count": rule_matched_count,
            "fallback_count": fallback_count,
            "fallback_used": fallback_count > 0,
            "recommendations": recommendations,
        }


class _FPNode:
    """Internal prefix-tree node for FP-Growth frequent itemset mining."""

    __slots__ = ("item", "count", "parent", "children", "node_link")

    def __init__(self, item: str | None, count: int, parent: "_FPNode | None") -> None:
        self.item = item
        self.count = count
        self.parent = parent
        self.children: dict[str, _FPNode] = {}
        self.node_link: _FPNode | None = None


class FPGrowthMiner:
    """Exact FP-Growth frequent itemset miner for equivalence verification against Apriori."""

    def __init__(self, min_support_count: int = 2, max_itemset_len: int = 3) -> None:
        if min_support_count < 1:
            raise ValueError("min_support_count must be >= 1")
        self.min_support_count = int(min_support_count)
        self.max_itemset_len = int(max_itemset_len)
        self.frequent_itemsets_by_len: dict[int, dict[tuple[str, ...], int]] = {}
        self.fit_time_seconds: float = 0.0
        self.peak_memory_bytes: int = 0
        self._is_fitted: bool = False

    def _build_fptree(
        self, weighted_transactions: list[tuple[list[str], int]]
    ) -> tuple[_FPNode, dict[str, tuple[int, _FPNode | None]]]:
        item_counts: Counter[str] = Counter()
        for items, weight in weighted_transactions:
            for item in items:
                item_counts[item] += weight

        frequent_items = {
            item: cnt
            for item, cnt in item_counts.items()
            if cnt >= self.min_support_count
        }
        if not frequent_items:
            return _FPNode(None, 0, None), {}

        # Canonical ordering: descending support count, then ascending item ID
        order_key = lambda x: (-frequent_items[x], x)
        header_table: dict[str, tuple[int, _FPNode | None]] = {
            item: (frequent_items[item], None)
            for item in sorted(frequent_items, key=order_key)
        }
        root = _FPNode(None, 0, None)

        for items, weight in weighted_transactions:
            ordered = sorted(
                (item for item in items if item in frequent_items), key=order_key
            )
            curr = root
            for item in ordered:
                if item in curr.children:
                    curr.children[item].count += weight
                else:
                    new_node = _FPNode(item, weight, curr)
                    curr.children[item] = new_node
                    tot_cnt, head_node = header_table[item]
                    if head_node is None:
                        header_table[item] = (tot_cnt, new_node)
                    else:
                        while head_node.node_link is not None:
                            head_node = head_node.node_link
                        head_node.node_link = new_node
                curr = curr.children[item]

        return root, header_table

    def _mine_tree(
        self,
        header_table: dict[str, tuple[int, _FPNode | None]],
        suffix: tuple[str, ...],
        results: dict[int, dict[tuple[str, ...], int]],
    ) -> None:
        # Process items from bottom of header table (least frequent first)
        for item in reversed(list(header_table.keys())):
            supp_cnt, node = header_table[item]
            new_pattern = tuple(sorted((item, *suffix)))
            k = len(new_pattern)
            if k <= self.max_itemset_len:
                results.setdefault(k, {})[new_pattern] = int(supp_cnt)

            if k >= self.max_itemset_len:
                continue

            # Build conditional pattern base for new_pattern
            cond_patterns: list[tuple[list[str], int]] = []
            curr = node
            while curr is not None:
                path: list[str] = []
                parent = curr.parent
                while parent is not None and parent.item is not None:
                    path.append(parent.item)
                    parent = parent.parent
                if path:
                    cond_patterns.append((path, curr.count))
                curr = curr.node_link

            if cond_patterns:
                _, cond_header = self._build_fptree(cond_patterns)
                if cond_header:
                    self._mine_tree(cond_header, new_pattern, results)

    def fit(self, train_transactions: Sequence[BasketTransaction]) -> "FPGrowthMiner":
        tracemalloc.start()
        t0 = time.perf_counter()
        for t in train_transactions:
            if t.split != "train":
                raise ValueError("Non-train transaction passed to FPGrowthMiner.fit")

        weighted_tx = [(sorted(t.itemset), 1) for t in train_transactions if t.itemset]
        _, header = self._build_fptree(weighted_tx)
        mined: dict[int, dict[tuple[str, ...], int]] = {
            k: {} for k in range(1, self.max_itemset_len + 1)
        }
        if header:
            self._mine_tree(header, (), mined)

        self.frequent_itemsets_by_len = {
            k: dict(sorted(v.items())) for k, v in sorted(mined.items()) if v
        }
        self.fit_time_seconds = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_memory_bytes = int(peak_bytes)
        self._is_fitted = True
        return self


def recommend_frequently_bought_together(
    fitted_model: AprioriAssociationRuleMiner | PairCooccurrenceBaseline | OrderPopularityBaseline,
    basket_product_ids: Sequence[str],
    top_k: int = 5,
    allow_fallback: bool = True,
) -> dict[str, Any]:
    """Explicit module-level FBT helper requiring an already-fitted model instance."""
    if isinstance(fitted_model, OrderPopularityBaseline):
        return fitted_model.recommend_for_basket(basket_product_ids, top_k=top_k)
    return fitted_model.recommend_for_basket(
        basket_product_ids, top_k=top_k, allow_fallback=allow_fallback
    )
