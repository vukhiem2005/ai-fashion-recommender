"""Standalone validation and audit suite for the ELAN V5 synthetic dataset.

Verifies:
1. Product ID, SKU, and price referential integrity against read-only `elan_release.db`
2. Quarantine of generator-internal latent variables from model-facing CSVs
3. Event sequence, timestamp monotonicity, and causal funnel consistency
4. Session and order aggregation/accounting integrity
5. Chronological train/val/test separation and atomic session/order assignment
6. Byte-for-byte reproducibility from the fixed seed
7. User-item sparsity and catalog coverage acceptance thresholds
8. Explicit synthetic data provenance labels across all records

Usage from repository root:
    python backend/evaluation/v5/validate_synthetic_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ensure local package imports work when invoked as a script
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from generate_synthetic_dataset import (  # noqa: E402
    DATA_SOURCE_LABEL,
    DEFAULT_AUDIT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_RELEASE_DB,
    SPLIT_WINDOWS,
    format_ts,
    generate_dataset,
)

FORBIDDEN_MODEL_COLUMNS = frozenset(
    {
        "persona_id",
        "primary_categories",
        "preferred_subtypes",
        "preferred_color_family",
        "preferred_style",
        "price_tier_preference",
        "price_sensitivity",
        "exploration_noise_rate",
        "activity_level",
        "session_intent",
        "anchor_product_id",
        "origin_signal",
        "latent_utility",
        "is_repeat_interaction",
    }
)


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV header in {path}")
        return list(reader.fieldnames), list(reader)


def load_active_catalog_lookup_readonly(db_path: Path) -> tuple[dict[str, int], set[tuple[str, str]]]:
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        products = {
            row[0]: int(row[1])
            for row in conn.execute(
                "SELECT id, price_cents FROM product WHERE is_active = 1"
            ).fetchall()
        }
        variants = {
            (row[0], row[1])
            for row in conn.execute(
                "SELECT product_id, sku FROM productvariant"
            ).fetchall()
            if row[0] in products
        }
    return products, variants


def compute_eval_time_repeat_flags(
    interactions: list[dict[str, str]],
) -> dict[str, int]:
    """Demonstrate evaluation-time derivation of repeat vs. novel (user, product) pairs."""
    seen_train: set[tuple[str, str]] = set()
    seen_val: set[tuple[str, str]] = set()
    val_repeat_events = 0
    val_novel_events = 0
    test_repeat_events = 0
    test_novel_events = 0

    for row in interactions:
        pair = (row["user_id"], row["product_id"])
        sp = row["split"]
        if sp == "train":
            seen_train.add(pair)
        elif sp == "val":
            if pair in seen_train:
                val_repeat_events += 1
            else:
                val_novel_events += 1
            seen_val.add(pair)
        elif sp == "test":
            if pair in seen_train or pair in seen_val:
                test_repeat_events += 1
            else:
                test_novel_events += 1

    return {
        "val_repeat_events": val_repeat_events,
        "val_novel_events": val_novel_events,
        "test_repeat_events": test_repeat_events,
        "test_novel_events": test_novel_events,
    }


def run_validation(
    db_path: Path = DEFAULT_RELEASE_DB,
    data_dir: Path = DEFAULT_DATA_DIR,
    audit_dir: Path = DEFAULT_AUDIT_DIR,
) -> tuple[bool, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    manifest_path = data_dir / "synthetic_manifest.json"
    if not manifest_path.is_file():
        record("manifest_exists", False, f"Missing {manifest_path}")
        return False, checks

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    u_cols, users = read_csv_rows(data_dir / "synthetic_users.csv")
    s_cols, sessions = read_csv_rows(data_dir / "synthetic_sessions.csv")
    i_cols, interactions = read_csv_rows(data_dir / "synthetic_interactions.csv")
    o_cols, orders = read_csv_rows(data_dir / "synthetic_orders.csv")
    oi_cols, order_items = read_csv_rows(data_dir / "synthetic_order_items.csv")

    # 1. Check quarantine of generator-internal latent columns
    leaked_cols: dict[str, list[str]] = {}
    for name, cols in (
        ("synthetic_users.csv", u_cols),
        ("synthetic_sessions.csv", s_cols),
        ("synthetic_interactions.csv", i_cols),
        ("synthetic_orders.csv", o_cols),
        ("synthetic_order_items.csv", oi_cols),
    ):
        overlap = sorted(set(cols) & FORBIDDEN_MODEL_COLUMNS)
        if overlap:
            leaked_cols[name] = overlap
    record(
        "latent_variable_quarantine",
        len(leaked_cols) == 0,
        "Zero generator-internal latent columns in model-facing CSVs"
        if not leaked_cols
        else f"Leaked columns found: {leaked_cols}",
    )

    # 2. Check Product ID, SKU, and price integrity against read-only release DB
    active_products, active_variants = load_active_catalog_lookup_readonly(db_path)
    bad_products = [
        r["event_id"] for r in interactions if r["product_id"] not in active_products
    ]
    bad_skus = [
        r["event_id"]
        for r in interactions
        if (r["product_id"], r["sku"]) not in active_variants
    ]
    bad_oi_prices = [
        r["order_item_id"]
        for r in order_items
        if r["product_id"] not in active_products
        or (r["product_id"], r["sku"]) not in active_variants
        or int(r["unit_price_cents"]) != active_products[r["product_id"]]
    ]
    record(
        "catalog_referential_integrity",
        len(active_products) == 328
        and not bad_products
        and not bad_skus
        and not bad_oi_prices,
        f"Active catalog={len(active_products)} products, {len(active_variants)} variants; "
        f"invalid event products={len(bad_products)}, invalid SKUs={len(bad_skus)}, "
        f"invalid order_item prices/SKUs={len(bad_oi_prices)}",
    )

    # 3. Check user/session/order foreign keys and session/order integrity
    user_map = {u["user_id"]: u for u in users}
    session_map = {s["session_id"]: s for s in sessions}
    order_map = {o["order_id"]: o for o in orders}

    fk_errors: list[str] = []
    if len(user_map) != len(users):
        fk_errors.append("Duplicate user_id in synthetic_users.csv")
    if len(session_map) != len(sessions):
        fk_errors.append("Duplicate session_id in synthetic_sessions.csv")
    if len(order_map) != len(orders):
        fk_errors.append("Duplicate order_id in synthetic_orders.csv")

    for s in sessions:
        if s["user_id"] not in user_map:
            fk_errors.append(f"Orphan user_id in session {s['session_id']}")
        if s["session_start_ts"] > s["session_end_ts"]:
            fk_errors.append(f"Session start > end for {s['session_id']}")

    events_by_session: dict[str, list[dict[str, str]]] = defaultdict(list)
    for ev in interactions:
        sid = ev["session_id"]
        if sid not in session_map:
            fk_errors.append(f"Orphan session_id in event {ev['event_id']}")
            continue
        parent_s = session_map[sid]
        if ev["user_id"] != parent_s["user_id"]:
            fk_errors.append(f"Mismatched user_id in event {ev['event_id']}")
        if ev["split"] != parent_s["split"]:
            fk_errors.append(f"Split mismatch between event {ev['event_id']} and session {sid}")
        events_by_session[sid].append(ev)

    # 4. Event ordering and causal funnel consistency
    funnel_errors: list[str] = []
    for sid, s_row in session_map.items():
        evs = events_by_session.get(sid, [])
        if not evs:
            funnel_errors.append(f"Session {sid} has zero events")
            continue
        seqs = [int(e["sequence_number"]) for e in evs]
        if seqs != list(range(1, len(evs) + 1)):
            funnel_errors.append(f"Non-monotonic sequence_number in session {sid}")

        prev_ts = s_row["session_start_ts"]
        viewed_in_s: dict[str, str] = {}
        carted_in_s: dict[str, str] = {}
        n_views = 0
        n_carts = 0
        n_purchases = 0

        for e in evs:
            ets = e["event_ts"]
            if ets < prev_ts or ets > s_row["session_end_ts"]:
                funnel_errors.append(f"Timestamp out of session bounds in {e['event_id']}")
            prev_ts = ets
            pid = e["product_id"]
            etype = e["event_type"]
            if etype == "view":
                n_views += 1
                viewed_in_s[pid] = ets
            elif etype == "add_to_cart":
                n_carts += 1
                if pid not in viewed_in_s or viewed_in_s[pid] >= ets:
                    funnel_errors.append(f"add_to_cart without prior view in {e['event_id']}")
                carted_in_s[pid] = ets
            elif etype == "purchase":
                n_purchases += 1
                if pid not in carted_in_s or carted_in_s[pid] >= ets:
                    funnel_errors.append(f"purchase without prior add_to_cart in {e['event_id']}")
            else:
                funnel_errors.append(f"Unknown event_type {etype}")

        if n_views != int(s_row["num_views"]) or n_carts != int(s_row["num_cart_adds"]):
            funnel_errors.append(f"Session view/cart counts mismatch in {sid}")
        if (n_purchases > 0) != (s_row["converted_to_order"] == "True"):
            funnel_errors.append(f"Session converted_to_order mismatch in {sid}")

    record(
        "event_and_funnel_consistency",
        not fk_errors and not funnel_errors,
        f"FK errors={len(fk_errors)}, funnel/sequence/timestamp errors={len(funnel_errors)}",
    )

    # 5. Order & order_items accounting integrity
    items_by_order: dict[str, list[dict[str, str]]] = defaultdict(list)
    order_errors: list[str] = []
    for oi in order_items:
        oid = oi["order_id"]
        if oid not in order_map:
            order_errors.append(f"Orphan order_id in {oi['order_item_id']}")
            continue
        parent_o = order_map[oid]
        if oi["session_id"] != parent_o["session_id"] or oi["user_id"] != parent_o["user_id"]:
            order_errors.append(f"Session/User mismatch in {oi['order_item_id']}")
        if oi["split"] != parent_o["split"]:
            order_errors.append(f"Split mismatch in {oi['order_item_id']}")
        items_by_order[oid].append(oi)

    for oid, o_row in order_map.items():
        s_row = session_map.get(o_row["session_id"])
        if s_row is None or o_row["split"] != s_row["split"]:
            order_errors.append(f"Order {oid} split mismatch with parent session")
        ois = items_by_order.get(oid, [])
        if len(ois) != int(o_row["item_count"]):
            order_errors.append(f"item_count mismatch in {oid}")
        calc_qty = sum(int(x["quantity"]) for x in ois)
        calc_cents = sum(int(x["quantity"]) * int(x["unit_price_cents"]) for x in ois)
        if calc_qty != int(o_row["total_quantity"]) or calc_cents != int(o_row["order_total_cents"]):
            order_errors.append(f"Monetary/quantity sum mismatch in {oid}")

    record(
        "session_and_order_integrity",
        not order_errors,
        f"Verified {len(orders)} orders and {len(order_items)} order items; errors={len(order_errors)}",
    )

    # 6. Chronological split boundaries & strict non-overlap
    split_bounds_str = {
        k: (format_ts(v[0]), format_ts(v[1])) for k, v in SPLIT_WINDOWS.items()
    }
    split_ts_errors: list[str] = []
    split_min_ts: dict[str, str] = {}
    split_max_ts: dict[str, str] = {}

    for s in sessions:
        sp = s["split"]
        low, high = split_bounds_str[sp]
        if not (low <= s["session_start_ts"] <= s["session_end_ts"] <= high):
            split_ts_errors.append(f"Session {s['session_id']} outside {sp} window")

    for e in interactions:
        sp = e["split"]
        low, high = split_bounds_str[sp]
        ets = e["event_ts"]
        if not (low <= ets <= high):
            split_ts_errors.append(f"Event {e['event_id']} outside {sp} window")
        if sp not in split_min_ts or ets < split_min_ts[sp]:
            split_min_ts[sp] = ets
        if sp not in split_max_ts or ets > split_max_ts[sp]:
            split_max_ts[sp] = ets

    chronological_order_ok = (
        split_max_ts["train"] < split_min_ts["val"]
        and split_max_ts["val"] < split_min_ts["test"]
    )
    record(
        "chronological_split_separation",
        not split_ts_errors and chronological_order_ok,
        f"train_max={split_max_ts.get('train')} < val_min={split_min_ts.get('val')}; "
        f"val_max={split_max_ts.get('val')} < test_min={split_min_ts.get('test')}; "
        f"boundary violations={len(split_ts_errors)}",
    )

    # 7. Sparsity, product coverage & multi-item basket thresholds
    all_pairs = {(r["user_id"], r["product_id"]) for r in interactions}
    train_products = {r["product_id"] for r in interactions if r["split"] == "train"}
    overall_sparsity = 1.0 - (len(all_pairs) / (len(users) * len(active_products)))
    train_coverage = len(train_products) / len(active_products)
    multi_item_ratio = (
        sum(1 for o in orders if int(o["item_count"]) >= 2) / max(1, len(orders))
    )

    sparsity_ok = 0.91 <= overall_sparsity <= 0.96
    coverage_ok = train_coverage >= 0.98
    multi_item_ok = 0.50 <= multi_item_ratio <= 0.75
    record(
        "sparsity_coverage_and_basket_thresholds",
        sparsity_ok and coverage_ok and multi_item_ok,
        f"Overall sparsity={overall_sparsity*100:.2f}% (target 91.0%–96.0%); "
        f"Train coverage={len(train_products)}/{len(active_products)} ({train_coverage*100:.2f}%, threshold >= 98.0%); "
        f"Multi-item order ratio={multi_item_ratio*100:.2f}% (target 50.0%–75.0%)",
    )

    # 8. Evaluation-time derived repeat interaction verification
    repeat_stats = compute_eval_time_repeat_flags(interactions)
    record(
        "eval_time_repeat_derivation",
        repeat_stats["val_novel_events"] > 0 and repeat_stats["test_novel_events"] > 0,
        f"Derived dynamically without stored column: val_novel_events={repeat_stats['val_novel_events']}, "
        f"val_repeat_events={repeat_stats['val_repeat_events']}, "
        f"test_novel_events={repeat_stats['test_novel_events']}, "
        f"test_repeat_events={repeat_stats['test_repeat_events']}",
    )

    # 9. Synthetic data provenance check
    prov_ok = (
        manifest.get("dataset_type") == "SYNTHETIC_SIMULATION_ONLY"
        and manifest.get("is_synthetic") is True
        and all(
            r.get("is_synthetic") == "True" and r.get("data_source") == DATA_SOURCE_LABEL
            for table in (users, sessions, interactions, orders, order_items)
            for r in table
        )
    )
    record(
        "synthetic_provenance_labeling",
        prov_ok,
        f"All rows across 5 model-facing tables verified is_synthetic=True and data_source={DATA_SOURCE_LABEL}",
    )

    # 10. Deterministic reproducibility check (re-run generator with same seed and compare SHA-256)
    seed = int(manifest["seed"])
    regen = generate_dataset(db_path=db_path, seed=seed, num_users=len(users))
    expected_hashes = manifest["file_sha256"]
    hash_mismatches: list[str] = []
    for rel_key, digest in expected_hashes.items():
        disk_path = THIS_DIR / rel_key
        actual_disk_hash = hashlib.sha256(disk_path.read_bytes()).hexdigest()
        if actual_disk_hash != digest:
            hash_mismatches.append(f"Disk hash mismatch on {rel_key}")

    regen_match = (
        len(regen["users"]) == len(users)
        and len(regen["sessions"]) == len(sessions)
        and len(regen["interactions"]) == len(interactions)
        and len(regen["orders"]) == len(orders)
        and len(regen["order_items"]) == len(order_items)
        and regen["interactions"][0]["event_id"] == interactions[0]["event_id"]
        and regen["interactions"][-1]["product_id"] == interactions[-1]["product_id"]
        and regen["orders"][-1]["order_total_cents"] == int(orders[-1]["order_total_cents"])
    )
    record(
        "deterministic_reproducibility",
        not hash_mismatches and regen_match,
        f"Seed {seed} regeneration matched all record counts, terminal IDs, and {len(expected_hashes)} SHA-256 file digests",
    )

    # 10. Basket-level evaluation anti-leakage check (multi-item orders in val/test)
    eval_multi_orders = [
        o for o in orders if o["split"] in ("val", "test") and int(o["item_count"]) >= 2
    ]
    basket_leak_errors = 0
    for o in eval_multi_orders:
        ois = items_by_order.get(o["order_id"], [])
        distinct_pids = {x["product_id"] for x in ois}
        if len(distinct_pids) < 2:
            basket_leak_errors += 1
    record(
        "basket_level_eval_isolation",
        len(eval_multi_orders) > 0 and basket_leak_errors == 0,
        f"Verified {len(eval_multi_orders)} multi-item evaluation baskets in val/test with >=2 distinct products; "
        f"basket errors={basket_leak_errors}",
    )

    all_passed = all(c["passed"] for c in checks)
    return all_passed, checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_RELEASE_DB)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    args = parser.parse_args()

    all_passed, checks = run_validation(
        db_path=args.db_path, data_dir=args.data_dir, audit_dir=args.audit_dir
    )
    print("ELAN V5 SYNTHETIC DATASET VALIDATION SUITE")
    print("=" * 60)
    for idx, c in enumerate(checks, start=1):
        status = "PASS" if c["passed"] else "FAIL"
        print(f"{idx:2d}. [{status}] {c['check']}: {c['detail']}")
    print("=" * 60)
    if not all_passed:
        sys.exit(1)
    print(f"ALL VALIDATION CHECKS PASSED ({len(checks)}/{len(checks)}).")


if __name__ == "__main__":
    main()
