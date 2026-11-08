r"""Reproduce the frozen, blind 18-reference V1/V2/V3/V4 recommendation evaluation.

No model execution, database writes or external dependencies. Reads the saved
four-model union and the blind relevance labels only; generates reproducible
metrics/reports after checking that every expected pair and Top-5 rank is valid.

Example from repository root (Windows PowerShell):
    .\backend\.venv\Scripts\python.exe .\backend\evaluation\evaluate_holdout_v4_results.py

Default inputs (beside this script):
    holdout_v4_candidate_union.csv
    holdout_v4_blind_labeled.csv
Outputs (beside this script):
    holdout_v4_candidate_union_labeled.csv
    holdout_v4_metrics.json
    holdout_v4_evaluation_report.md
Outputs are NOT overwritten unless --overwrite is supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

MODELS = ("v1", "v2", "v3", "v4")
EXPECTED_REFS = 18
EXPECTED_POSITIONS = 5
EXPECTED_UNION = 140
BOOTSTRAP_REPS = 20000
BOOTSTRAP_SEED = 20261108


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def key(row: dict[str, str]) -> tuple[str, str]:
    return row["reference_id"].strip(), row["recommended_id"].strip()


def dcg(relevances: list[int]) -> float:
    # Gain is 2**relevance - 1; log2 discounts start at rank 1.
    return sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(relevances))


def avg(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def bootstrap_mean_difference(
    paired: list[tuple[float, float]], n: int = BOOTSTRAP_REPS
) -> tuple[float, float, float]:
    """Paired-reference percentile bootstrap, 95% descriptive interval."""
    diffs = [a - b for a, b in paired]
    rng = random.Random(BOOTSTRAP_SEED)
    means = []
    for _ in range(n):
        means.append(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs))
    means.sort()
    return statistics.mean(diffs), means[int(n * 0.025)], means[int(n * 0.975)]


def evaluate(union: list[dict[str, str]], blind: list[dict[str, str]]):
    if len(union) != EXPECTED_UNION or len(blind) != EXPECTED_UNION:
        raise ValueError(f"Expected {EXPECTED_UNION} union and label rows, got {len(union)} and {len(blind)}")

    ufields = {"reference_id", "reference_name", "reference_category", "recommended_id",
               "recommended_name", "recommended_category", "blind_order"} | {f"{m}_rank" for m in MODELS}
    if not ufields.issubset(union[0]):
        raise ValueError("Union CSV missing fields: " + str(sorted(ufields - set(union[0]))))
    if not {"reference_id", "recommended_id", "rank", "relevance_0_1_2"}.issubset(blind[0]):
        raise ValueError("Blind label CSV missing required fields")

    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for item in blind:
        pair = key(item)
        if pair in lookup:
            raise ValueError(f"Duplicate blind label for {pair}")
        if item["relevance_0_1_2"].strip() not in {"0", "1", "2"}:
            raise ValueError(f"Invalid relevance label for {pair}")
        lookup[pair] = item

    keys = [key(row) for row in union]
    if len(set(keys)) != len(keys) or set(keys) != set(lookup):
        raise ValueError("Union and blind label pair sets differ or contain duplicates")

    merged = []
    by_reference = defaultdict(list)
    for row in union:
        pair = key(row)
        lab = lookup[pair]
        if str(row["blind_order"]).strip() != str(lab["rank"]).strip():
            raise ValueError(f"Blind order differs for {pair}")
        if any(str(row[field]) != str(lab[field]) for field in (
            "reference_name", "reference_category", "recommended_name", "recommended_category"
        )):
            raise ValueError(f"Catalog fields changed for {pair}")
        if row["relevance_0_1_2"].strip() and row["relevance_0_1_2"].strip() != lab["relevance_0_1_2"].strip():
            raise ValueError(f"Conflicting preexisting relevance label for {pair}")
        result = dict(row)
        result["relevance_0_1_2"] = lab["relevance_0_1_2"].strip()
        result["notes"] = lab.get("notes", "")
        merged.append(result)
        by_reference[row["reference_id"].strip()].append(result)

    if len(by_reference) != EXPECTED_REFS:
        raise ValueError(f"Expected {EXPECTED_REFS} reference IDs; got {len(by_reference)}")
    cat_counts = Counter(group[0]["reference_category"] for group in by_reference.values())
    if len(cat_counts) != 6 or set(cat_counts.values()) != {3}:
        raise ValueError(f"Expected 3 references per category, got {dict(cat_counts)}")

    per_reference = []
    for ref in sorted(by_reference):
        grp = by_reference[ref]
        ref_name = grp[0]["reference_name"]
        cat = grp[0]["reference_category"]
        ideal_gains = sorted((int(row["relevance_0_1_2"]) for row in grp), reverse=True)[:EXPECTED_POSITIONS]
        ideal_dcg = dcg(ideal_gains)
        for model in MODELS:
            rank_key = f"{model}_rank"
            selected = []
            for row in grp:
                value = row[rank_key].strip()
                if value:
                    rank = int(value)
                    selected.append((rank, row))
            selected.sort(key=lambda item: item[0])
            if [rank for rank, _ in selected] != list(range(1, EXPECTED_POSITIONS + 1)):
                raise ValueError(f"Invalid Top-5 for {ref}, {model}: {[i[0] for i in selected]}")
            grades = [int(r["relevance_0_1_2"]) for _, r in selected]
            related = sum(int(x >= 1) for x in grades)
            close = grades.count(2)
            per_reference.append({
                "reference_id": ref,
                "reference_name": ref_name,
                "category": cat,
                "model": model.upper(),
                "grades": grades,
                "candidate_ids": [r["recommended_id"] for _, r in selected],
                "related_count": related,
                "strict_count": close,
                "precision_at_5": related / EXPECTED_POSITIONS,
                "strict_precision_at_5": close / EXPECTED_POSITIONS,
                "dcg_at_5": dcg(grades),
                "ideal_dcg_at_5": ideal_dcg,
                # Undefined if every candidate in this pooled set has relevance zero.
                "ndcg_at_5": dcg(grades) / ideal_dcg if ideal_dcg > 0 else None,
            })
    summary = {}
    for model in MODELS:
        rows = [r for r in per_reference if r["model"] == model.upper()]
        ndcg_vals = [r["ndcg_at_5"] for r in rows if r["ndcg_at_5"] is not None]
        summary[model.upper()] = {
            "precision_at_5": avg([r["precision_at_5"] for r in rows]),
            "strict_precision_at_5": avg([r["strict_precision_at_5"] for r in rows]),
            "ndcg_at_5_mean_valid_queries": avg(ndcg_vals),
            "ndcg_at_5_mean_zero_as_zero": sum(ndcg_vals) / len(rows),
            "relevant_recommendation_slots": sum(r["related_count"] for r in rows),
            "strict_recommendation_slots": sum(r["strict_count"] for r in rows),
            "total_recommendation_slots": len(rows) * EXPECTED_POSITIONS,
            "queries_with_defined_ndcg": len(ndcg_vals),
        }
    categories = {}
    for cat in sorted(cat_counts):
        categories[cat] = {}
        for model in MODELS:
            rows = [r for r in per_reference if r["model"] == model.upper() and r["category"] == cat]
            categories[cat][model.upper()] = {
                "precision_at_5": avg([r["precision_at_5"] for r in rows]),
                "strict_precision_at_5": avg([r["strict_precision_at_5"] for r in rows]),
            }
    by_ref_model = {(r["reference_id"], r["model"]): r for r in per_reference}
    ci = {}
    for comparator in ("V1", "V2", "V3"):
        ci["V4_minus_" + comparator] = {}
        for metric in ("precision_at_5", "strict_precision_at_5", "ndcg_at_5"):
            pairs = [
                (by_ref_model[(ref, "V4")][metric], by_ref_model[(ref, comparator)][metric])
                for ref in sorted(by_reference)
            ]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            mean, low, high = bootstrap_mean_difference(pairs)
            ci["V4_minus_" + comparator][metric] = {
                "mean_difference": mean, "bootstrap_95_percentile_lower": low,
                "bootstrap_95_percentile_upper": high, "paired_queries": len(pairs)
            }
    zero_refs = sorted(ref for ref, grp in by_reference.items()
                       if max(int(r["relevance_0_1_2"]) for r in grp) == 0)
    rankings_equal_v2_v3 = all(
        by_ref_model[(ref, "V2")]["candidate_ids"] == by_ref_model[(ref, "V3")]["candidate_ids"]
        for ref in by_reference
    )
    output = {
        "method": {
            "source": "18 new H&M references; blind AI-labeled union candidate pool",
            "relevance_labels": {"0": "not relevant", "1": "related", "2": "close substitute"},
            "top_k": EXPECTED_POSITIONS, "references": EXPECTED_REFS,
            "union_pairs": len(merged), "models": list(summary),
            "dcg_gain": "2**relevance - 1",
            "ndcg_ideal": "Top-5 labels sorted descending from pooled union candidates per reference",
            "ndcg_zero_ideal": "undefined; excluded from mean_valid_queries, reported zero-as-zero as sensitivity",
            "zero_ideal_reference_ids": zero_refs,
            "bootstrap": f"{BOOTSTRAP_REPS} paired reference-resampling replicates, seed {BOOTSTRAP_SEED}",
            "v2_v3_identical_rankings_on_this_holdout": rankings_equal_v2_v3,
            "evaluation_limitations": [
                "Human-independent ground truth not available: the relevance labels were AI-assisted.",
                "No evidence of online conversion, personalization, or performance at larger catalog scale.",
                "18 references and 3 per category imply wide uncertainty; no significance claim.",
                "Pooled union labels cover only retrieved candidates, not all catalog items.",
            ],
        },
        "summary": summary, "per_category": categories,
        "paired_bootstrap_v4_differences": ci, "per_reference": per_reference,
    }
    return merged, output


def report(metrics: dict) -> str:
    ss = metrics["summary"]
    cats = metrics["per_category"]
    p = lambda x: f"{x * 100:.1f}%"
    md = [
        "# ELAN Recommendation: V1–V4 Blind Holdout Evaluation",
        "",
        "## Experimental design",
        "",
        "- 18 new reference products (3 per category), 140 unique retrieved pairs; 90 Top-5 slots per model.",
        "- Pair relevance was judged from images and names before model rankings were unblinded.",
        "- Relevance: 0 = irrelevant, 1 = related, 2 = close substitute. Precision@5 uses relevance >= 1; Strict P@5 uses relevance == 2.",
        "- nDCG@5 uses gain = 2^relevance - 1, log2(rank+1) discount, and the ideal five labels selected from the pooled union for each query.",
        f"- For {len(metrics['method']['zero_ideal_reference_ids'])} queries with no pooled relevant candidate, nDCG is undefined and excluded from the main mean (n=16). Precision counts their five positions as irrelevant.",
        "",
        "## Aggregate results",
        "",
        "| Model | P@5 | Strict P@5 | nDCG@5 (defined queries) | Relevant slots | Strict slots |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, m in ss.items():
        md.append(f"| {model} | {p(m['precision_at_5'])} | {p(m['strict_precision_at_5'])} | {m['ndcg_at_5_mean_valid_queries']:.3f} | {m['relevant_recommendation_slots']}/90 | {m['strict_recommendation_slots']}/90 |")
    md += ["", "## Per-category Precision@5", "",
           "| Category | V1 | V2 | V3 | V4 |", "|---|---:|---:|---:|---:|"]
    for category, vals in cats.items():
        md.append(f"| {category} | " + " | ".join(p(vals[v]['precision_at_5']) for v in ("V1", "V2", "V3", "V4")) + " |")
    delta = metrics['paired_bootstrap_v4_differences']['V4_minus_V1']
    d = delta['precision_at_5']
    md += [
        "", "## Main findings", "",
        f"- V4 increases P@5 over V1 by {p(d['mean_difference'])} (four additional relevant slots per 90), but lowers mean nDCG@5 from {ss['V1']['ndcg_at_5_mean_valid_queries']:.3f} to {ss['V4']['ndcg_at_5_mean_valid_queries']:.3f}.",
        f"- Reference-level paired-bootstrap 95% percentile interval for V4–V1 P@5: {p(d['bootstrap_95_percentile_lower'])} to {p(d['bootstrap_95_percentile_upper'])}. This interval includes zero; the small exploratory sample does not establish an improvement.",
        "- V4 and V1 tie on Strict P@5 (22 close-substitute slots each).",
        "- V4 regresses on Tops (53.3% versus V1 73.3%) despite improvement on Bottoms and Outerwear.",
        f"- V2 and V3 returned identical five ranked IDs on all 18 references: {metrics['method']['v2_v3_identical_rankings_on_this_holdout']}.",
        "- MAGIC GLOVES and Alma Onesie have no relevant candidate in the pooled set; nDCG for these queries is undefined, not perfect.",
        "", "## Deployment recommendation", "",
        "**Keep V1 as production baseline for now.** Although V4 leads P@5 by a small margin, it loses nDCG@5, ties on strict substitutability, and has a substantial Tops regression. V1 is already the stable production route; no model switch is justified by these AI-labeled, small-sample results.",
        "V4 additionally depends on a private local H&M `articles.csv` that is not included in the production deployment. Do not point the existing API at V4 without solving metadata provisioning, test coverage, and rollback.",
        "", "## Scope and limitations", "",
        "- Labels were created by an AI evaluator and have not been independently verified by human raters. These metrics are exploratory portfolio evidence, not a verified consumer-quality benchmark.",
        "- Each category has only three references; sample uncertainty is substantial. Avoid tuning on this holdout if it will be reused as a final test.",
        "- Candidates not retrieved by any of the four models were not graded. nDCG is pooled-candidate nDCG, not catalog-exhaustive nDCG.",
        "- Data taxonomy/category mismatch can affect results (including Tops, Outerwear, Accessories).",
        "", "## Reproducibility", "",
        "Run `evaluate_holdout_v4_results.py` from `backend/evaluation/` with the frozen `holdout_v4_candidate_union.csv` and `holdout_v4_blind_labeled.csv` files. Verify 140 merged pairs and 90 slots per model.",
        "",
    ]
    return "\n".join(md)


def write_text(path: Path, data: str, overwrite: bool):
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite only if intended.")
    path.write_text(data, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    workdir = Path(__file__).resolve().parent
    parser.add_argument('--union', type=Path, default=workdir / 'holdout_v4_candidate_union.csv')
    parser.add_argument('--labels', type=Path, default=workdir / 'holdout_v4_blind_labeled.csv')
    parser.add_argument('--out-dir', type=Path, default=workdir)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    for src in (args.union, args.labels):
        if not src.is_file():
            raise FileNotFoundError(f"Missing required input: {src}")
    union_fields, union = read_rows(args.union)
    _, blind = read_rows(args.labels)
    merged, metrics = evaluate(union, blind)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = (
        args.out_dir / 'holdout_v4_candidate_union_labeled.csv',
        args.out_dir / 'holdout_v4_metrics.json',
        args.out_dir / 'holdout_v4_evaluation_report.md',
    )
    for path in targets:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists, nothing written: {path}")
    # CSV write: no model/DB changes.
    with targets[0].open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=union_fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(merged)
    write_text(targets[1], json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", args.overwrite)
    write_text(targets[2], report(metrics), args.overwrite)
    print('V1-V4 BLIND HOLDOUT EVALUATION: PASS')
    print('Paired labeled rows:', len(merged))
    print('Reference products:', metrics['method']['references'])
    print('Recommendation positions per model:', metrics['summary']['V1']['total_recommendation_slots'])
    for model, m in metrics['summary'].items():
        print(f"{model}: P@5={m['precision_at_5']:.1%} StrictP@5={m['strict_precision_at_5']:.1%} nDCG@5={m['ndcg_at_5_mean_valid_queries']:.3f}")
    for path in targets:
        print('Saved:', path)
    print('Models, database, production API: NOT MODIFIED')


if __name__ == '__main__':
    main()
