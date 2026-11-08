# ELAN Recommendation: V1–V4 Blind Holdout Evaluation

## Experimental design

- 18 new reference products (3 per category), 140 unique retrieved pairs; 90 Top-5 slots per model.
- Pair relevance was judged from images and names before model rankings were unblinded.
- Relevance: 0 = irrelevant, 1 = related, 2 = close substitute. Precision@5 uses relevance >= 1; Strict P@5 uses relevance == 2.
- nDCG@5 uses gain = 2^relevance - 1, log2(rank+1) discount, and the ideal five labels selected from the pooled union for each query.
- For 2 queries with no pooled relevant candidate, nDCG is undefined and excluded from the main mean (n=16). Precision counts their five positions as irrelevant.

## Aggregate results

| Model | P@5 | Strict P@5 | nDCG@5 (defined queries) | Relevant slots | Strict slots |
|---|---:|---:|---:|---:|---:|
| V1 | 66.7% | 24.4% | 0.784 | 60/90 | 22/90 |
| V2 | 70.0% | 22.2% | 0.741 | 63/90 | 20/90 |
| V3 | 70.0% | 22.2% | 0.741 | 63/90 | 20/90 |
| V4 | 71.1% | 24.4% | 0.759 | 64/90 | 22/90 |

## Per-category Precision@5

| Category | V1 | V2 | V3 | V4 |
|---|---:|---:|---:|---:|
| Accessories | 26.7% | 26.7% | 26.7% | 26.7% |
| Bottoms | 40.0% | 46.7% | 46.7% | 66.7% |
| Dresses | 100.0% | 100.0% | 100.0% | 100.0% |
| Knitwear | 80.0% | 80.0% | 80.0% | 80.0% |
| Outerwear | 80.0% | 86.7% | 86.7% | 100.0% |
| Tops | 73.3% | 80.0% | 80.0% | 53.3% |

## Main findings

- V4 increases P@5 over V1 by 4.4% (four additional relevant slots per 90), but lowers mean nDCG@5 from 0.784 to 0.759.
- Reference-level paired-bootstrap 95% percentile interval for V4–V1 P@5: -7.8% to 15.6%. This interval includes zero; the small exploratory sample does not establish an improvement.
- V4 and V1 tie on Strict P@5 (22 close-substitute slots each).
- V4 regresses on Tops (53.3% versus V1 73.3%) despite improvement on Bottoms and Outerwear.
- V2 and V3 returned identical five ranked IDs on all 18 references: True.
- MAGIC GLOVES and Alma Onesie have no relevant candidate in the pooled set; nDCG for these queries is undefined, not perfect.

## Deployment recommendation

**Keep V1 as production baseline for now.** Although V4 leads P@5 by a small margin, it loses nDCG@5, ties on strict substitutability, and has a substantial Tops regression. V1 is already the stable production route; no model switch is justified by these AI-labeled, small-sample results.
V4 additionally depends on a private local H&M `articles.csv` that is not included in the production deployment. Do not point the existing API at V4 without solving metadata provisioning, test coverage, and rollback.

## Scope and limitations

- Labels were created by an AI evaluator and have not been independently verified by human raters. These metrics are exploratory portfolio evidence, not a verified consumer-quality benchmark.
- Each category has only three references; sample uncertainty is substantial. Avoid tuning on this holdout if it will be reused as a final test.
- Candidates not retrieved by any of the four models were not graded. nDCG is pooled-candidate nDCG, not catalog-exhaustive nDCG.
- Data taxonomy/category mismatch can affect results (including Tops, Outerwear, Accessories).

## Reproducibility

Run `evaluate_holdout_v4_results.py` from `backend/evaluation/` with the frozen `holdout_v4_candidate_union.csv` and `holdout_v4_blind_labeled.csv` files. Verify 140 merged pairs and 90 slots per model.
