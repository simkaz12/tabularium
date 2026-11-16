# Admission case set

One YAML file per case, scoring engine admission-mutation behaviour end to end. Every later piece
on the admission-mutation map (P3–P9) adds its own case rows here — none are written by P2.

## Shape

`id`, `class`, `origin` (`store` | `authored`), `existing[]` (`alias`, `text`, `kind`, `scope`,
`lifecycle`, `validity`, `created_at`), `incoming` (the save call as written), `expect`
(`selection`, `material`, `mutation.acceptable`/`mutation.ideal`), `why`. Checks come from a fixed
vocabulary only — never inline code. `origin: store` marks text taken from a real store; every case in this repository is `authored`.

## Falsity block (A3 cleaner-integration piece, scope 5)

A case may carry `falsity` instead of `expect` — never both, never neither:

```yaml
falsity:
  judge_a: {verdict: replace, reason: correction, why: "..."}
  judge_b: {verdict: coexist, why: "..."}
  assessor: correction        # coexist | correction | changed | unavailable, omit for no call
  expect: supersede:correction  # coexist | contested | supersede:correction | supersede:changed | global_reported
```

`existing[]` names exactly one row — the pair's OLDER record; `incoming` is the newer one, same
field every material/mutation case already uses. Every falsity case shares one store and one
scope, seeded once through production `save`, then scored from a single `clean(scope)` pass's
`CleanSummary.falsity` by the older record's id. Write each pair's texts on an unrelated subject —
the runner probes each case's own newcomer back to the store and asserts the best hit is that
case's own older record; a fixture whose text drifts close enough to another case's pair fails
there, and the fix is the fixture text, never the scoring.

## Freeze rule

`scripts/freeze_vectors.py` reads every case file's texts; add a case, then run
`uv run python scripts/freeze_vectors.py` before the default lane can embed it hermetically.
