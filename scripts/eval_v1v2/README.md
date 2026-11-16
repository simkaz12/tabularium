# v1 vs v2 eval bench

An offline bench that compares the Python engine (v1) with the TypeScript server (v2) on the same inputs. It runs on throwaway local stores and never touches a deployed store.

## What it measures

Save side, per system: for each labelled pair of records (an existing one and an incoming one), does the system catch the duplicate, flag it, or write it as new? The buckets are `caught`, `flagged`, `new` and `wrong_target`. The report also counts records wrongly merged away, false alarms on distinct pairs, replacements flagged, and the time of each save call. Each pair runs against a store that already holds 1,000 background records, so a pair is not found just because nothing else is there.

Search side, per arm (v1, v2 with rerank, v2 without rerank, plus fact-only variants): ranking quality against judged relevance for a set of real questions, and the time of each recall call.

The answer key comes from two LLM judges (the `claude` and `codex` CLIs) grading the same items blind to which engine produced them. A label enters the key only when both agree. Pairs they disagree on open in an HTML page for you to settle. The report states the limits of the measurement next to the numbers.

## What you bring

No real data ships with this repository. Set `TABULARIUM_EVAL_DATA` to a directory of your own (default: `dev-corpus/` in the repo, which is gitignored). Inputs, all optional except the pool:

| File | Content |
| --- | --- |
| `burial/pool-full.json` | Snapshot of a v1 store: a list of records with `id`, `type`, `content`, `scope`, `lifecycle`, `validity`, `vector` (768 dims), `recall_count`, `created_at`. Used as the search corpus and as noise for the save side. |
| `bounce-log.json` | Optional. A log of dedup-gate bounces to turn into unlabelled pairs. |
| `ledger-export.json` | Optional. Used to resolve bounce rows back to full text. |

Labelled pairs come from `tests/fixtures/jev_pairs_sample.json` (a small synthetic sample in the schema the bench reads), `tests/cases/admission/` and the golden pairs in `scripts/freeze_vectors.py`. Replace the sample with your own pairs in the same schema to get a meaningful save-side result.

Search questions are mined from recall tool calls in agent session transcripts (JSON lines). Point `TABULARIUM_EVAL_TRANSCRIPTS` at the transcript directory and `TABULARIUM_EVAL_PROJECTS` at the directory that holds your project checkouts, or pass `--transcripts` and `--projects` to `eval_v1v2.questions`. The folder-to-scope map `PATH_SCOPES` lives in `questions.py`; edit it for your layout. `EXCLUDED_SCOPES` in `corpus.py` and `pairs.py` lists scopes whose text must never be sent to the judge CLIs. It ships with placeholder names, so set it to yours before a real run.

## Requirements

- Ollama on `localhost:11434` with `nomic-embed-text`.
- The `claude` and `codex` CLIs on `PATH`, logged in.
- `TYPESAFE_API_KEY` in the environment for the Jev assessor both systems use.
- Node 22+ and `npm ci` in `v2/`.

## Run

```
PYTHONPATH=scripts uv run python -m eval_v1v2.run --limit 10
PYTHONPATH=scripts uv run python -m eval_v1v2.run
PYTHONPATH=scripts uv run python -m eval_v1v2.run --stage report
```

`--limit` runs a small smoke pass first. `--stage` runs one stage; the stages are `pairs`, `judge-relation`, `disputes`, `save-v1`, `save-v2`, `judge-targets`, `score-save`, `corpus`, `questions`, `search-v1`, `search-v2`, `judge-relevance`, `score-search` and `report`. Stages write JSON lines and the final `report.md` and `report.html` under `$TABULARIUM_EVAL_DATA/evals/v1v2/`, and finished stages are skipped unless you pass `--force`.

`prod_check.py` compares a recorded replay of real saves (a directory of rows you produce, `experiments/prod-replay` by default, no URL involved) with the bench result:

```
PYTHONPATH=scripts uv run python -m eval_v1v2.prod_check --replay path/to/replay
```

## Tests

The bench has its own tests (`tests/test_eval_v1v2_*.py`, `tests/test_eval_pmv2_*.py`, `v2/eval/*.test.ts`). They use small synthetic fixtures and run in the default `uv run pytest` and `npm test` lanes.

## PersonaMem-v2 lane

`pmv2/` is a second lane built on a public benchmark, PersonaMem-v2, fetched from Hugging Face on first use. `prep.py` downloads five personas and writes questions and chat chunks. `extract.py` has the `claude` CLI pull notes out of each chunk. `v1_lane.py` and `v2/eval/pmv2.ts` save those notes through each engine. `read.py` answers the benchmark's multiple-choice questions from what each engine recalls, and `report.py` writes the report. Each script is run as `PYTHONPATH=scripts uv run python -m eval_v1v2.pmv2.<name>`.
