# 0003 — Provider layer: litellm underneath, Pydantic AI on top

> **Frozen 2026-08-07.** Historical prior — it records what was decided and why at the time, is
> not maintained, and diverges from the built system in places. There is no present-truth
> document: the code is what is true now, and new architectural decisions are recorded in the
> Tabularium store, not as new ADR files. `README.md` orients.

**Status:** accepted

Tabularium's LLM + embeddings layer is **litellm** for provider routing, ordered fallback (local→cloud), retry/backoff, embeddings (incl. local Ollama), and per-call cost/token/latency telemetry callbacks — with **Pydantic AI** layered on top for structured-output validation (`output_validator` + `ModelRetry`) and any future agent-loop orchestration. **No LangChain / LlamaIndex** (they would wrap the very control flow — routing, retry, telemetry, output parsing — Tabularium must own to back its claims). **No PyTorch / sentence-transformers** — Ollama embeddings are first-party in litellm, so local embeddings need no torch dependency.

## Considered options

- **litellm alone** — covers all five requirements (Ollama embeddings, local→cloud routing + fallback, retry/backoff, per-call telemetry callbacks, cross-provider structured output) first-party. Chosen as the base layer.
- **Pydantic AI alone** — NOT a substitute: it's an agent-loop + structured-output framework; does completion fallback but **not embedding routing**, and its telemetry is OpenTelemetry spans, not per-call Python callbacks. A different slot — layered on top, not instead.
- **Direct provider SDKs + thin DIY router** — ~500 lines of hand-maintained routing/retry/telemetry; rejected as reinventing what litellm ships, on the exact layer the project wants demonstrably built "for real."

## Consequences

- litellm is large and fast-moving → pin the version.
- Tabularium writes only: the escalation/degradation *policy*, the telemetry *sink*, and a **retry-on-malformed-JSON** loop (litellm won't retry an HTTP-200 with a bad JSON body).
- litellm's `CustomLogger` (fires on completions AND embeddings) is the seam for the observability sink — closes project-b's §7 zero-observability-on-AI-calls weak spot.
- Resolves T11 (framework posture). Feeds T06 (recall graceful degradation) and the telemetry fog.
