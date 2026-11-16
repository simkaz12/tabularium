"""Agent-facing text for the MCP server: the server instructions and each tool and resource
description. These strings are the wire contract."""

RECALL_DESCRIPTION = (
    "Retrieve relevant memories by hybrid search (vector + BM25). "
    "Returns hits ranked by relevance and importance. "
    "Set ``include_invalid=True`` to include invalidated records; "
    "``include_archived=True`` to include archived (historical) records."
)

DECISION_DESCRIPTION = (
    "Return the full view of a single Decision by id. "
    "Includes live stats (recall_count, last_referenced_at), "
    "lineage (supersedes / superseded_by / tip), invalidation cause and "
    "class (``invalidated_by`` / ``invalidation_reason``, both null when "
    "valid), the list of past ``revalidations``, and the byte-stable "
    "rendered text (identical to what ``decision://{id}`` serves)."
)

DECISIONS_QUERY_DESCRIPTION = (
    "Search in-force Decisions by hybrid retrieval. "
    "Returns lean triage links (uri / title / lifecycle / validity / scope). "
    "Set ``include_superseded=True`` to widen to superseded and revoked "
    "Decisions. Set ``include_invalid=True`` to widen to invalidated "
    "Decisions. Follow up with the ``decision`` tool to read the full view."
)

CONTEXT_DESCRIPTION = (
    "Compose the session-start grounding payload: two packed tiers "
    "(``hot`` Facts, then ``ember``) sized to a token ``budget``. "
    "Decisions are not included — reach them on demand via "
    "``decisions_query`` / ``decision``. A registered-but-empty scope "
    "returns both tiers present and empty, not an error. ``role`` IS "
    "REQUIRED — omitting it (or passing null) is a ``ToolError`` "
    "naming every registered role; there is no default profile to "
    "fall back to (``\"default\"`` is retired — curation block task "
    "07: an unspecified audience is a caller bug, not a state this "
    "tool resolves silently, since omitting it used to return an "
    "empty payload with no error at all). ``role`` is validated "
    "against the role registry AND filters both tiers (routing, not "
    "boosting, explicit enumeration only): a candidate reaches this "
    "call's payload only when ``role`` is named in its "
    "``audience_hint``. An empty or NULL hint reaches NO role. "
    "``for everyone`` is spelled "
    "by naming all four roles (``decider``/``builder``/``reviewer``/"
    "``navigator``), never by omission. This never affects "
    "``recall`` — audience routing stays push-only."
)

DECISION_RESOURCE_DESCRIPTION = (
    "Byte-stable render of a Decision. "
    "Bytes are identical to DecisionView.rendered for the same id. "
    "Bytes change only when the Decision has a semantic transition "
    "(lifecycle change, validity flip, supersession)."
)

SERVER_INSTRUCTIONS = (
    "Read-only archive of pre-v2 memory. This server accepts no writes. "
    "``recall`` retrieves memories and ``decision`` / ``decisions_query`` navigate architectural "
    "decisions; ``context`` returns the session-start payload. "
    "Propose new facts and decisions through T3 Code's ``memory_propose``."
)
