# legacy-store-294fa14

A LanceDB store written entirely through production `Tabularium.save` / `bury` / `revoke` at
revision `294fa14` (pre-P1 schema): 10 Facts (2 whitespace-padded, 2 supersession edges), 3
Decisions (one superseding another, one revoked), 1 buried Fact. 12 rows in `records`, 1 row in
`tombstones`.

Rebuilt with: `tests/fixtures/build_legacy_store.py` — see that script's module docstring for the
exact worktree + command sequence.
