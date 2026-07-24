# Repository Legibility

`AGENTS.md` is the short map. Durable behavior and operating knowledge belong
in the focused files under `docs/`.

When changing the repository:

1. Read `README.md`, `docs/README.md`, and the topic document that owns the
   behavior.
2. Read `DESIGN_DECISIONS.md` before changing scientific defaults.
3. Confirm documentation against code and configs rather than relying on issue
   or chat history.
4. Add validation or a test when prose describes an enforceable invariant.
5. Update the owning document in the same change when launch behavior, config
   semantics, artifacts, or interpretation changes.

Generated runs, datasets, tokenizer outputs, checkpoints, and reports stay
outside Git. Small, intentional test fixtures are the exception and should be
documented.

Historical observations without public evidence must be labeled preliminary.
Do not add personal filesystem paths, hostnames, credentials, private artifact
identifiers, or institution-specific scheduler commands to tracked files.
