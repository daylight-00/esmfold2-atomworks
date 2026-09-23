# Design record

Read in order. Decisions carry IDs (`D-001` …) and the code cites them.

| doc | what it settles |
|---|---|
| [00_SCOPE.md](00_SCOPE.md) | what this project is, and the eight decisions |
| [01_ADAPTER.md](01_ADAPTER.md) | `AtomArray -> StructurePredictionInput`, and what it refuses |
| [02_PARITY.md](02_PARITY.md) | the milestone: method, result, and what it does not cover |
| [03_FOUNDRY_INTEGRATION.md](03_FOUNDRY_INTEGRATION.md) | how this drops into `foundry/models/`, and upstream gotchas |
| [04_ENVIRONMENT.md](04_ENVIRONMENT.md) | source trees, venv, compute |
| [05_ROADMAP.md](05_ROADMAP.md) | Phase 3 — the generative surgery |

The record is authoritative. If the code and a doc disagree, that is a bug in
one of them, not a matter of taste.
