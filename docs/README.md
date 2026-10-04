# Design record

Read in order. Decisions carry IDs (`D-001` …); code and tests cite some of them.

| doc | what it settles |
|---|---|
| [00_SCOPE.md](00_SCOPE.md) | what this project is, and the eight decisions |
| [01_ADAPTER.md](01_ADAPTER.md) | `AtomArray -> StructurePredictionInput`, and what it refuses |
| [02_PARITY.md](02_PARITY.md) | the milestone: method, result, and what it does not cover |
| [03_MODEL.md](03_MODEL.md) | the model wrapper, the AtomWorks pipeline, the engine, the per-call record, and how to read the confidence |
| [04_ENVIRONMENT.md](04_ENVIRONMENT.md) | installing, verified versions, the ESMFold2 packagings, checks, environment variables, gotchas |
| [05_ROADMAP.md](05_ROADMAP.md) | status, what an upstream limits, and what remains |
| [06_FOUNDRY_INTEGRATION.md](06_FOUNDRY_INTEGRATION.md) | the optional Foundry integration: compatibility contracts and training |

The pinned environment the published results were produced in is documented
beside the record rather than in it, under
[`reproducibility/`](../reproducibility/README.md).

The record is authoritative. If the code and a doc disagree, that is a bug in
one of them, not a matter of taste.
