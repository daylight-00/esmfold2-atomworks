"""Do two ESMFold2 checkpoints hold the same weights?

Both are loaded through esm's own loader, which rewrites a checkpoint published
in the Hugging Face port's layout onto the native one (renaming, packing q/k/v),
and the two native state dicts are compared in fp32, tensor by tensor. No name
mapping is guessed here.

Parameters the native module allocates but reads nowhere -- upstream lists them
in ``EsmFold2Model._keys_to_ignore_on_load_missing``, and a checkpoint in the
port's layout does not carry them -- are reported separately, not as
differences: a side that does not carry them leaves them uninitialised.

The ESMC backbone is compared as well: taken from the checkpoint when it is
bundled, otherwise from the directory given for that side.

    python scripts/compare_checkpoints.py REFERENCE CANDIDATE \\
        [--reference-esmc DIR] [--candidate-esmc DIR] [--json OUT]

Exit status 0 when every compared tensor is identical. Needs roughly three times
the larger checkpoint's size in memory (fp32, on CPU). The record identifies
each side by Hub repo and revision, read off the directory layout, never by a
local path.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def identity(path: Path) -> dict[str, str]:
    """Repo id and revision of a checkpoint directory, where its layout says."""
    resolved = path.resolve()
    parts = resolved.parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if (
            index > 0
            and parts[index - 1].startswith("models--")
            and index + 1 < len(parts)
        ):
            repo = parts[index - 1][len("models--") :].replace("--", "/", 1)
            return {"repo": repo, "revision": parts[index + 1]}
    metadata = resolved / ".cache/huggingface/download/config.json.metadata"
    if metadata.is_file():
        return {"repo": "", "revision": metadata.read_text().splitlines()[0]}
    return {"repo": "", "revision": "", "directory": resolved.name}


def load_fold(path: Path) -> Any:
    from esm.models.esmfold2.model import EsmFold2Model

    return EsmFold2Model.from_pretrained(
        str(path), load_esmc=False, esmc_precision="fp32", device="cpu"
    )


def load_esmc(path: Path) -> Any:
    import torch
    from esm.models.esmc import EsmcModel

    return EsmcModel.from_pretrained(str(path), device="cpu", dtype=torch.float32)


def split(state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """(trunk, bundled backbone with its ``esmc.`` prefix removed)."""
    trunk = {k: v for k, v in state.items() if not k.startswith("esmc.")}
    esmc = {k[len("esmc.") :]: v for k, v in state.items() if k.startswith("esmc.")}
    return trunk, esmc


def compare(reference: dict, candidate: dict, unread: list[re.Pattern]) -> dict:
    import torch

    common = sorted(set(reference) & set(candidate))
    skipped = [k for k in common if any(p.search(k) for p in unread)]
    differ, identical = [], 0
    for key in common:
        if key in skipped:
            continue
        a, b = reference[key], candidate[key]
        if a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b):
            identical += 1
        else:
            differ.append(key)
    return {
        "compared": len(common) - len(skipped),
        "identical": identical,
        "differ": differ,
        "only_in_reference": sorted(set(reference) - set(candidate)),
        "only_in_candidate": sorted(set(candidate) - set(reference)),
        "allocated_but_unread": skipped,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--reference-esmc", type=Path)
    parser.add_argument("--candidate-esmc", type=Path)
    parser.add_argument("--json", type=Path, help="also write the record here")
    args = parser.parse_args(argv)

    import torch

    torch.set_grad_enabled(False)
    record: dict[str, Any] = {}
    sides = {}
    for name, path, esmc_dir in (
        ("reference", args.reference, args.reference_esmc),
        ("candidate", args.candidate, args.candidate_esmc),
    ):
        model = load_fold(path)
        trunk, esmc = split(model.state_dict())
        unread = [re.compile(p) for p in type(model)._keys_to_ignore_on_load_missing]
        if esmc:
            esmc_source: dict[str, str] | str = "bundled"
        elif esmc_dir is not None:
            esmc = load_esmc(esmc_dir).state_dict()
            esmc_source = identity(esmc_dir)
        else:
            parser.error(
                f"the {name} checkpoint bundles no backbone; pass --{name}-esmc"
            )
        record[name] = {**identity(path), "esmc": esmc_source}
        sides[name] = (trunk, esmc, unread)

    (ref_trunk, ref_esmc, unread), (cand_trunk, cand_esmc, _) = (
        sides["reference"],
        sides["candidate"],
    )
    record["trunk"] = compare(ref_trunk, cand_trunk, unread)
    record["esmc"] = compare(ref_esmc, cand_esmc, [])
    record["same_weights"] = all(
        not record[part][key]
        for part in ("trunk", "esmc")
        for key in ("differ", "only_in_reference", "only_in_candidate")
    )

    text = json.dumps(record, indent=2) + "\n"
    if args.json is not None:
        args.json.write_text(text)
    sys.stdout.write(text)
    return 0 if record["same_weights"] else 1


if __name__ == "__main__":
    sys.exit(main())
