"""Verify the Hydra configs survive into an installed wheel and compose.

Run against an *installed* package, not a source checkout::

    uv build --wheel -o dist
    uv venv /tmp/env --python 3.14
    uv pip install --python /tmp/env/bin/python dist/*.whl hydra-core
    /tmp/env/bin/python scripts/check_packaging.py

The failure this exists to catch is invisible from a source tree: the configs
sit at the repo root, `pkg://esmfold2_foundry.configs` expects them inside the
package, and nothing in a normal test run notices that the wheel shipped
without them.

It also composes rather than merely listing files, because two of the defects
found when this was first written -- a searchpath that could not resolve, and a
`_target_` landing under a group namespace instead of the root -- are visible
only once Hydra actually assembles the config.
"""

from __future__ import annotations

import sys
from pathlib import Path

EXPECTED_TARGETS = {
    "model/esmfold2.yaml": "esmfold2_foundry.model.esmfold2.FoundryESMFold2",
    "data/atomworks.yaml": "esmfold2_foundry.data.pipelines.build_esmfold2_pipeline",
    "inference_engine/esmfold2.yaml": (
        "esmfold2_foundry.inference.engine.ESMFold2InferenceEngine"
    ),
}


def main(require_installed: bool = True) -> int:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from esmfold2_foundry import paths

    package_dir = Path(paths.__file__).resolve().parent
    if require_installed and "site-packages" not in str(package_dir):
        print(
            f"FAIL: running from a source checkout ({package_dir}); this check "
            "only means something against an installed wheel",
            file=sys.stderr,
        )
        return 2

    config_dir = paths.config_dir()
    print(f"package:    {package_dir}")
    print(f"configs:    {config_dir}")
    if require_installed and config_dir != package_dir / "configs":
        print(
            f"FAIL: configs resolved outside the package: {config_dir}", file=sys.stderr
        )
        return 1

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="inference", overrides=["inputs=x", "out_dir=y"])

    target = cfg.get("_target_")
    want = EXPECTED_TARGETS["inference_engine/esmfold2.yaml"]
    if target != want:
        print(
            f"FAIL: composed inference config has _target_={target!r}, expected "
            f"{want!r}. A group file missing `# @package _global_` puts it under "
            "the group namespace, where instantiate(cfg) will not find it.",
            file=sys.stderr,
        )
        return 1
    print(f"composed:   inference -> {target}")

    # Sampling defaults follow the shipped checkpoint, not the dataclass
    # defaults; a silent change here would alter every downstream run.
    for key, expected in (("num_loops", 3), ("num_sampling_steps", 100)):
        if cfg[key] != expected:
            print(f"FAIL: {key} is {cfg[key]}, expected {expected}", file=sys.stderr)
            return 1

    for rel, expected in EXPECTED_TARGETS.items():
        path = config_dir / rel
        if not path.exists():
            print(f"FAIL: {rel} is not in the package", file=sys.stderr)
            return 1
        node = OmegaConf.load(path)
        if node.get("_target_") != expected:
            print(
                f"FAIL: {rel} has _target_={node.get('_target_')!r}, expected "
                f"{expected!r}",
                file=sys.stderr,
            )
            return 1
        print(f"target:     {rel} -> {expected}")

    print("OK: packaged configs compose from an installed wheel")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(require_installed="--allow-source" not in sys.argv))
