"""Inference engine: one resident ESMFold2, many AtomWorks structures.

``initialize`` loads the model once; ``run`` (also ``__call__``) folds a path, a
list of paths, an ``AtomArray`` or a mapping of them, and writes CIF plus
metrics when given an ``out_dir``. The surface matches Foundry's
``BaseInferenceEngine``, so call sites read the same, but nothing here imports
Foundry; why it does not subclass it is in docs/03.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from esmfold2_atomworks.model.esmfold2 import AtomWorksESMFold2, FoldingConfig

if TYPE_CHECKING:
    from biotite.structure import AtomArray

__all__ = ["ESMFold2InferenceEngine", "ESMFold2Output"]


@dataclass
class ESMFold2Output:
    """One prediction, in AtomWorks terms plus the model's own confidence.

    Mirrors ``RFD3Output`` / ``RF3Output``: an ``AtomArray``, a metadata dict
    and an ``example_id``, with a ``dump`` that writes the pair.
    """

    atom_array: AtomArray
    metadata: dict[str, Any] = field(default_factory=dict)
    example_id: str = "pred"

    def dump(self, out_dir: str | Path, verbose: bool = True) -> Path:
        """Write ``<example_id>.cif`` and ``<example_id>.json``."""
        import json

        from atomworks.io.utils.io_utils import to_cif_file

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        structure_path = out_dir / f"{self.example_id}.cif"
        to_cif_file(self.atom_array, structure_path)
        (out_dir / f"{self.example_id}.json").write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True, default=float)
        )
        if verbose:
            print(f"wrote {structure_path}")
        return structure_path


class ESMFold2InferenceEngine:
    """Keeps one ESMFold2 resident and folds AtomWorks structures with it."""

    def __init__(
        self,
        ckpt_path: str | Path | None = None,
        *,
        device: Any | None = None,
        num_loops: int | None = None,
        num_sampling_steps: int = 100,
        num_diffusion_samples: int = 1,
        lm_dropout: float | None = 0.3,
        lm_mask_pct: float | None = None,
        seed: int | None = None,
        chunk_size: int | None = 64,
        load_esmc: bool = True,
        verbose: bool = False,
        allow: Any = (),
    ) -> None:
        self.ckpt_path = ckpt_path
        self.device = device
        self.load_esmc = load_esmc
        self.chunk_size = chunk_size
        self.verbose = verbose
        from esmfold2_atomworks.data.atomworks_to_esm import allow_kwargs

        # Validated at construction: a misspelt name must fail before the model
        # loads, not after minutes of weight download.
        self.adapter_policy = allow_kwargs(allow)
        self.folding = FoldingConfig(
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            num_diffusion_samples=num_diffusion_samples,
            lm_dropout=lm_dropout,
            lm_mask_pct=lm_mask_pct,
            seed=seed,
        )
        self._model: AtomWorksESMFold2 | None = None

    # -- lifecycle ---------------------------------------------------------

    def initialize(self) -> Self:
        """Load the model. Idempotent, so ``run`` can call it unconditionally."""
        if self._model is None:
            self._model = AtomWorksESMFold2(
                self.ckpt_path,
                device=self.device,
                load_esmc=self.load_esmc,
                chunk_size=self.chunk_size,
            )
            if self.verbose:
                print(self._model.provenance())
        return self

    @property
    def model(self) -> AtomWorksESMFold2:
        if self._model is None:
            self.initialize()
        assert self._model is not None
        return self._model

    def __enter__(self) -> Self:
        return self.initialize()

    def __exit__(self, *exc: object) -> None:
        self._model = None

    # -- the work ----------------------------------------------------------

    def run(
        self,
        inputs: Any,
        *,
        out_dir: str | Path | None = None,
        **overrides: Any,
    ) -> list[ESMFold2Output]:
        """Fold one or many inputs.

        Args:
            inputs: a path to a structure, a list of paths, an ``AtomArray``, a
                list of them, or a mapping of ``example_id -> input``.
            out_dir: when given, each output is also written there.

        Returns:
            One :class:`ESMFold2Output` per input, in input order.
        """
        from esmfold2_atomworks.data.atomworks_to_esm import AdapterReport
        from esmfold2_atomworks.metrics import fold_metrics

        self.initialize()
        named = _canonicalize_inputs(inputs)

        outputs: list[ESMFold2Output] = []
        for example_id, item in named.items():
            atoms, chain_info = _load(item)
            report = AdapterReport()
            structure, result = self.model.fold_atom_array(
                atoms,
                chain_info=chain_info,
                config=self.folding,
                adapter_kwargs={**self.adapter_policy, "report": report},
                **overrides,
            )
            accepted = _accepted_degradations(report)
            if isinstance(structure, list):
                # num_diffusion_samples > 1: emit one output per sample.
                for index, (one, res) in enumerate(zip(structure, result, strict=True)):
                    outputs.append(
                        ESMFold2Output(
                            atom_array=one,
                            metadata=fold_metrics(res) | {"sample": index} | accepted,
                            example_id=f"{example_id}_{index}",
                        )
                    )
            else:
                outputs.append(
                    ESMFold2Output(
                        atom_array=structure,
                        metadata=fold_metrics(result) | accepted,
                        example_id=example_id,
                    )
                )

        if out_dir is not None:
            for output in outputs:
                output.dump(out_dir, verbose=self.verbose)
        return outputs

    forward = run
    __call__ = run


def _accepted_degradations(report: Any) -> dict[str, Any]:
    """What the caller opted into *and what actually happened*, for the output.

    Opting in says a degradation is acceptable; it does not say which structure
    it happened to. Recording it per output is what makes the permissive run
    auditable afterwards -- the JSON beside each CIF says what was dropped or
    approximated for that one, keyed by the same names the caller opted in
    with. Empty when nothing was.
    """
    accepted = report.accepted_degradations()
    return {"adapter.degradations": accepted} if accepted else {}


def _canonicalize_inputs(inputs: Any) -> dict[str, Any]:
    """Normalize the accepted input shapes to ``example_id -> item``."""
    if isinstance(inputs, dict):
        return {str(k): v for k, v in inputs.items()}
    if isinstance(inputs, (str, Path)):
        return {Path(inputs).name.split(".")[0]: inputs}
    if isinstance(inputs, (list, tuple)):
        named: dict[str, Any] = {}
        for index, item in enumerate(inputs):
            key = (
                Path(item).name.split(".")[0]
                if isinstance(item, (str, Path))
                else f"input_{index}"
            )
            named[key if key not in named else f"{key}_{index}"] = item
        return named
    return {"pred": inputs}


def _load(item: Any) -> tuple[Any, dict | None]:
    """``(atom_array, chain_info)`` for a path or an already-parsed structure."""
    if isinstance(item, (str, Path)):
        from atomworks.io import parse

        parsed = parse(item)
        return parsed["asym_unit"][0], parsed["chain_info"]
    if isinstance(item, tuple) and len(item) == 2:
        return item
    return item, None
