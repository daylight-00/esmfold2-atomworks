"""Console entry point: ``esmfold2-foundry <command>``.

Follows Foundry's ``models/<model>/src/<model>/cli.py`` pattern -- a small typer
app whose commands import their heavy dependencies lazily, so that ``--help`` and
argument errors do not pay for torch.
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    add_completion=False,
    help="AtomWorks-compatible, Foundry-trainable ESMFold2.",
    no_args_is_help=True,
)


@app.command()
def doctor() -> None:
    """Check that every source tree, weight directory and import resolves."""
    from esmfold2_foundry.doctor import main

    raise typer.Exit(code=main())


@app.command()
def parity(
    structures: list[Path] = typer.Argument(..., help="Structures to check."),
    seed: int = typer.Option(0, help="Seed for conformer generation."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Check that the adapter featurizes a structure like a hand-written input.

    Runs on CPU and needs no weights. This is the Phase 1 milestone check.
    """
    from esmfold2_foundry.parity.run import run_feature_parity

    failures = run_feature_parity(structures, seed=seed, verbose=verbose)
    raise typer.Exit(code=1 if failures else 0)


@app.command()
def fold(
    structures: list[Path] = typer.Argument(..., help="Structures to fold."),
    out_dir: Path = typer.Option(Path("runs/fold"), help="Where to write results."),
    weights: str | None = typer.Option(None, help="Weight directory or HF repo id."),
    num_loops: int = typer.Option(3, help="Trunk recycle count."),
    num_sampling_steps: int = typer.Option(100, help="Diffusion steps."),
    num_diffusion_samples: int = typer.Option(1),
    seed: int | None = typer.Option(None),
    device: str | None = typer.Option(None, help="cuda, cuda:1, cpu..."),
    allow: list[str] = typer.Option(
        [],
        "--allow",
        help=(
            "Accept one named degradation instead of raising; repeatable. "
            "One of: unsupported_chains, unresolved_covalent_bonds, "
            "inferred_chain_kind, unplaceable_modifications."
        ),
    ),
) -> None:
    """Fold structures through the AtomWorks adapter and write CIF + metrics.

    Strict by default: anything that would fold a different molecule than the
    one described raises, and ``--allow`` accepts it by name. Whatever was
    accepted is recorded per structure under ``adapter.degradations`` in the
    JSON written beside each CIF.
    """
    from esmfold2_foundry.inference.engine import ESMFold2InferenceEngine

    engine = ESMFold2InferenceEngine(
        weights,
        device=device,
        num_loops=num_loops,
        num_sampling_steps=num_sampling_steps,
        num_diffusion_samples=num_diffusion_samples,
        seed=seed,
        verbose=True,
        allow=allow,
    )
    outputs = engine.run([str(p) for p in structures], out_dir=out_dir)
    for output in outputs:
        plddt = output.metadata.get("esm.mean_plddt")
        shown = f"{plddt:.3f}" if plddt is not None else "n/a"
        typer.echo(f"{output.example_id}: mean pLDDT {shown} (0-1 scale)")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
