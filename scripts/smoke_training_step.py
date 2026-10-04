"""Run training steps through the Foundry integration, on one GPU.

The contract tests check that ``ESMFold2Trainer`` fits Foundry's trainer; this runs
it. An AtomWorks structure goes through the pipeline (features and labels), the
experimental checkpoint is built through the trainer's own ``construct_model``,
and Foundry's ``fit`` loop takes the steps. The objective is a distogram
cross-entropy on the trunk against the structure's own coordinates -- the one
output of the experimental forward that carries gradients; the diffusion sampler
runs without autograd. The bin range is this script's choice, not the
checkpoint's.

    python scripts/smoke_training_step.py --structure tests/data/structures/1a8o.cif

Exit status 0 when every step produced a finite loss and gradients, the loss on
this one structure fell between the first step and the last, and the trunk's
parameters changed.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument(
        "--weights", default=None, help="experimental checkpoint (default: the mirror)"
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-loops", type=int, default=1)
    parser.add_argument("--num-sampling-steps", type=int, default=2)
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--json", type=Path, help="write the run's record here")
    args = parser.parse_args(argv)

    import torch
    import torch.nn.functional as F
    from omegaconf import OmegaConf

    from esmfold2_atomworks import paths
    from esmfold2_atomworks.data.loading import parse_structure
    from esmfold2_atomworks.data.pipelines import build_esmfold2_pipeline
    from esmfold2_atomworks.training.trainer import make_trainer_class

    if not torch.cuda.is_available():
        print("needs a GPU", file=sys.stderr)
        return 2
    weights = args.weights or str(paths.ESMFOLD2_WEIGHTS.experimental)

    atoms, chain_info = parse_structure(args.structure)
    example = build_esmfold2_pipeline(is_inference=False, attach_labels=True)(
        {
            "atom_array": atoms,
            "chain_info": chain_info,
            "example_id": args.structure.stem,
        }
    )
    feats = example["feats"]
    # The experimental forward gates autograd on a soft sequence.
    feats["res_type_soft"] = F.one_hot(feats["res_type"].long(), 33).float()
    feats["num_loops"] = args.num_loops
    feats["num_sampling_steps"] = args.num_sampling_steps

    base = make_trainer_class()

    class SmokeTrainer(base):
        def compute_loss(self, output, example):
            logits = output["distogram_logits"][0].float()
            bins = logits.shape[-1]
            device = logits.device
            labels = example["labels"]
            coords = torch.as_tensor(labels["atom_coords"], device=device)
            present = torch.as_tensor(labels["atom_mask"], device=device)
            representative = example["feats"]["distogram_atom_idx"].to(device).long()
            tokens = example["feats"]["token_attention_mask"].to(device).bool()
            usable = present[representative] & tokens
            positions = coords[representative]
            edges = torch.linspace(2.0, 52.0, bins - 1, device=device)
            target = (torch.cdist(positions, positions).unsqueeze(-1) > edges).sum(-1)
            pairs = usable[:, None] & usable[None, :]
            self.n_pairs = int(pairs.sum())
            return F.cross_entropy(logits[pairs], target[pairs].clamp(max=bins - 1))

    class Probe:
        """Records what each step did, from the callbacks Foundry fires."""

        def __init__(self) -> None:
            self.losses: list[float] = []
            self.grad_norms: list[float] = []

        def on_train_batch_end(self, trainer, outputs, batch, batch_idx):
            self.losses.append(float(outputs["loss"]))
            grads = [
                p.grad.norm()
                for p in trainer.state["model"].parameters()
                if p.grad is not None
            ]
            self.grad_norms.append(float(torch.stack(grads).norm()) if grads else 0.0)

    probe = Probe()
    with tempfile.TemporaryDirectory() as scratch:
        trainer = SmokeTrainer(
            accelerator="cuda",
            devices_per_node=1,
            strategy="auto",
            precision=args.precision,
            max_epochs=1,
            limit_train_batches=args.steps,
            n_examples_per_epoch=args.steps,
            output_dir=scratch,
            checkpoint_every_n_epochs=1000,
            callbacks=[probe],
        )
        trainer.initialize_or_update_trainer_state(
            {
                "train_cfg": OmegaConf.create(
                    {
                        "model": {
                            "net": {
                                "_target_": "esmfold2_atomworks.model.esmfold2.AtomWorksESMFold2",
                                "weights": weights,
                            },
                            "optimizer": {
                                "_target_": "torch.optim.AdamW",
                                "lr": args.lr,
                            },
                            "lr_scheduler": None,
                            "ema": None,
                        }
                    }
                )
            }
        )
        trainer.fabric.launch()
        trainer.construct_model()
        trainer.construct_optimizer()
        trainer.construct_scheduler()

        trainable = [p for p in trainer.state["model"].parameters() if p.requires_grad]
        before = trainable[0].detach().clone()
        print(
            f"trainable tensors: {len(trainable)}, "
            f"parameters: {sum(p.numel() for p in trainable) / 1e6:.1f} M"
        )

        loader = torch.utils.data.DataLoader(
            [example] * args.steps, batch_size=1, collate_fn=lambda batch: batch
        )
        trainer.fit(loader)
        after = trainable[0].detach()

    changed = float((after.cpu() - before.cpu()).abs().max())
    print(f"pairs in the loss: {trainer.n_pairs}")
    print("loss per step:", [round(x, 4) for x in probe.losses])
    print("gradient norm per step:", [round(x, 4) for x in probe.grad_norms])
    print(f"largest change of the first trainable tensor: {changed:.3e}")
    ok = (
        len(probe.losses) == args.steps
        and all(map(torch.isfinite, map(torch.tensor, probe.losses)))
        and all(g > 0 for g in probe.grad_norms)
        and probe.losses[-1] < probe.losses[0]
        and changed > 0
    )
    print("OK" if ok else "FAILED")
    if args.json is not None:
        import json
        import subprocess

        def git(*arguments: str) -> str:
            return subprocess.run(
                ["git", *arguments], capture_output=True, text=True, check=False
            ).stdout.strip()

        record = {
            "ok": ok,
            "esmfold2_atomworks_commit": git("rev-parse", "HEAD"),
            "esmfold2_atomworks_dirty": bool(
                git("status", "--porcelain", "--untracked-files=no")
            ),
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "checkpoint": paths.checkpoint_identity(Path(weights)),
            "structure": args.structure.name,
            "steps": args.steps,
            "learning_rate": args.lr,
            "num_loops": args.num_loops,
            "num_sampling_steps": args.num_sampling_steps,
            "precision": args.precision,
            "trainable_parameters": sum(p.numel() for p in trainable),
            "pairs_in_the_loss": trainer.n_pairs,
            "loss_per_step": probe.losses,
            "gradient_norm_per_step": probe.grad_norms,
            "largest_change_of_first_trainable_tensor": changed,
        }
        args.json.write_text(json.dumps(record, indent=2) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
