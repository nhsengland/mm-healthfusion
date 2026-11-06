"""Plot mTAND attention across observation times for saved checkpoints.

Evaluate the test split and write plots and heatmap CSVs to each run's mtand_attention directory.

Usage:
    python -m src.evaluation.mtand_attention --checkpoint <run>/checkpoint_best.ckpt
    python -m src.evaluation.mtand_attention --path <run>
    python -m src.evaluation.mtand_attention --path <runs-root>
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.splits import get_dataloaders
from src.evaluation.eval_utils import (
    CONFIG_NAME,
    DEFAULT_CHECKPOINT_NAME,
    _usable_device,
    discover_checkpoints,
    load_config,
)
from src.training.lightning import MMModel
from src.encoders import mtand as mtand_mod
from src.encoders.mtand import MTANDSequenceEncoder

OUTPUT_DIR_NAME = "mtand_attention"
MARGINAL_PLOT_NAME = "attn_marginal_over_time.png"
_ORDER = ("ecg", "cxr", "ts")
BIN_W = 6.0  # Width of each observation time bin in minutes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot mTAND attention over observation times for a single checkpoint "
            "(--checkpoint) or for every run found under a directory tree (--path)."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to a single trained Lightning checkpoint. Outputs are written to "
            "the checkpoint's run directory (its parent folder)."
        ),
    )
    source.add_argument(
        "--path",
        default=None,
        help=(
            "Run directory, or a root directory to search recursively for run "
            "checkpoints. Every checkpoint_best.ckpt file found under it is evaluated "
            "in place, writing outputs to each run's own directory."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            f"In --path mode, skip runs that already have {OUTPUT_DIR_NAME}/{MARGINAL_PLOT_NAME} "
            "(useful to resume a batch)."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "In --path mode, keep going when a run fails and report the failures at the end "
            "instead of aborting the whole batch."
        ),
    )
    return parser.parse_args(argv)


def mtand_enabled(cfg: Mapping[str, object]) -> bool:
    """Return whether ``config.yaml`` enables the mTAND temporal encoders."""
    mtand_cfg = cfg.get("mtand")
    return isinstance(mtand_cfg, Mapping) and bool(mtand_cfg.get("enabled", False))


def make_hook(name, captured):
    """Capture attention averaged over heads and value channels for one modality."""
    orig = mtand_mod.multiTimeAttention.attention

    def wrapped(self, query, key, value, mask=None, dropout=None):
        out, p_attn = orig(self, query, key, value, mask, dropout)
        # Keep sample, reference time and observation axes
        captured[name] = p_attn.mean(dim=1).mean(dim=-1).detach().cpu()
        return out, p_attn

    return wrapped


def plot_heatmap(out_dir, run_id, m, heat, n_stays, bin_edges, ref_min):
    """Save average attention by reference time and observation time."""
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        heat,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[bin_edges[0], bin_edges[-1], ref_min[0], ref_min[-1]],
    )
    ax.set_xlabel("Observation time (minutes since ED arrival)")
    ax.set_ylabel("mTAND reference query time (minutes)")
    ax.set_title(
        f"Mean mTAND attention to observation times — {m.upper()}\n"
        f"{run_id}  (n={n_stays} stays)"
    )
    fig.colorbar(im, ax=ax, label="mean attention weight")
    fig.tight_layout()
    p = out_dir / f"attn_heatmap_{m}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_marginal(out_dir, run_id, heat, nseen, bin_cent):
    """Save attention over observation times, averaged across reference times."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in heat:
        if not nseen[m]:
            continue
        marg = heat[m].mean(axis=0)  # Average over reference times
        marg = marg / marg.sum() if marg.sum() else marg
        ax.plot(bin_cent, marg, marker="o", label=f"{m.upper()} (n={nseen[m]})")
    ax.set_xlabel("Observation time (minutes since ED arrival)")
    ax.set_ylabel("Normalised attention mass")
    ax.set_title("Where mTAND attends over time (marginal over reference grid)\n" f"{run_id}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / MARGINAL_PLOT_NAME
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_examples(out_dir, m, ex, ref_min):
    """Save attention for example stays when observations are available."""
    if not ex:
        return None
    n = len(ex)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4), squeeze=False)
    for j, (ab, tb, sid, lab) in enumerate(ex):
        ax = axes[0][j]
        ax.imshow(
            ab,
            aspect="auto",
            origin="lower",
            cmap="magma",
            extent=[0, len(tb), ref_min[0], ref_min[-1]],
        )
        ax.set_xticks(np.arange(len(tb)) + 0.5)
        ax.set_xticklabels([f"{t:.0f}" for t in tb], fontsize=7, rotation=90)
        ax.set_xlabel("obs time (min)")
        if j == 0:
            ax.set_ylabel("reference time (min)")
        ax.set_title(f"stay {sid}\nlabel={lab}", fontsize=8)
    fig.suptitle(f"Per-stay mTAND attention — {m.upper()} (reference x observation)")
    fig.tight_layout()
    p = out_dir / f"attn_examples_{m}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def write_run_attention(run_dir: str | Path, checkpoint: str | Path | None = None) -> list[Path]:
    """Write attention plots and heatmap CSVs for one run and return their paths.

    Read config.yaml and the checkpoint (default checkpoint_best.ckpt) from the run directory
    and write to its mtand_attention folder.
    """
    run_dir = Path(run_dir)
    checkpoint = run_dir / DEFAULT_CHECKPOINT_NAME if checkpoint is None else Path(checkpoint)
    run_id = run_dir.name
    config_path = run_dir / CONFIG_NAME
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = load_config(config_path)
    if not mtand_enabled(cfg):
        raise ValueError(
            f"{run_id}: mtand.enabled is not true in {CONFIG_NAME}. Time attention only "
            "exists for runs trained with mTAND temporal encoders."
        )
    time_scale = float(cfg["mtand"]["time_scale"])  # Minutes since ED arrival
    tt_max = int(cfg["mtand"]["tt_max"])
    ref_min = np.linspace(0.0, 1.0, tt_max) * time_scale  # Reference times in minutes

    device = _usable_device()
    model = MMModel.load_from_checkpoint(str(checkpoint), map_location=device)
    model.eval().to(device)
    inner = model.model

    # Find the modalities with temporal encoders
    found = {
        name[: -len("_encoder")]: mod
        for name, mod in inner._modules.items()
        if name.endswith("_encoder") and isinstance(mod, MTANDSequenceEncoder)
    }
    if not found:
        raise ValueError(
            f"{run_id}: no MTANDSequenceEncoder on {type(inner).__name__}, so there is no "
            "time attention to plot."
        )
    seq_mods = tuple([m for m in _ORDER if m in found] + sorted(set(found) - set(_ORDER)))
    encoders = {m: found[m] for m in seq_mods}
    print(f"mTAND modalities: {', '.join(seq_mods)}")

    # Capture attention weights discarded by the encoder output
    captured: dict[str, torch.Tensor] = {}
    originals = {m: e.encoder.time_attn.attention for m, e in encoders.items()}
    for m, enc in encoders.items():
        enc.encoder.time_attn.attention = make_hook(m, captured).__get__(
            enc.encoder.time_attn, mtand_mod.multiTimeAttention
        )

    # Collect attention by reference time and observation time bin
    bin_edges = np.arange(0.0, time_scale + BIN_W, BIN_W)
    bin_cent = (bin_edges[:-1] + bin_edges[1:]) / 2
    nb = len(bin_cent)

    heat = {m: np.zeros((tt_max, nb)) for m in seq_mods}  # Total attention weight
    nseen = {m: 0 for m in seq_mods}
    examples = {m: [] for m in seq_mods}  # (attn[R,L], times[L], stay_id, label)

    dl = get_dataloaders(cfg, splits=("test",))["test"]
    try:
        with torch.no_grad():
            for batch in dl:
                gpu = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                # Hospitalisation batches identify stays by hadm_id, deterioration by ed_stay_id
                id_column = "hadm_id" if "hadm_id" in batch else "ed_stay_id"
                captured.clear()
                model(gpu)
                for m in seq_mods:
                    if m not in captured:
                        continue
                    a = captured[m].numpy()  # [B, R, L]
                    times = batch[f"{m}_time"].numpy()  # [B, L] minutes
                    lengths = batch[f"{m}_lengths"].numpy()
                    B, R, L = a.shape
                    for b in range(B):
                        Lb = int(lengths[b])
                        if Lb <= 0:
                            continue  # modality absent for this stay
                        tb = times[b, :Lb]
                        ab = a[b, :, :Lb]  # [R, Lb], rows ~sum to 1
                        idx = np.clip(np.digitize(tb, bin_edges) - 1, 0, nb - 1)
                        for r in range(R):
                            np.add.at(heat[m][r], idx, ab[r])
                        nseen[m] += 1
                        if len(examples[m]) < 6 and Lb >= (3 if m != "ecg" else 1):
                            examples[m].append(
                                (
                                    ab.copy(),
                                    tb.copy(),
                                    int(batch[id_column][b]),
                                    int(batch["label"][b]),
                                )
                            )
    finally:
        for m, enc in encoders.items():
            enc.encoder.time_attn.attention = originals[m]

    for m in seq_mods:
        if nseen[m]:
            heat[m] /= nseen[m]  # Average attention per stay
        print(f"{m}: {nseen[m]} stays with >=1 observation")

    out_dir = run_dir / OUTPUT_DIR_NAME
    out_dir.mkdir(exist_ok=True)
    written = []
    for m in seq_mods:
        if not nseen[m]:
            continue
        written.append(plot_heatmap(out_dir, run_id, m, heat[m], nseen[m], bin_edges, ref_min))
        e = plot_examples(out_dir, m, examples[m], ref_min)
        if e:
            written.append(e)
        # Save the average attention as CSV
        with (out_dir / f"attn_heatmap_{m}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ref_time_minutes"] + [f"obs_{c:.0f}min" for c in bin_cent])
            for r in range(tt_max):
                w.writerow([f"{ref_min[r]:.2f}"] + [f"{v:.6f}" for v in heat[m][r]])
        written.append(out_dir / f"attn_heatmap_{m}.csv")
    written.append(plot_marginal(out_dir, run_id, heat, nseen, bin_cent))
    return written


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.checkpoint is not None:
        checkpoint = Path(args.checkpoint)
        written = write_run_attention(checkpoint.parent, checkpoint)
        print("WROTE:")
        for p in written:
            print(" ", p)
        return

    checkpoints = discover_checkpoints(args.path)
    if not checkpoints:
        print(f"No '{DEFAULT_CHECKPOINT_NAME}' checkpoints found under {args.path}.")
        return

    print(f"Found {len(checkpoints)} checkpoint(s) under {args.path}.")
    failures: list[tuple[Path, str]] = []
    evaluated = 0
    skipped = 0
    for index, checkpoint in enumerate(checkpoints, start=1):
        run_dir = checkpoint.parent
        if args.skip_existing and (run_dir / OUTPUT_DIR_NAME / MARGINAL_PLOT_NAME).exists():
            print(f"[{index}/{len(checkpoints)}] skip (exists): {run_dir}")
            continue

        run_config = run_dir / CONFIG_NAME
        if run_config.exists() and not mtand_enabled(load_config(run_config)):
            skipped += 1
            print(f"[{index}/{len(checkpoints)}] skip (no mTAND): {run_dir}")
            continue

        print(f"[{index}/{len(checkpoints)}] evaluating: {run_dir}")
        try:
            written = write_run_attention(run_dir)
            evaluated += 1
            print("WROTE:")
            for p in written:
                print(" ", p)
        except Exception as error:  # noqa: BLE001 - reported per-run below
            if not args.continue_on_error:
                raise
            failures.append((run_dir, f"{type(error).__name__}: {error}"))
            print(f"    FAILED: {type(error).__name__}: {error}")

    print(
        f"Done. Evaluated {evaluated}/{len(checkpoints)} run(s), "
        f"skipped {skipped} run(s) without mTAND."
    )
    if failures:
        print(f"{len(failures)} run(s) failed:")
        for run_dir, message in failures:
            print(f"  {run_dir}: {message}")


if __name__ == "__main__":
    main()
