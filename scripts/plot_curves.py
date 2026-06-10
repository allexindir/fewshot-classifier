"""Plot train vs validation loss and accuracy curves from saved training history."""
import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_history(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    run_name = data.get("run_name", os.path.splitext(os.path.basename(path))[0])
    return {"run_name": run_name, "history": data["history"]}


def plot_metric(ax, run: dict, kind: str) -> None:
    history = run["history"]
    epochs = [h["epoch"] for h in history]
    train = [h[f"train_{kind}"] for h in history]
    val = [h[f"val_{kind}"] for h in history]

    label = "Loss" if kind == "loss" else "Accuracy"
    ax.plot(epochs, train, label=f"Train {label.lower()}", color="tab:blue", marker=".")
    ax.plot(epochs, val, label=f"Val {label.lower()}", color="tab:orange", marker=".")

    if kind == "loss":
        best_idx = min(range(len(val)), key=lambda i: val[i])
    else:
        best_idx = max(range(len(val)), key=lambda i: val[i])
    ax.scatter(
        [epochs[best_idx]],
        [val[best_idx]],
        color="tab:red",
        zorder=5,
        label=f"Best val {label.lower()} {val[best_idx]:.3f} @ ep{epochs[best_idx]}",
    )

    ax.set_title(f"{run['run_name']} — {label}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
    ax.legend()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        nargs="+",
        default=None,
        help="History JSON file(s) defaults to outputs/metrics/*_history.json",
    )
    parser.add_argument(
        "--out",
        default="outputs/plots/training_curves.png",
        help="Output PNG path",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    paths = args.history or sorted(glob.glob("outputs/metrics/*_history.json"))
    if not paths:
        print("No history files found. Train a model first, then re-run.")
        sys.exit(1)

    runs = [load_history(p) for p in paths]

    fig, axes = plt.subplots(
        2, len(runs), figsize=(6 * len(runs), 9), squeeze=False
    )
    for col, run in enumerate(runs):
        plot_metric(axes[0][col], run, "loss")
        plot_metric(axes[1][col], run, "acc")

    fig.suptitle("Train vs Validation — Loss and Accuracy")
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {args.out} ({len(runs)} run(s): {', '.join(r['run_name'] for r in runs)})")


if __name__ == "__main__":
    main()
