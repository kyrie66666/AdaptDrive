#!/usr/bin/env python
"""
Visualize SAC Training Progress
===============================

Plot training curves from TensorBoard logs or saved metrics.

Usage:
    python visualize_training.py --logdir ./sac_logs
    python visualize_training.py --logdir ./sac_logs --output training_curves.png
"""

import argparse
import os
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("Please install required packages: pip install matplotlib tensorboard")
    raise


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize SAC training progress")
    parser.add_argument("--logdir", required=True, help="TensorBoard log directory")
    parser.add_argument("--output", default="training_curves.png", help="Output image file")
    parser.add_argument("--smoothing", default=0.9, type=float, help="Smoothing factor (0-1)")
    return parser.parse_args()


def smooth(data, weight=0.9):
    """Exponential moving average smoothing."""
    smoothed = []
    last = data[0] if data else 0
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def load_tensorboard_data(logdir):
    """Load data from TensorBoard logs."""
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()

    data = {}
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = (steps, values)

    return data


def plot_training_curves(data, output_path, smoothing=0.9):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("SAC Training on Bench2Drive", fontsize=16)

    # Episode reward
    ax = axes[0, 0]
    if "train/episode_reward" in data:
        steps, values = data["train/episode_reward"]
        ax.plot(steps, values, alpha=0.3, color="blue", label="Raw")
        if len(values) > 10:
            ax.plot(steps, smooth(values, smoothing), color="blue", label="Smoothed")
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.set_title("Episode Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Critic loss
    ax = axes[0, 1]
    if "train/critic_loss" in data:
        steps, values = data["train/critic_loss"]
        ax.plot(steps, values, alpha=0.3, color="red")
        if len(values) > 10:
            ax.plot(steps, smooth(values, smoothing), color="red")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Critic Loss")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    # Actor loss
    ax = axes[0, 2]
    if "train/actor_loss" in data:
        steps, values = data["train/actor_loss"]
        ax.plot(steps, values, alpha=0.3, color="green")
        if len(values) > 10:
            ax.plot(steps, smooth(values, smoothing), color="green")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Actor Loss")
        ax.grid(True, alpha=0.3)

    # Alpha (temperature)
    ax = axes[1, 0]
    if "train/alpha" in data:
        steps, values = data["train/alpha"]
        ax.plot(steps, values, color="purple")
        ax.set_xlabel("Step")
        ax.set_ylabel("Alpha")
        ax.set_title("Entropy Temperature")
        ax.grid(True, alpha=0.3)

    # Evaluation reward
    ax = axes[1, 1]
    if "eval/mean_reward" in data:
        steps, values = data["eval/mean_reward"]
        ax.plot(steps, values, marker="o", color="orange", label="Mean Reward")
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.set_title("Evaluation Reward")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Success rate
    ax = axes[1, 2]
    if "eval/success_rate" in data:
        steps, values = data["eval/success_rate"]
        ax.plot(steps, values, marker="o", color="cyan")
        ax.set_xlabel("Step")
        ax.set_ylabel("Success Rate")
        ax.set_title("Route Completion Rate")
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")


def main():
    args = parse_args()

    if not os.path.exists(args.logdir):
        raise FileNotFoundError(f"Log directory not found: {args.logdir}")

    print(f"Loading data from: {args.logdir}")
    data = load_tensorboard_data(args.logdir)

    print(f"Available metrics: {list(data.keys())}")
    plot_training_curves(data, args.output, args.smoothing)


if __name__ == "__main__":
    main()
