import argparse
import subprocess
import sys
from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = "configs/final_run.yaml"

REQUIRED_EXPERIMENT_FIELDS = {
    "run_name",
    "augmentation_type",
    "epochs",
    "batch_size",
    "lr",
}

TRAIN_ARG_FIELDS = [
    "run_name",
    "batch_size",
    "epochs",
    "lr",
    "augmentation_type",
    "seed",
    "image_dir",
    "mask_dir",
    "model_dir",
    "results_path",
    "num_workers",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one or more training experiments from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to experiment YAML config. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the resolved training commands without running them.",
    )

    return parser.parse_args()


def load_config(path):
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    experiments = config.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("Config must contain a non-empty 'experiments' list.")

    defaults = config.get("defaults", {})
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, dict):
        raise ValueError("'defaults' must be a mapping when provided.")

    return defaults, experiments


def resolve_experiment(defaults, experiment):
    if not isinstance(experiment, dict):
        raise ValueError("Each experiment must be a YAML mapping.")

    resolved = {**defaults, **experiment}
    missing = sorted(REQUIRED_EXPERIMENT_FIELDS - resolved.keys())
    if missing:
        run_name = resolved.get("run_name", "<unknown>")
        raise ValueError(f"Experiment {run_name} is missing fields: {missing}")

    return resolved


def build_command(experiment):
    command = [
        sys.executable,
        "src/train.py",
    ]

    for field in TRAIN_ARG_FIELDS:
        if field in experiment and experiment[field] is not None:
            command.extend([f"--{field}", str(experiment[field])])

    return command


def main():
    args = parse_args()
    defaults, experiments = load_config(args.config)

    print(f"Loaded config: {args.config}")
    print(f"Experiments: {len(experiments)}")

    for idx, experiment in enumerate(experiments, start=1):
        exp = resolve_experiment(defaults, experiment)
        command = build_command(exp)

        print("\n" + "=" * 80)
        print(f"Running experiment {idx}/{len(experiments)}: {exp['run_name']}")
        print("Command:", " ".join(command))
        print("=" * 80)

        if args.dry_run:
            continue

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"FAILED: {exp['run_name']} with return code {e.returncode}")
            print("Continuing with next experiment...")


if __name__ == "__main__":
    main()
