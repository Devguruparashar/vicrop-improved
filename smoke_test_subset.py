import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from info import task_to_question_path


def ensure_dataset_files(repo_root: Path, task: str) -> tuple[Path, Path]:
    question_path = (repo_root / task_to_question_path[task]).resolve()
    if not question_path.exists():
        raise FileNotFoundError(
            f"Missing prepared dataset file for task '{task}': {question_path}. "
            "Prepare the dataset in the repo's expected data.json format first."
        )

    backup_path = question_path.with_name("data_full_backup.json")
    if not backup_path.exists():
        shutil.copyfile(question_path, backup_path)

    return backup_path, question_path


def write_subset(backup_path: Path, question_path: Path, num_samples: int) -> int:
    with backup_path.open("r", encoding="utf-8") as f:
        full_data = json.load(f)

    subset = full_data[:num_samples]

    with question_path.open("w", encoding="utf-8") as f:
        json.dump(subset, f, indent=4)

    return len(subset)


def restore_full_data(backup_path: Path, question_path: Path) -> None:
    shutil.copyfile(backup_path, question_path)


def run_command(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Low-memory subset smoke test for prepared repo datasets across supported models."
    )
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=Path("."),
        help="Repository root containing run.py and get_score.py.",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["llava", "blip"],
        help="Model shortcut to evaluate.",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=sorted(task_to_question_path.keys()),
        help="Prepared dataset task to evaluate.",
    )
    parser.add_argument(
        "--method",
        default="rel_att",
        choices=["rel_att", "grad_att", "pure_grad", "rel_att_high", "grad_att_high", "pure_grad_high"],
        help="Attribution method to evaluate.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to keep in the dataset data.json for the smoke test.",
    )
    parser.add_argument(
        "--save_path",
        default="./data/results",
        help="Directory where JSON outputs should be written.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for child processes.",
    )
    parser.add_argument(
        "--keep_subset",
        action="store_true",
        help="Leave the smoke-test subset in place instead of restoring the full backup.",
    )
    parser.add_argument(
        "--no_quant",
        action="store_true",
        help="Disable 4-bit loading even for supported models.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    backup_path, question_path = ensure_dataset_files(repo_root, args.task)

    written_count = write_subset(backup_path, question_path, args.num_samples)
    print(f"Prepared {args.task} smoke subset with {written_count} examples at {question_path}")

    results_dir = (repo_root / args.save_path).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    result_file = results_dir / f"{args.model}-{args.task}-{args.method}.json"
    if result_file.exists():
        result_file.unlink()

    run_cmd = [
        args.python,
        "run.py",
        "--task",
        args.task,
        "--model",
        args.model,
        "--method",
        args.method,
        "--save_path",
        args.save_path,
    ]
    if not args.no_quant:
        run_cmd.append("--load_in_4bit")

    try:
        run_command(run_cmd, cwd=repo_root)
        run_command(
            [
                args.python,
                "get_score.py",
                "--data_dir",
                args.save_path,
                "--save_path",
                ".",
            ],
            cwd=repo_root,
        )
        print(f"Smoke test finished. Results file: {result_file}")
    finally:
        if args.keep_subset:
            print("Leaving the smoke-test subset in place because --keep_subset was set.")
        else:
            restore_full_data(backup_path, question_path)
            print(f"Restored full dataset file from {backup_path}")


if __name__ == "__main__":
    main()
