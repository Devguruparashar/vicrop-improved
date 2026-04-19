import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def build_textvqa_records(raw_data):
    records = []
    for data_id, data in enumerate(raw_data["data"]):
        records.append(
            {
                "id": str(data_id).zfill(10),
                "question": data["question"],
                "labels": data["answers"],
                "image_path": f"{data['image_id']}.jpg",
            }
        )
    return records


def ensure_textvqa_files(repo_root: Path) -> tuple[Path, Path]:
    data_dir = repo_root / "data" / "textvqa"
    data_dir.mkdir(parents=True, exist_ok=True)

    full_backup_path = data_dir / "data_full_backup.json"
    unified_data_path = data_dir / "data.json"
    official_json_path = data_dir / "TextVQA_0.5.1_val.json"

    if full_backup_path.exists():
        return full_backup_path, unified_data_path

    if not official_json_path.exists():
        raise FileNotFoundError(
            "Missing TextVQA source file. Expected "
            f"{official_json_path}. Download/prepare TextVQA first."
        )

    with official_json_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    records = build_textvqa_records(raw_data)

    with full_backup_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=4)

    return full_backup_path, unified_data_path


def write_subset(backup_path: Path, unified_data_path: Path, num_samples: int) -> int:
    with backup_path.open("r", encoding="utf-8") as f:
        full_data = json.load(f)

    subset = full_data[:num_samples]

    with unified_data_path.open("w", encoding="utf-8") as f:
        json.dump(subset, f, indent=4)

    return len(subset)


def restore_full_data(backup_path: Path, unified_data_path: Path) -> None:
    shutil.copyfile(backup_path, unified_data_path)


def run_command(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Low-memory smoke test for llava + textvqa + rel_att."
    )
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=Path("."),
        help="Repository root containing run.py and get_score.py.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of TextVQA examples to keep in data/textvqa/data.json for the smoke test.",
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
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    full_backup_path, unified_data_path = ensure_textvqa_files(repo_root)

    written_count = write_subset(full_backup_path, unified_data_path, args.num_samples)
    print(f"Prepared TextVQA smoke subset with {written_count} examples at {unified_data_path}")

    results_dir = (repo_root / args.save_path).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    result_file = results_dir / "llava-textvqa-rel_att.json"
    if result_file.exists():
        result_file.unlink()

    try:
        run_command(
            [
                args.python,
                "run.py",
                "--task",
                "textvqa",
                "--model",
                "llava",
                "--method",
                "rel_att",
                "--save_path",
                args.save_path,
                "--load_in_4bit",
            ],
            cwd=repo_root,
        )
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
            restore_full_data(full_backup_path, unified_data_path)
            print(f"Restored full TextVQA data file from {full_backup_path}")


if __name__ == "__main__":
    main()
