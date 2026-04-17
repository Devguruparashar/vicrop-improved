import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the repository's evaluation pipeline sequentially on a single Colab GPU."
    )
    parser.add_argument("task", help="Dataset name, for example: textvqa")
    parser.add_argument("model", help="Model shortcut: llava, blip, or qwen2_5")
    parser.add_argument("method", help="Method name, for example: rel_att")
    parser.add_argument(
        "--total_chunks",
        type=int,
        default=1,
        help="How many chunks to split the dataset into. Chunks run sequentially in Colab.",
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
    args = parser.parse_args()

    for chunk_id in range(args.total_chunks):
        cmd = [
            args.python,
            "run.py",
            "--chunk_id",
            str(chunk_id),
            "--total_chunks",
            str(args.total_chunks),
            "--model",
            args.model,
            "--task",
            args.task,
            "--method",
            args.method,
            "--save_path",
            args.save_path,
        ]
        print(f"Running chunk {chunk_id + 1}/{args.total_chunks}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
