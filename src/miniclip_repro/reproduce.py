from __future__ import annotations

import argparse
from pathlib import Path

from .eval_retrieval import run_retrieval_eval
from .eval_zeroshot import run_zeroshot_eval
from .train import train_model


def run_reproduction(
    config_path: str | Path,
    output_dir: str | Path,
    fast_dev_run: bool = False,
    synthetic_data: bool = False,
    skip_zeroshot: bool = False,
    device_name: str | None = None,
) -> None:
    run_dir, best_checkpoint = train_model(
        config_path=config_path,
        output_dir=output_dir,
        fast_dev_run=fast_dev_run,
        synthetic_data=synthetic_data,
        device_name=device_name,
    )
    run_dir = Path(run_dir)
    last_checkpoint = run_dir / "checkpoint_last.pt"

    retrieval_dir = run_dir / "retrieval_last"
    run_retrieval_eval(
        config_path=config_path,
        checkpoint_path=last_checkpoint,
        output_dir=retrieval_dir,
        split="test",
        fast_dev_run=fast_dev_run,
        synthetic_data=synthetic_data,
        device_name=device_name,
    )

    if not skip_zeroshot:
        zeroshot_dir = run_dir / "zeroshot_best"
        run_zeroshot_eval(
            config_path=config_path,
            checkpoint_path=best_checkpoint,
            output_dir=zeroshot_dir,
            fast_dev_run=fast_dev_run,
            device_name=device_name,
        )

    print(f"Training artifacts: {run_dir.resolve()}")
    print(f"Retrieval artifacts: {retrieval_dir.resolve()}")
    if not skip_zeroshot:
        print(f"Zero-shot artifacts: {(run_dir / 'zeroshot_best').resolve()}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full Mini-CLIP reproduction used for the report.")
    parser.add_argument("--config", default="configs/flickr8k_strong.yaml")
    parser.add_argument("--output-dir", default="outputs/flickr8k-strong-160-b128")
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--skip-zeroshot", action="store_true")
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_reproduction(
        config_path=args.config,
        output_dir=args.output_dir,
        fast_dev_run=args.fast_dev_run,
        synthetic_data=args.synthetic_data,
        skip_zeroshot=args.skip_zeroshot,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
