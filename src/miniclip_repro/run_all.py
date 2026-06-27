from __future__ import annotations

import argparse
from pathlib import Path

from .eval_retrieval import run_retrieval_eval
from .eval_zeroshot import run_zeroshot_eval
from .train import train_model


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate the Mini-CLIP reproduction.")
    parser.add_argument("--config", default="configs/flickr8k.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--skip-zeroshot", action="store_true")
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_dir, checkpoint = train_model(
        config_path=args.config,
        output_dir=args.output_dir,
        fast_dev_run=args.fast_dev_run,
        synthetic_data=args.synthetic_data,
        device_name=args.device,
    )
    print("Evaluating Flickr8k retrieval on test split...")
    run_retrieval_eval(
        config_path=args.config,
        checkpoint_path=checkpoint,
        output_dir=run_dir,
        split="test",
        fast_dev_run=args.fast_dev_run,
        synthetic_data=args.synthetic_data,
        device_name=args.device,
    )
    if not args.skip_zeroshot:
        print("Evaluating CIFAR-10 zero-shot prompt ablation...")
        run_zeroshot_eval(
            config_path=args.config,
            checkpoint_path=checkpoint,
            output_dir=run_dir,
            fast_dev_run=args.fast_dev_run,
            device_name=args.device,
        )
    print(f"Artifacts written to: {Path(run_dir).resolve()}")


if __name__ == "__main__":
    main()

