import argparse

from ml_playground.train_baselines import run_baseline_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline classification models.")
    parser.add_argument("--data", default="data/sample_iot_windows.csv", help="Path to CSV data")
    parser.add_argument("--target", default="label", help="Target column name")
    parser.add_argument("--output", default="reports", help="Output folder")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run_baseline_experiment(
        data_path=args.data,
        target=args.target,
        output_dir=args.output,
    )
    print(results.to_string(index=False))
