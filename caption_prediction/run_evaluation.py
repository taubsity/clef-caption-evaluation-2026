#!/usr/bin/env python3
import sys
import os
from evaluator import CaptionEvaluator


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_evaluation.py [valid|test]")
        sys.exit(1)

    dataset_type = sys.argv[1].lower()

    if dataset_type not in ["valid", "test"]:
        print("Error: Argument must be either 'valid' or 'test'")
        sys.exit(1)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    ground_truth_path = os.path.join(current_dir, f"data/{dataset_type}/captions.csv")
    submission_file_path = os.path.join(
        current_dir, f"data/{dataset_type}/submission.csv"
    )

    print(f"Running evaluation for {dataset_type} dataset...")
    print(f"Ground truth path: {ground_truth_path}")
    print(f"Submission file path: {submission_file_path}")

    if not os.path.exists(ground_truth_path):
        print(f"Error: Ground truth file not found at {ground_truth_path}")
        sys.exit(1)

    if not os.path.exists(submission_file_path):
        print(f"Error: Submission file not found at {submission_file_path}")
        sys.exit(1)

    caption_evaluator = CaptionEvaluator(ground_truth_path=ground_truth_path)
    _client_payload = {"submission_file_path": submission_file_path}
    _context = {}

    result = caption_evaluator._evaluate(_client_payload, _context)
    print(f"\nEvaluation complete for {dataset_type} dataset!")
    print(result)


if __name__ == "__main__":
    main()
