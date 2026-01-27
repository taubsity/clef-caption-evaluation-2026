#!/usr/bin/env python3
import sys
import os
import json
import traceback
from evaluator import CaptionEvaluator
from submission_check import check_submission, SubmissionFormatError


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
    # submission is mounted into /app/submission.csv per README
    submission_file_path = os.path.join(current_dir, "submission.csv")

    print(f"Running evaluation for {dataset_type} dataset...")
    print(f"Ground truth path: {ground_truth_path}")
    print(f"Submission file path: {submission_file_path}")

    if not os.path.exists(ground_truth_path):
        print(f"Error: Ground truth file not found at {ground_truth_path}")
        sys.exit(1)

    if not os.path.exists(submission_file_path):
        print(f"Error: Submission file not found at {submission_file_path}")
        sys.exit(1)

    # Run format checks before evaluation
    try:
        check_submission(
            submission_path=submission_file_path,
            ground_truth_path=ground_truth_path,
            dataset_type=dataset_type,
        )
        print("Submission format check passed.")
    except SubmissionFormatError as e:
        print("Submission format error detected:\n" + str(e))
        sys.exit(1)
    except Exception:
        # Edge case: show full error for participants
        print("Unexpected error during submission check. Full traceback:")
        traceback.print_exc()
        sys.exit(1)

    caption_evaluator = CaptionEvaluator(ground_truth_path=ground_truth_path)
    _client_payload = {"submission_file_path": submission_file_path}
    _context = {}

    result = caption_evaluator._evaluate(_client_payload, _context)
    print(f"\nEvaluation complete for {dataset_type} dataset!")
    print(result)
    
    # Write scores.json for AI4MediaBench platform
    scores_output_path = os.path.join("/app/output", "scores.json")
    # create output directory if it doesn't exist
    os.makedirs(os.path.dirname(scores_output_path), exist_ok=True)
    with open(scores_output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nScores written to {scores_output_path}")


if __name__ == "__main__":
    main()
