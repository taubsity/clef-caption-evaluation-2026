from evaluator import ConceptEvaluator
import sys
import os
import traceback
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
    ground_truth_path = os.path.join(current_dir, f"data/{dataset_type}/concepts.csv")
    secondary_ground_truth_path = os.path.join(
        current_dir, f"data/{dataset_type}/concepts_manual.csv"
    )
    submission_file_path = os.path.join(current_dir, "submission.csv")

    print(f"Running evaluation for {dataset_type} dataset...")
    print(f"Ground truth path: {ground_truth_path}")
    print(f"Secondary ground truth path: {secondary_ground_truth_path}")
    print(f"Submission file path: {submission_file_path}")

    if not os.path.exists(ground_truth_path):
        print(f"Error: Ground truth file not found at {ground_truth_path}")
        sys.exit(1)

    if not os.path.exists(secondary_ground_truth_path):
        print(
            f"Error: Secondary ground truth file not found at {secondary_ground_truth_path}"
        )
        sys.exit(1)

    if not os.path.exists(submission_file_path):
        print(f"Error: Submission file not found at {submission_file_path}")
        sys.exit(1)

    # Run format checks before evaluation
    try:
        check_submission(
            submission_path=submission_file_path,
            primary_gt_path=ground_truth_path,
            secondary_gt_path=secondary_ground_truth_path,
            dataset_type=dataset_type,
        )
        print("Submission format check passed.")
    except SubmissionFormatError as e:
        print("Submission format error detected:\n" + str(e))
        sys.exit(1)
    except Exception:
        print("Unexpected error during submission check. Full traceback:")
        traceback.print_exc()
        sys.exit(1)

    evaluator = ConceptEvaluator(
        ground_truth_path=ground_truth_path,
        secondary_ground_truth_path=secondary_ground_truth_path,
    )
    result = evaluator._evaluate({"submission_file_path": submission_file_path})
    print(result)


if __name__ == "__main__":
    main()
