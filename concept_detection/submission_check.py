import os
import csv
import re
import argparse
import traceback


class SubmissionFormatError(Exception):
    pass


def _read_utf8(path: str) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read()
        return data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise SubmissionFormatError(
            "File encoding error: submission.csv must be UTF-8.\nDetails: " + str(e)
        )


def _load_ground_truth_ids(primary_path: str, secondary_path: str) -> list:
    # Order must follow primary concepts.csv; secondary adds IDs if present
    ids = []
    seen = set()
    for path in (primary_path, secondary_path):
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header or len(header) < 1:
                raise SubmissionFormatError(
                    f"Ground truth file header missing or invalid: {path}"
                )
            id_idx = 0
            for row in reader:
                if not row:
                    continue
                _id = row[id_idx].strip()
                if _id not in seen:
                    ids.append(_id)
                    seen.add(_id)
    return ids


def check_submission(
    submission_path: str,
    primary_gt_path: str,
    secondary_gt_path: str,
    dataset_type: str,
) -> None:
    if not os.path.exists(submission_path):
        raise SubmissionFormatError(f"Submission file not found at {submission_path}.")

    raw_text = _read_utf8(submission_path)

    lines = raw_text.splitlines()
    if any(len(line.strip()) == 0 for line in lines if line != lines[0]):
        raise SubmissionFormatError(
            "Empty line detected: No extra blank lines are allowed."
        )

    try:
        with open(submission_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                raise SubmissionFormatError(
                    "Missing header row. Expected columns: ID,CUIs."
                )
            if len(header) != 2 or header[0] != "ID" or header[1] != "CUIs":
                raise SubmissionFormatError(
                    f"Invalid header. Expected exactly two columns: ID,CUIs. Found: {header}"
                )

            seen_ids = set()
            gt_ids_list = _load_ground_truth_ids(primary_gt_path, secondary_gt_path)
            gt_ids = set(gt_ids_list)
            missing_in_submission = set(gt_ids)

            for i, row in enumerate(reader, start=2):
                if not row or len(row) != 2:
                    raise SubmissionFormatError(
                        f"Row {i}: Expected exactly 2 columns. Found: {row}"
                    )
                image_id, cuis_str = row[0], row[1]

                if image_id != image_id.strip():
                    raise SubmissionFormatError(
                        f"Row {i}: ID has leading/trailing whitespace. Found: '{image_id}'."
                    )
                if cuis_str != cuis_str.strip():
                    raise SubmissionFormatError(
                        f"Row {i}: CUIs field has leading/trailing whitespace. Found: '{cuis_str}'."
                    )

                if image_id in seen_ids:
                    raise SubmissionFormatError(
                        f"Row {i}: Duplicate ID detected: {image_id}."
                    )
                position = len(seen_ids)
                if position >= len(gt_ids_list):
                    raise SubmissionFormatError(
                        f"Row {i}: Extra ID beyond ground truth length: {image_id}."
                    )
                expected_id = gt_ids_list[position]
                if image_id != expected_id:
                    raise SubmissionFormatError(
                        f"Row {i}: ID order mismatch. Expected '{expected_id}' at position {position + 1}, found '{image_id}'."
                    )

                if image_id not in gt_ids:
                    raise SubmissionFormatError(
                        f"Row {i}: ID '{image_id}' not in {dataset_type} ground truth set."
                    )
                seen_ids.add(image_id)
                if image_id in missing_in_submission:
                    missing_in_submission.remove(image_id)

                # CUIs may be empty; otherwise must be ';' separated, no duplicates, and valid format
                if cuis_str.strip():
                    parts = [p.strip() for p in cuis_str.split(";")]
                    if any(len(p) == 0 for p in parts):
                        raise SubmissionFormatError(
                            f"Row {i}: Empty CUI detected (possible trailing ';' or consecutive separators)."
                        )
                    invalid = [p for p in parts if not re.fullmatch(r"C\d+", p)]
                    if invalid:
                        raise SubmissionFormatError(
                            f"Row {i}: Invalid CUI format for entries: {invalid}. Expected 'C' followed by digits."
                        )
                    if len(set(parts)) != len(parts):
                        dupes = sorted([c for c in set(parts) if parts.count(c) > 1])
                        raise SubmissionFormatError(
                            f"Row {i}: Duplicate CUIs not allowed for an ID. Duplicates: {dupes}"
                        )

            if missing_in_submission:
                missing_count = len(missing_in_submission)
                sample = sorted(list(missing_in_submission))[:5]
                raise SubmissionFormatError(
                    f"Submission incomplete: {missing_count} official IDs are missing. Example: {sample}"
                )

    except csv.Error as e:
        raise SubmissionFormatError(f"CSV parsing error: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate concept submission.csv format without Docker."
    )
    parser.add_argument("--submission", required=True, help="Path to submission.csv")
    parser.add_argument("--primary-gt", dest="primary_gt", help="Path to concepts.csv")
    parser.add_argument(
        "--secondary-gt", dest="secondary_gt", help="Path to concepts_manual.csv"
    )
    parser.add_argument(
        "--dataset",
        choices=["valid", "test"],
        default="valid",
        help="Dataset split to validate against (used to infer default ground truth paths).",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    primary_gt = args.primary_gt or os.path.join(
        base_dir, "data", args.dataset, "concepts.csv"
    )
    secondary_gt = args.secondary_gt or os.path.join(
        base_dir, "data", args.dataset, "concepts_manual.csv"
    )

    missing_paths = [p for p in (primary_gt, secondary_gt) if not os.path.exists(p)]
    if missing_paths:
        print(f"Ground truth file(s) not found: {missing_paths}")
        raise SystemExit(1)

    try:
        check_submission(args.submission, primary_gt, secondary_gt, args.dataset)
        print("Submission format check passed.")
    except SubmissionFormatError as e:
        print("Submission format error detected:\n" + str(e))
        raise SystemExit(1)
    except Exception:
        print("Unexpected error during submission check. Full traceback:")
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
