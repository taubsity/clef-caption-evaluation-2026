#!/usr/bin/env python3
import os
import csv
import base64
import numpy as np
import torch
import sys
from typing import List
from tqdm import tqdm
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))

# Ensure MedImageInsight is importable
med_image_insights_dir = os.path.join(current_dir, "MedImageInsights")
if not os.path.exists(med_image_insights_dir):
    raise FileNotFoundError(
        f"MedImageInsights directory not found at {med_image_insights_dir}"
    )

sys.path.insert(0, med_image_insights_dir)

from medimageinsightmodel import MedImageInsight


def load_image_ids(dataset_type: str) -> List[str]:
    gt_path = os.path.join(current_dir, f"data/{dataset_type}/captions.csv")
    image_ids = []
    with open(gt_path) as csvfile:
        reader = csv.reader(csvfile)
        first_line = next(reader)
        if first_line and first_line[0].lower() != "id":
            image_ids.append(first_line[0])
        for row in reader:
            image_ids.append(row[0])
    return image_ids


def encode_batch(image_paths: List[str], scorer: MedImageInsight):
    with open(image_paths[0], "rb") as f:
        pass  # quick existence check to surface early errors

    images = []
    for p in image_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image file not found: {p}")
        with open(p, "rb") as f:
            images.append(base64.b64encode(f.read()).decode("utf-8"))

    with torch.inference_mode():
        outputs = scorer.encode(images=images)
        image_vecs = outputs["image_embeddings"]
        if hasattr(image_vecs, "detach"):
            image_vecs = image_vecs.detach().cpu().numpy()
    return image_vecs


def encode_dataset_images(
    dataset_type: str, scorer: MedImageInsight, batch_size: int = 32
):
    image_dir = os.path.join(current_dir, f"data/{dataset_type}/images")
    image_ids = load_image_ids(dataset_type)
    embeddings = {}

    for start in tqdm(
        range(0, len(image_ids), batch_size), desc=f"Encoding {dataset_type}"
    ):
        batch_ids = image_ids[start : start + batch_size]
        batch_paths = [
            os.path.join(image_dir, image_id + ".jpg") for image_id in batch_ids
        ]
        image_vecs = encode_batch(batch_paths, scorer)
        for image_id, vec in zip(batch_ids, image_vecs):
            embeddings[image_id] = vec

    return embeddings


def save_embeddings(dataset_type: str, embeddings):
    os.makedirs(os.path.join(current_dir, "precomputed"), exist_ok=True)
    save_path = os.path.join(
        current_dir, "precomputed", f"image_embeddings_{dataset_type}.npz"
    )
    np.savez(save_path, **embeddings)
    print(f"Saved {len(embeddings)} embeddings for {dataset_type} to {save_path}")


def main():

    parser = argparse.ArgumentParser(
        description="Precompute image embeddings for a dataset."
    )
    parser.add_argument(
        "--dataset",
        choices=["valid", "test"],
        default="valid",
        help="Dataset to precompute embeddings for (default: valid).",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "GPU is required for MedImageInsight model."
    scorer = MedImageInsight(
        model_dir=os.path.join(current_dir, "MedImageInsights/2024.09.27"),
        vision_model_name="medimageinsigt-v1.0.0.pt",
        language_model_name="language_model.pth",
    )
    scorer.load_model()
    if hasattr(scorer, "device"):
        scorer.device = device
    if hasattr(scorer, "to"):
        try:
            scorer.to(device)
        except Exception:
            pass
    print(f"MedImageInsight device: {device}")

    print(f"Precomputing embeddings for {args.dataset}...")
    embeddings = encode_dataset_images(args.dataset, scorer)
    save_embeddings(args.dataset, embeddings)


if __name__ == "__main__":
    main()
