#!/usr/bin/env python3
import os
import csv
import base64
import numpy as np
import torch
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

# Ensure MedImageInsight is importable
med_image_insights_dir = os.path.join(current_dir, "MedImageInsights")
if not os.path.exists(med_image_insights_dir):
    raise FileNotFoundError(
        f"MedImageInsights directory not found at {med_image_insights_dir}"
    )

sys.path.insert(0, med_image_insights_dir)

from medimageinsightmodel import MedImageInsight


def load_image_ids(dataset_type: str):
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


def encode_dataset_images(dataset_type: str, scorer: MedImageInsight):
    image_dir = os.path.join(current_dir, f"data/{dataset_type}/images")
    image_ids = load_image_ids(dataset_type)
    embeddings = {}
    for image_id in image_ids:
        image_path = os.path.join(image_dir, image_id + ".jpg")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        with torch.no_grad():
            outputs = scorer.encode(images=[image_b64], texts=[""])
            image_vec = outputs["image_embeddings"][0]
            if hasattr(image_vec, "detach"):
                image_vec = image_vec.detach().cpu().numpy()
            embeddings[image_id] = image_vec
    return embeddings


def save_embeddings(dataset_type: str, embeddings):
    os.makedirs(os.path.join(current_dir, "precomputed"), exist_ok=True)
    save_path = os.path.join(
        current_dir, "precomputed", f"image_embeddings_{dataset_type}.npz"
    )
    np.savez(save_path, **embeddings)
    print(f"Saved {len(embeddings)} embeddings for {dataset_type} to {save_path}")


def main():
    scorer = MedImageInsight(
        model_dir=os.path.join(current_dir, "MedImageInsights/2024.09.27"),
        vision_model_name="medimageinsigt-v1.0.0.pt",
        language_model_name="language_model.pth",
    )
    scorer.load_model()

    for dataset in ["valid", "test"]:
        print(f"Precomputing embeddings for {dataset}...")
        embeddings = encode_dataset_images(dataset, scorer)
        save_embeddings(dataset, embeddings)


if __name__ == "__main__":
    main()
