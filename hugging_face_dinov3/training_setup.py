import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import evaluate


class MultiLabelTrainer(Trainer):
    """
    Custom trainer for multi-label classification
    """
    def __init__(self, *args, print_every=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.print_every = print_every

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        # Log a clean scalar you control
        step = getattr(self.state, "global_step", 0)
        if step % self.print_every == 0:
            with torch.no_grad():
                self.log({"train_bce": loss.detach().float()})
                print(
                    f"[step {step}] loss={loss.item():.6f} "
                    f"logits shape={tuple(logits.shape)} "
                    f"logits min/max={logits.min().item():.3f}/{logits.max().item():.3f}"
                )

        return (loss, outputs) if return_outputs else loss

def create_datasets_from_csv(csv_path, images_dir, image_processor, test_size=0.2):
    """
    Create train and test datasets from CSV file
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Get image names and labels
    image_names = df.iloc[:, 0].values
    labels = df.iloc[:, 1:].values
    class_names = list(df.columns[1:])

    # Create label mappings
    label2id = {cls: str(i) for i, cls in enumerate(class_names)}
    id2label = {str(i): cls for i, cls in enumerate(class_names)}

    # Process images and labels
    processed_images = []
    processed_labels = []

    for i, img_name in enumerate(image_names):
        img_path = Path(images_dir) / img_name
        if img_path.exists():
            image = Image.open(img_path).convert('RGB')

            # Process image
            processed = image_processor(image, return_tensors="pt")
            processed_images.append(processed.pixel_values.squeeze(0))
            processed_labels.append(torch.FloatTensor(labels[i]))

    # Split data
    dataset_size = len(processed_images)
    indices = list(range(dataset_size))
    split_point = int(dataset_size * (1 - test_size))

    train_dataset = CustomDataset(
        processed_images[:split_point],
        processed_labels[:split_point]
    )
    test_dataset = CustomDataset(
        processed_images[split_point:],
        processed_labels[split_point:]
    )

    return train_dataset, test_dataset, class_names, label2id, id2label


def create_dataset_from_csv(csv_path, images_dir, image_processor):
    """
    Create dataset from CSV file (multi-label)
    """
    print("\nReading CSV:", csv_path)
    df = pd.read_csv(csv_path)
    print("CSV rows:", len(df), "CSV columns:", len(df.columns))

    image_names = df.iloc[:, 0].values
    labels = df.iloc[:, 1:].values
    class_names = list(df.columns[1:])

    print("Num classes:", len(class_names))
    print("First 5 image names:", list(image_names[:5]))
    print("First 5 classes:", class_names[:5])

    # Label mappings
    label2id = {cls: str(i) for i, cls in enumerate(class_names)}
    id2label = {str(i): cls for i, cls in enumerate(class_names)}

    # Basic label sanity checks
    try:
        row_sums = labels.sum(axis=1)
        print("Rows with 0 labels:", int((row_sums == 0).sum()))
        print("Rows with >1 labels:", int((row_sums > 1).sum()))
        col_sums = labels.sum(axis=0)
        print("Per-class positives min/mean/max:",
              float(col_sums.min()), float(col_sums.mean()), float(col_sums.max()))
    except Exception as e:
        print("Could not compute label stats:", e)

    processed_images = []
    processed_labels = []

    missing_count = 0
    processed_count = 0

    for i, img_name in enumerate(image_names):
        img_path = Path(images_dir) / img_name

        if not img_path.exists():
            missing_count += 1
            if missing_count <= 5:
                print("Missing image file:", str(img_path))
            continue

        image = Image.open(img_path).convert('RGB')

        processed = image_processor(image, return_tensors="pt")
        pv = processed.pixel_values.squeeze(0)  # [C,H,W]
        lab = torch.FloatTensor(labels[i])

        processed_images.append(pv)
        processed_labels.append(lab)
        processed_count += 1

        # Print first 3 samples
        if processed_count <= 3:
            print(f"Loaded sample {processed_count}: {img_name}")
            print("  pixel_values shape:", tuple(pv.shape))
            print("  labels shape:", tuple(lab.shape), "labels sum:", float(lab.sum().item()))

        # Progress print (every 500)
        if processed_count % 500 == 0:
            print("Processed images so far:", processed_count)

    print("Total processed:", processed_count)
    print("Total missing files:", missing_count)

    dataset = CustomDataset(processed_images, processed_labels)
    return dataset, class_names, label2id, id2label


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            'pixel_values': self.images[idx],
            'labels': self.labels[idx]
        }


def create_dataset_from_csv_lazy(csv_path, images_dir):
    print("\nReading CSV:", csv_path)
    df = pd.read_csv(csv_path)
    print("CSV rows:", len(df), "CSV columns:", len(df.columns))

    image_names = df.iloc[:, 0].values
    labels_np = df.iloc[:, 1:].values
    class_names = list(df.columns[1:])

    label2id = {cls: str(i) for i, cls in enumerate(class_names)}
    id2label = {str(i): cls for i, cls in enumerate(class_names)}

    images_dir = Path(images_dir)
    image_paths = []
    valid_labels = []

    missing = 0
    for i, img_name in enumerate(image_names):
        p = images_dir / img_name
        if not p.exists():
            missing += 1
            if missing <= 5:
                print("Missing image file:", str(p))
            continue
        image_paths.append(p)
        valid_labels.append(labels_np[i])

    if len(image_paths) == 0:
        raise ValueError(f"No images found for {csv_path} in {images_dir} (missing={missing}).")

    labels_tensor = torch.tensor(np.asarray(valid_labels), dtype=torch.float32)

    print("Num classes:", len(class_names))
    print("Valid samples:", len(image_paths))
    print("Missing samples:", missing)
    print("Labels tensor shape:", tuple(labels_tensor.shape))

    dataset = LazyMultiLabelImageDataset(image_paths, labels_tensor)
    return dataset, class_names, label2id, id2label


class LazyMultiLabelImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths  # list[str] or list[Path]
        self.labels = labels            # torch.FloatTensor [N, num_classes]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Return path + label; image loading happens in the collator (faster with workers)
        return {
            "image_path": str(self.image_paths[idx]),
            "labels": self.labels[idx],
        }


# Main setup function
def setup_training(train_csv_path, train_images_dir, val_csv_path, val_images_dir, model_name="facebook/dinov2-base"):
    print("\n================= SETUP TRAINING =================")
    print("Model name:", model_name)

    image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    print("Loaded image processor:", type(image_processor).__name__)

    print("\n--- Building TRAIN dataset (lazy) ---")
    train_dataset, class_names, label2id, id2label = create_dataset_from_csv_lazy(train_csv_path, train_images_dir)
    print("Train dataset item keys:", train_dataset[0].keys())
    print("\n--- Building VAL dataset (lazy) ---")
    val_dataset, class_names2, label2id2, id2label2 = create_dataset_from_csv_lazy(val_csv_path, val_images_dir)

    if class_names2 != class_names:
        print("WARNING: Train/Val class columns differ!")
        print("Train classes head:", class_names[:10])
        print("Val classes head:", class_names2[:10])

    print("\n--- Loading model ---")
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    print("Loaded model:", type(model).__name__)
    print("Model hidden_size:", getattr(model.config, "hidden_size", None))
    print("Model config num_labels:", getattr(model.config, "num_labels", None))

    # Replace head for multi-label (as you already do)
    in_features = model.classifier.in_features
    print("Original classifier in_features:", in_features)

    # Replace classifier with correct input dim
    model.classifier = torch.nn.Linear(in_features, len(class_names))
    print("Replaced classifier -> in_features:", model.classifier.in_features,
          "out_features:", model.classifier.out_features)

    print("=================================================\n")
    return model, train_dataset, val_dataset, image_processor, class_names
