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
from training_setup import setup_training, MultiLabelTrainer

class MultiLabelImageCollator:
    def __init__(self, image_processor, print_first_batch=True):
        self.image_processor = image_processor
        self.print_first_batch = print_first_batch
        self._printed = False

    def __call__(self, features):
        if not self._printed:
            print("Collator received keys:", features[0].keys())

        images = []
        labels = []

        for f in features:
            img_path = f["image_path"]  # will fail immediately if wrong dataset
            with Image.open(img_path) as im:
                images.append(im.convert("RGB"))
            labels.append(f["labels"])

        processed = self.image_processor(images, return_tensors="pt")
        pixel_values = processed["pixel_values"]
        labels = torch.stack(labels)

        if self.print_first_batch and (not self._printed):
            print("\n--- First batch (lazy collator) ---")
            print("pixel_values batch shape:", tuple(pixel_values.shape))
            print("labels batch shape:", tuple(labels.shape))
            self._printed = True

        return {"pixel_values": pixel_values, "labels": labels}

def train_model(model, train_dataset, val_dataset, image_processor, output_dir="multi-label-model"):
    printed_once = False

    #def custom_data_collator(features):
    #    nonlocal printed_once
    #
    #    images = []
    #    labels = []
    #
    #    for f in features:
    #        img_path = f["image_path"]
    #        image = Image.open(img_path).convert("RGB")
    #        images.append(image)
    #        labels.append(f["labels"])
    #
    #    processed = image_processor(images, return_tensors="pt")
    #    pixel_values = processed["pixel_values"]
    #    labels = torch.stack(labels)
    #
    #    if not printed_once:
    #        print("\n--- First batch (lazy collator) ---")
    #        print("pixel_values batch shape:", tuple(pixel_values.shape))
    #        print("labels batch shape:", tuple(labels.shape))
    #        printed_once = True
    #
    #    return {"pixel_values": pixel_values, "labels": labels}

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=10,
        max_grad_norm=1.0,
        learning_rate = 5e-6,
        warmup_ratio=0.1,
        logging_steps=10,
        logging_first_step=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,
        remove_unused_columns=False,
        # Performance knobs:
        dataloader_num_workers=4,  # try 4, 8 depending on CPU
        dataloader_pin_memory=True,  # good if using CUDA
    )  #

    def compute_metrics(eval_pred):
        logits, labels = eval_pred

        logits = torch.tensor(logits)
        labels = torch.tensor(labels).float()

        preds = (torch.sigmoid(logits) > 0.5).float()

        # Micro metrics (across all labels and samples)
        tp = (preds * labels).sum()
        fp = (preds * (1 - labels)).sum()
        fn = ((1 - preds) * labels).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # Example: exact-match accuracy is too strict for multi-label,
        # so instead compute per-label average correctness:
        accuracy = (preds == labels).float().mean()

        return {
            "accuracy": accuracy.item(),
            "precision": precision.item(),
            "recall": recall.item(),
            "f1": f1.item(),
        }

    collator = MultiLabelImageCollator(image_processor=image_processor, print_first_batch=True)
    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    return trainer


# Usage example
if __name__ == "__main__":
    # Setup
    train_csv_path = r"D:\expandAI-hiring\fulldataset\SewerML_Val_jpg.csv"
    val_csv_path = train_csv_path
    #val_csv_path = r"D:\expandAI-hiring\fulldataset\SewerML_Val_jpg.csv"
    train_images_dir = r"D:\expandAI-hiring\expandai-hiring-sewer\test_images"
    val_images_dir = train_images_dir
    model_name = "facebook/dinov2-base"
    # Create model and datasets
    model, train_dataset, test_dataset, image_processor, class_names = setup_training(
        train_csv_path, train_images_dir, val_csv_path, val_images_dir, model_name=model_name,
    )

    # Train
    trainer = train_model(model, train_dataset, test_dataset, image_processor, output_dir="multi-label-model-val-test")

    # Save model
    trainer.save_model()
