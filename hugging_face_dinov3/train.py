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

def train_model(model, train_dataset, val_dataset, image_processor, output_dir="multi-label-model"):
    """
    Train the multi-label classification model
    """

    # Custom data collator for multi-label
    def custom_data_collator(features):
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        labels = torch.stack([f["labels"] for f in features])

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,
    )

    # Custom compute metrics for multi-label
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = torch.sigmoid(torch.tensor(predictions)) > 0.5

        # Calculate metrics for multi-label
        accuracy = (predictions == labels).float().mean()
        precision = (predictions * labels).sum() / (predictions.sum() + 1e-8)
        recall = (predictions * labels).sum() / (labels.sum() + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            "accuracy": accuracy.item(),
            "precision": precision.item(),
            "recall": recall.item(),
            "f1": f1.item()
        }

    # Create trainer
    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=custom_data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    return trainer


# Usage example
if __name__ == "__main__":
    # Setup
    train_csv_path = r"D:\expandAI-hiring\expandai-hiring-sewer\train.csv"
    val_csv_path = r"D:\expandAI-hiring\fulldataset\SewerML_Val.csv"
    train_images_dir = r"D:\expandAI-hiring\expandai-hiring-sewer\train_images"
    val_images_dir = r"D:\expandAI-hiring\expandai-hiring-sewer\test_images"
    model_name = "facebook/dinov2-base"
    # Create model and datasets
    model, train_dataset, test_dataset, image_processor, class_names = setup_training(
        train_csv_path, train_images_dir, val_csv_path, val_images_dir, model_name=model_name,
    )

    # Train
    trainer = train_model(model, train_dataset, test_dataset, image_processor)

    # Save model
    trainer.save_model()