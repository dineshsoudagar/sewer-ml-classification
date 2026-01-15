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

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Use BCEWithLogitsLoss for multi-label classification
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)

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

    dataset = CustomDataset(
        processed_images,
        processed_labels
    )

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


# Main setup function
def setup_training(train_csv_path, train_images_dir, val_csv_path, val_images_dir, model_name="facebook/dinov2-base"):
    """
    Complete setup for multi-label classification training
    """
    # Load image processor
    image_processor = AutoImageProcessor.from_pretrained(model_name)

    # Create datasets

    train_dataset, class_names, label2id, id2label = create_dataset_from_csv(
        train_csv_path, train_images_dir, image_processor
    )
    print("Train dataset size: {}".format(len(train_dataset)))
    print("Train dataset classes: {}".format(class_names))
    val_dataset, class_names, label2id, id2label = create_dataset_from_csv(
        val_csv_path, val_images_dir, image_processor
    )
    print("Val dataset size: {}".format(len(val_dataset)))
    print("Val dataset classes: {}".format(class_names))
    # Load model
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # Modify model for multi-label classification
    # Replace the classification head
    model.classifier = torch.nn.Linear(model.config.hidden_size, len(class_names))

    return model, train_dataset, val_dataset, image_processor, class_names
