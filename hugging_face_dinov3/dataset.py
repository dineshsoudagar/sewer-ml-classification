import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from datasets import Dataset, DatasetDict, Features, Sequence, Value
import os
from pathlib import Path


class CSVMultiLabelImageDataset(Dataset):
    """
    Custom dataset for CSV-based multi-label image classification
    """

    def __init__(self, csv_path, images_dir, image_processor=None, split='train', test_size=0.2):
        """
        Args:
            csv_path: Path to CSV file with image names and one-hot labels
            images_dir: Directory containing the images
            image_processor: Hugging Face image processor (optional)
            split: 'train' or 'test'
            test_size: Fraction of data to use for testing
        """
        self.csv_path = csv_path
        self.images_dir = Path(images_dir)
        self.image_processor = image_processor

        # Load CSV
        self.df = pd.read_csv(csv_path)

        # Get image names and labels
        self.image_names = self.df.iloc[:, 0].values  # First column: image names
        self.labels = self.df.iloc[:, 1:].values  # Rest columns: one-hot labels

        # Get class names from CSV columns (excluding first column)
        self.class_names = list(self.df.columns[1:])
        self.num_classes = len(self.class_names)

        # Create label mappings
        self.label2id = {cls: str(i) for i, cls in enumerate(self.class_names)}
        self.id2label = {str(i): cls for i, cls in enumerate(self.class_names)}

        # Split dataset
        dataset_size = len(self.df)
        indices = list(range(dataset_size))
        split_point = int(dataset_size * (1 - test_size))

        if split == 'train':
            self.indices = indices[:split_point]
        else:
            self.indices = indices[split_point:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get actual index from our subset
        actual_idx = self.indices[idx]

        # Get image name and labels
        img_name = self.image_names[actual_idx]
        labels = self.labels[actual_idx]

        # Load image
        img_path = self.images_dir / img_name
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert('RGB')

        # Process image if processor is provided
        if self.image_processor:
            # Apply transforms
            processed = self.image_processor(image, return_tensors="pt")
            pixel_values = processed.pixel_values.squeeze(0)
        else:
            # Convert to tensor if no processor
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            pixel_values = transform(image)

        # Convert labels to tensor
        labels_tensor = torch.FloatTensor(labels)

        return {
            'pixel_values': pixel_values,
            'labels': labels_tensor,
            'image_name': img_name
        }


# Alternative: Create Hugging Face Dataset directly
def create_hf_dataset_from_csv(csv_path, images_dir, test_size=0.2):
    """
    Create Hugging Face Dataset from CSV and images directory
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Get image names and labels
    image_names = df.iloc[:, 0].values
    labels = df.iloc[:, 1:].values

    # Get class names
    class_names = list(df.columns[1:])

    # Create label mappings
    label2id = {cls: str(i) for i, cls in enumerate(class_names)}
    id2label = {str(i): cls for i, cls in enumerate(class_names)}

    # Load images and create dataset
    images = []
    labels_list = []
    image_paths = []

    for img_name in image_names:
        img_path = Path(images_dir) / img_name
        if img_path.exists():
            images.append(Image.open(img_path).convert('RGB'))
            # Find the corresponding labels
            idx = list(image_names).index(img_name)
            labels_list.append(labels[idx].tolist())
            image_paths.append(str(img_path))

    # Create dataset dictionary
    dataset_dict = {
        'image': images,
        'labels': labels_list
    }

    # Create dataset with proper features
    features = Features({
        'image': datasets.Image(),
        'labels': Sequence(Value('float32'))
    })

    dataset = Dataset.from_dict(dataset_dict, features=features)

    # Split dataset
    dataset = dataset.train_test_split(test_size=test_size)

    return dataset, class_names, label2id, id2label