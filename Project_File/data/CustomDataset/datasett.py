import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, Dataset
import random
from PIL import Image

class ChunkedStreamingDataset(IterableDataset):
    def __init__(self, image_path, label_path, batch_size=128, transform=None):
        self.image_path = image_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.transform = transform
    def load_in_chunks(self):
        # Corrected: Load using file paths, not file handles
        X = np.load(self.image_path, allow_pickle=False, mmap_mode='r')
        y = np.load(self.label_path, allow_pickle=False, mmap_mode='r')

        num_samples = X.shape[0]
        indices = list(range(num_samples))
        random.shuffle(indices)

        for i in range(0, num_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]

            images, labels = [], []
            for idx in batch_indices:
                img = X[idx]  # Read one image at a time
                label = y[idx]

                img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
                label = torch.tensor(label, dtype=torch.long)

                if self.transform:
                    img = self.transform(img)

                images.append(img)
                labels.append(label)

            yield torch.stack(images), torch.tensor(labels)

    def __iter__(self):
        return iter(self.load_in_chunks())


class NumpyDataset(Dataset):
    def __init__(self, image_path, label_path, base_transform, minority_transform=None, minority_classes=None):
        # Use memory mapping for efficiency
        self.X = np.load(image_path, mmap_mode='r')
        self.y = np.load(label_path, mmap_mode='r')
        self.base_transform = base_transform
        self.minority_transform = minority_transform
        self.minority_classes = set(minority_classes) if minority_classes is not None else set()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        img = self.X[idx]
        label = self.y[idx]

        # Convert image from numpy array to tensor (channels first)
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        # Apply base transformation (which includes conversion to PIL)
        img = self.base_transform(img)

        # If label is in minority, apply additional (stronger) augmentation
        if self.minority_transform is not None and label in self.minority_classes:
            img = self.minority_transform(img)

        return img, torch.tensor(label, dtype=torch.long)


class TestNumpyDataset(Dataset):
    def __init__(self, image_paths, labels, base_transform, minority_transform=None, minority_classes=None,
                 image_size=(224, 224), is_train=True):
        """
        Args:
            image_paths (list): List of image file paths.
            labels (list): Corresponding labels.
            base_transform (torchvision.transforms.Compose): Transform pipeline for normal augmentation.
            minority_transform (torchvision.transforms.Compose, optional): Stronger augmentation for minority classes.
            minority_classes (list, optional): List or set of labels for which stronger augmentation should be applied.
            image_size (tuple): Desired image size.
        """
        self.image_paths = list(image_paths)  # ensure integer indexing
        self.labels = list(labels)
        self.base_transform = base_transform
        self.minority_transform = minority_transform
        self.minority_classes = set(minority_classes) if minority_classes is not None else set()
        self.image_size = image_size
        self.is_train = is_train

    def __len__(self):
        return len(self.image_paths)

    # Choose transform based on label
    def __getitem__(self, idx):
        # Load image from file and convert to PIL Image
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            original = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Error opening image {img_path}: {e}")

        # Resize the image to the desired size
        resized_img = original.resize(self.image_size)

        # For training images in minority classes, apply minority_transform first.
        if self.is_train and self.minority_transform is not None and label in self.minority_classes:
            # Apply the stronger (minority) augmentation first on the PIL image
            augmented_img = self.minority_transform(resized_img)
            # Then apply the base transform to get the final tensor output
            img = self.base_transform(augmented_img)
        else:
            # For all other cases (or in validation), only apply the base transform.
            img = self.base_transform(resized_img)

        return img, torch.tensor(label, dtype=torch.long)

# To be used in data has severe imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, label_smoothing=0.0, weight=None):
        """
        FocalLoss that combines label smoothing and class weighting.

        Args:
            alpha (float): Scaling factor.
            gamma (float): Focusing parameter.
            label_smoothing (float): Amount of label smoothing.
            weight (Tensor): A manual rescaling weight given to each class.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.ce = nn.CrossEntropyLoss(reduction='none',
                                      label_smoothing=label_smoothing,
                                      weight=weight)

    def forward(self, logits, labels):
        ce_loss = self.ce(logits, labels)
        pt = torch.exp(-ce_loss)  # probability for the true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()