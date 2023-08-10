import argparse
import json
import os
from copy import copy

import torch
import torchvision
from torchvision import transforms
import timm
import torch.optim as optim
import numpy as np
import wandb
from optmizerAd import DAdam
from torch.utils.data import random_split

VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def set_seed(seed):
    # Set seeds for determinism
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def compute_accuracy(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():  # Don't track gradients during evaluation
        for images, targets in data_loader:
            images = images.to(device)
            labels = [int(target['annotation']['object'][0]['name']) for target in targets]
            labels = torch.tensor(labels).to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

    accuracy = 100 * correct_preds / total_preds
    return accuracy


def apply_config(args: argparse.Namespace, config_path: str):
    """Overwrite the values in an arguments object by values of namesake
    keys in a JSON config file.

    :param args: The arguments object
    :param config_path: the path to a config JSON file.
    """
    config_path = copy(config_path)
    if config_path:
        # Opening JSON file
        f = open(config_path)
        config_overwrite = json.load(f)
        for k, v in config_overwrite.items():
            if k.startswith('_'):
                continue
            setattr(args, k, v)


def _parse_args():
    parser = argparse.ArgumentParser(description='Train a Vision Transformer (ViT) on the Pascal VOC dataset.')

    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--data_root', type=str, default='./data', help='Root directory for Pascal VOC dataset')
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224', help='Name of the ViT model variant')
    parser.add_argument('--config', type=str)
    parser.add_argument("--use_wandb", default=False)
    parser.add_argument('--seed', default=0)
    parser.add_argument('--optimizer')
    parser.add_argument('--save_init_weights_path', type=str, default=None,
                        help='Path to save the initial weights after model initialization')
    parser.add_argument('--load_init_weights_path', type=str, default=None,
                        help='Path to load the saved initial weights before training')

    args = parser.parse_args()

    apply_config(args, args.config)

    return args


def generate_wandb_name(args):
    name = f"model-{args.model_name}"
    name += f"_optim-{args.optimizer}"
    name += f"_dataset-{args.dataset}"

    if args.optimizer == 'fame':
        name += f"_b3-{args.beta3}"
        name += f"_b4-{args.beta4}"

    return name


def deterministic_stratified_split(dataset, train_ratio=0.8):
    # Creating a dictionary to hold indices for each class
    class_indices = {}
    for idx, (_, target) in enumerate(dataset):
        label = int(target['annotation']['object'][0]['name'])
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    train_indices = []
    val_indices = []

    # Splitting indices for each class
    for label, indices in class_indices.items():
        split = int(train_ratio * len(indices))
        train_indices.extend(indices[:split])
        val_indices.extend(indices[split:])

    return train_indices, val_indices


def collate_fn(batch):

    images, targets = zip(*batch)

    # Convert images into a single tensor
    images = torch.stack(images, 0)

    return images, targets


def target_transform(target):
    # Convert the annotations into a binary vector.
    classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    label = torch.zeros(len(classes))
    for obj in target['annotation']['object']:
        label[classes.index(obj['name'])] = 1
    return label


def calculate_metrics(logits, targets, threshold=0.5):
    """
    Compute per-label accuracy and exact match ratio.
    logits: Tensor of shape (batch_size, num_classes)
    targets: Tensor of shape (batch_size, num_classes)
    threshold: threshold for classification
    """
    # Convert logits to binary predictions
    predictions = (torch.sigmoid(logits) > threshold).float()

    # Per-label accuracy
    correct_per_label = (predictions == targets).float().sum(dim=0)
    accuracy_per_label = correct_per_label / targets.size(0)

    # Exact match ratio
    correct_samples = (predictions == targets).all(dim=1).float().sum()
    exact_match_ratio = correct_samples / targets.size(0)

    return accuracy_per_label, exact_match_ratio


def train(args):
    set_seed(args.seed)

    if args.use_wandb:
        wandb.init(project="FAME_optimizer",
                   entity="the-smadars",
                   name=generate_wandb_name(args),
                   config=args)

        wandb.run.summary["best_test_acc"] = 0

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ViT usually expects images of size 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
    ])
    download_2012 = not os.path.exists(os.path.join('data', 'VOCdevkit', 'VOC2012'))
    train_data = torchvision.datasets.VOCDetection(root=args.data_root, year='2012', image_set='train',
                                                   download=download_2012,
                                                   transform=transform)

    download_2007 = not os.path.exists(os.path.join('data', 'VOCdevkit', 'VOC2007'))
    val_data = torchvision.datasets.VOCDetection(root=args.data_root, year='2007', image_set='test',
                                                 download=download_2007,
                                                 transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False,
                                               collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                                             collate_fn=collate_fn)

    model = timm.create_model(args.model_name, pretrained=False, num_classes=20)  # 20 classes for Pascal VOC
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Save initial weights
    if args.save_init_weights_path:
        torch.save(model.state_dict(), args.save_init_weights_path)

    # Load initial weights
    if args.load_init_weights_path:
        if not os.path.exists(args.load_init_weights_path):
            raise ValueError(f"Provided weights path {args.load_init_weights_path} does not exist!")
        model.load_state_dict(torch.load(args.load_init_weights_path))

    criterion = torch.nn.BCEWithLogitsLoss()

    # Prepare optimizer and scheduler
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=0.9,
                                    weight_decay=0)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'fame':
        optimizer = DAdam(model.parameters(), lr=args.lr, beta3=args.beta3, beta4=args.beta4, eps=args.eps)
    else:
        raise Exception(f"no {args.optimizer} optimizer")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_accuracy_per_label = 0
        total_exact_match_ratio = 0

        for images, targets in train_loader:
            optimizer.zero_grad()

            images = images.to(device)
            targets = torch.stack([target_transform(target) for target in targets])

            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            accuracy_per_label, exact_match_ratio = calculate_metrics(outputs, targets)
            total_accuracy_per_label += accuracy_per_label
            total_exact_match_ratio += exact_match_ratio

        avg_loss = total_loss / len(train_loader)

        average_accuracy_per_label = total_accuracy_per_label / len(train_loader)
        average_exact_match_ratio = total_exact_match_ratio / len(train_loader)

        print(f"Epoch [{epoch + 1}/{args.epochs}], Train Loss: {avg_loss:.4f}, Train Accuracy: {average_accuracy_per_label:.4f}")

        val_loss = 0.0
        total_accuracy_per_label = 0
        total_exact_match_ratio = 0

        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = torch.stack([target_transform(target) for target in targets])

                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                accuracy_per_label, exact_match_ratio = calculate_metrics(outputs, targets)
                total_accuracy_per_label += accuracy_per_label
                total_exact_match_ratio += exact_match_ratio

        avg_val_loss = val_loss / len(val_loader)
        average_accuracy_per_label = total_accuracy_per_label / len(val_loader)
        average_exact_match_ratio = total_exact_match_ratio / len(val_loader)
        # print("Valid: Average per-label accuracy:", average_accuracy_per_label)
        # print("Valid: Average exact match ratio:", average_exact_match_ratio)

        if args.use_wandb:
            wandb.log(
                {
                    "test_acc": average_accuracy_per_label,
                }
            )

            wandb.run.summary["best_test_accuracy"] = \
                average_accuracy_per_label if average_accuracy_per_label > wandb.run.summary["best_test_acc"] \
                    else wandb.run.summary["best_test_acc"]

        print(f"Epoch [{epoch + 1}/{args.epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {average_accuracy_per_label:.4f}")


if __name__ == "__main__":
    args = _parse_args()
    train(args)
