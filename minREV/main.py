"""Train CIFAR10 with PyTorch."""
import argparse
import json
import time
from copy import copy

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler

from optmizerAd import DAdam

import wandb
from fast_rev import FastRevViT
from rev import RevViT
from revmvit import ReversibleMViT as RevMViT
from revswin import ReversibleSwinTransformer as RevSwin
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")

    # Optimizer options
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--bs", default=128, type=int, help="batch size")

    parser.add_argument("--seed", default=0, type=int, help="seed")
    parser.add_argument("--optimizer", default="adam", type=str, help="optimizer type, adam/sgd/fame")
    parser.add_argument("--beta3", default=0.3, type=float, help="beta3 for fame")
    parser.add_argument("--beta4", default=0.7, type=float, help="beta4 for fame")
    parser.add_argument("--use_wandb", default=False)
    parser.add_argument("--load", default=None, help="Path to init weights")
    parser.add_argument("--eps", default=1e-8, help="epsilon for fame optimizer")

    parser.add_argument("--config", type=str)

    parser.add_argument("--dataset", type=str, default='CIFAR100', help="CIFAR10/CIFAR100")

    parser.add_argument(
        "--epochs", default=200, type=int, help="number of classes in the dataset"
    )


    # Transformer options
    parser.add_argument("--model", default="vit", type=str, help="model name")
    parser.add_argument(
        "--embed_dim",
        default=256,
        type=int,
        help="embedding dimension of the transformer",
    )
    parser.add_argument(
        "--n_head", default=8, type=int, help="number of heads in the transformer"
    )
    parser.add_argument(
        "--depth", default=4, type=int, help="number of transformer blocks"
    )
    parser.add_argument(
        "--patch_size", default=(4, 4), help="patch size in patchification"
    )
    parser.add_argument("--image_size", default=(32, 32), help="input image size")
    parser.add_argument(
        "--num_classes",
        default=10,
        type=int,
        help="number of classes in the dataset",
    )

    # To train the reversible architecture with or without reversible backpropagation
    parser.add_argument(
        "--vanilla_bp",
        default=False,
        type=bool,
        help="whether to use reversible backpropagation or not",
    )
    parser.add_argument(
        "--pareprop",
        default=False,
        type=bool,
        help="whether to use fast, parallel reversible backpropagation or not",
    )
    parser.add_argument(
        "--amp",
        default=False,
        type=bool,
        help="whether to use mixed precision training or not",
    )

    args = parser.parse_args()

    return args

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

# Training
def train(epoch):
    t = time.time()
    print("\nTraining Epoch: %d" % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # We do not need to specify AMP autocast in forward pass here since
        # that is taken care of already in the forward of individual modules.
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # standard pytorch AMP training setup
        # scaler also works without amp training.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(f"Training Accuracy:{100. * correct / total: 0.2f}")
    print(f"Training Loss:{train_loss / (batch_idx + 1): 0.3f}")
    print(f"train time: {time.time() - t}")
    return 100.0 * correct / total, train_loss / (batch_idx + 1)


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    t = time.time()
    print("\nTesting Epoch: %d" % epoch)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print(f"Test Accuracy:{100. * correct / total: 0.2f}")
        print(f"Test Loss:{test_loss / (batch_idx + 1): 0.3f}")
        print(f"test time: {time.time() - t}")
        return 100.0 * correct / total, test_loss / (batch_idx + 1)


def generate_wandb_name(args):
    name = f"model-{args.model}"
    name += f"_optim-{args.optimizer}"
    name += f"_dataset-{args.dataset}"

    if args.optimizer == 'fame':
        name += f"_b3-{args.beta3}"
        name += f"_b4-{args.beta4}"

    return name


if __name__ == '__main__':
    args = parse_args()
    apply_config(args, args.config)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    print("==> Preparing data..")
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    if args.dataset == 'CIFAR10':
        # Will downloaded and save the dataset if needed
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.bs, shuffle=True, num_workers=2
        )

        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.bs, shuffle=False, num_workers=2
        )

    elif args.dataset == 'CIFAR100':
        CIFAR100_normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.ToTensor(), CIFAR100_normalize])
        test_transform = transforms.Compose([transforms.ToTensor(), CIFAR100_normalize])
        trainset = torchvision.datasets.CIFAR100(root='./dataset/train', train=True, download=True,
                                                 transform=train_transform)
        testset = torchvision.datasets.CIFAR100(root='./dataset/test', train=False, download=True,
                                                transform=test_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True,
                                                  num_workers=2, pin_memory=True, generator=generator)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False,
                                                 num_workers=2, pin_memory=True)
    else:
        raise Exception(f"no {args.dataset} dataset")

    if args.load is not None:
        print("loading weights from:", args.load)
        model = torch.load(args.load, map_location=torch.device(device))

    elif args.model == "vit":
        if args.pareprop:
            rev_arch = FastRevViT
        else:
            rev_arch = RevViT

        model = rev_arch(
            embed_dim=args.embed_dim,
            n_head=args.n_head,
            depth=args.depth,
            patch_size=args.patch_size,
            image_size=args.image_size,
            num_classes=args.num_classes,
            enable_amp=args.amp,
        )
    elif args.model == "swin":
        model = RevSwin(
            img_size=args.image_size,
            patch_size=args.patch_size,
            num_classes=args.num_classes,
            embed_dim=args.embed_dim,
            depths=[args.depth // 2, args.depth // 2],
            num_heads=[args.n_head, args.n_head * 2],
            window_size=4,
            fast_backprop=args.pareprop,
        )
    elif args.model == "mvit":
        model = RevMViT(
            img_size=args.image_size,
            patch_kernel=(3, 3),
            patch_stride=(2, 2),
            patch_padding=(1, 1),
            num_classes=args.num_classes,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.n_head,  # doubles every stage
            last_block_indexes=[0, 2],
            qkv_pool_kernel=(3, 3),
            adaptive_kv_stride=2,
            adaptive_window_size=16,
            fast_backprop=args.pareprop,
        )
    else:
        raise NotImplementedError(f"Model {args.model} not supported.")

    model = model.to(device)

    # Whether to use memory-efficient reversible backpropagation or vanilla backpropagation
    # Note that in both cases, the model is reversible.
    # For Swin, this requires iterating through the layers.
    model.no_custom_backward = args.vanilla_bp

    criterion = nn.CrossEntropyLoss()

    # set up optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == "fame":
        optimizer = DAdam(model.parameters(), lr=args.lr, beta3=args.beta3, beta4=args.beta4, eps=args.eps)
    else:
        raise Exception(f"no {args.optimizer} optimizer")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = GradScaler()

    if args.use_wandb:
        wandb.init(project="FAME_optimizer",
                   entity="the-smadars",
                   name=generate_wandb_name(args),
                   config=args)

        wandb.run.summary["best_test_acc"] = 0
        wandb.run.summary["best_test_loss"] = 999

    for epoch in range(args.epochs):
        train_acc, train_loss = train(epoch)
        test_acc, test_loss = test(epoch)
        if args.use_wandb:
            wandb.log(
                {
                    "train_acc": train_acc,
                    "train_loss": train_loss,
                    "test_acc": test_acc,
                    "test_loss": test_loss,
                }
            )

            wandb.run.summary["best_test_accuracy"] = \
                test_acc if test_acc > wandb.run.summary["best_test_acc"] \
                    else wandb.run.summary["best_test_acc"]

            wandb.run.summary["best_test_loss"] = \
                test_loss if test_loss < wandb.run.summary["best_test_loss"] \
                    else wandb.run.summary["best_test_loss"]

        scheduler.step(epoch - 1)

    # based on https://github.com/kentaroy47/vision-transformers-cifar10