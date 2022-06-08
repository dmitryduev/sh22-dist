import argparse
import os
from datetime import datetime
from typing import Callable

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets
from torchvision.transforms import ToTensor


def setup(rank: int, world_size: int, backend: str, init_method: str) -> None:
    torch.distributed.init_process_group(
        backend, rank=rank, world_size=world_size, init_method=init_method
    )


def cleanup() -> None:
    torch.distributed.destroy_process_group()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels):
        "Initialization"
        self.labels = labels
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        # Load data and get label
        X = self.inputs[index]
        y = self.labels[index]

        return X, y


def train(rank: int, args) -> None:
    print(f"Running train on rank {rank}.")
    setup(
        backend=args.backend,
        rank=rank,
        world_size=args.world_size,
        init_method=args.init_method,
    )

    train_dataset = datasets.FashionMNIST(
        root=args.data_path, train=True, download=True, transform=ToTensor()
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=rank
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler,
    )

    model = Model().to(rank)
    torch.cuda.set_device(rank)
    ddp_model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss().cuda(rank)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=args.lr)
    run = wandb.init(project="summer_hack_2022", group=args.group_name, config=args)

    total_num_steps = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):

            wandb.log({"images": wandb.Image(images)})
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)



            # FWD pass
            outputs = ddp_model(images)
            loss = criterion(outputs, labels)

            # BWD and OPTIM
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % args.log_interval == 0:
                print(
                    f"Train::\tRank: [{rank}/{args.world_size}]\t"
                    f"Epoch: [{epoch + 1}/{args.epochs}]\tStep [{i+1}/{total_num_steps}] "
                    f"({100. * (i + 1) * (epoch + 1) / args.epochs * total_num_steps:.0f}%)\t"
                    f"Loss: {loss.item():.6f}"
                )
                run.log({"loss": loss.item()})

            torch.save(model.state_dict(), './results/model.pth')
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file('./results/model.pth')
            run.log_artifact(artifact)
            torch.save(optimizer.state_dict(), './results/optimizer.pth')
            artifact = wandb.Artifact('optimizer', type='optimizer')
            artifact.add_file('./results/optimizer.pth')
            run.log_artifact(artifact)

    cleanup()
    run.finish()


def main(fn: Callable) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--nodes",
        default=1,
        type=int,
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "-g", "--gpus", default=1, type=int, help="number of gpus per node"
    )
    parser.add_argument(
        "-nr", "--nr", default=0, type=int, help="ranking within the nodes"
    )
    parser.add_argument(
        "--backend",
        default="gloo",
        type=str,
        help="Type of backend to use in torch distributed (default: gloo)",
    )

    parser.add_argument(
        "--init_method",
        default="env://",
        type=str,
        help="Init method for torch distributed (default: env://)",
    )
    parser.add_argument(
        "--epochs",
        default=2,
        type=int,
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        help="batch size of the training data",
    )
    parser.add_argument(
        "--log-interval",
        default=10,
        type=int,
        help="How often to log the resulting optimization step",
    )
    parser.add_argument(
        "--lr",
        default=1e-4,
        type=float,
        help="Learning rate for the optimization step",
    )
    parser.add_argument(
        "--data-path",
        default="./data",
        help="directory path where to download datasets",
    )

    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    args.group_name = datetime.now().strftime("%d-%m-%Y(%H:%M:%S.%f)")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # TODO nprocs should it be world_size or number of gpus?
    torch.multiprocessing.spawn(fn, nprocs=args.gpus, args=(args,), join=True)


if __name__ == "__main__":
    main(train)
