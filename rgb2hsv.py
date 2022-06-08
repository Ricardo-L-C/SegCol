from cmath import tanh
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler

from PIL import Image
from torchvision import transforms

import argparse
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

import random
import numpy as np

class RGB2ColorSpace(object):
    def __call__(self, img):
        return (img * 2 - 1.)

class TrainDataset(data.Dataset):
    def __init__(self, datapath):
        self.datalist = list(datapath.glob("*.png"))
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            RGB2ColorSpace(),
        ])

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        p = self.datalist[index]
        img = Image.open(p)
        rgb = img.convert("RGB")
        hsv = img.convert("HSV")

        return self.transform(rgb), self.transform(hsv)

class ResNext_block(nn.Module):
    def __init__(self):
        super(ResNext_block, self).__init__()

        self.conv1 = nn.Conv2d(64, 32, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 1, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.relu(y)

        y = self.conv3(y)

        y += x
        y = self.relu(y)

        return y

class Rgb2hsv(nn.Module):
    def __init__(self):
        super(Rgb2hsv, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 1, 1)
        self.conv3 = nn.Conv2d(64, 64, 1, 1)
        self.conv4 = nn.Conv2d(64, 64, 1, 1)
        self.conv5 = nn.Conv2d(64, 32, 1, 1)
        self.conv6 = nn.Conv2d(32, 3, 1, 1)
        self.tanh = nn.Tanh()

        self.res_blocks = nn.Sequential(*[ResNext_block() for _ in range(5)])

        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, rgb):
        hsv = self.conv1(rgb)
        hsv = self.relu(hsv)

        hsv = self.conv2(hsv)
        hsv = self.relu(hsv)

        hsv = self.conv3(hsv)
        hsv = self.relu(hsv)

        hsv = self.res_blocks(hsv)

        hsv = self.conv4(hsv)
        hsv = self.relu(hsv)

        hsv = self.conv5(hsv)
        hsv = self.relu(hsv)

        hsv = self.conv6(hsv)
        hsv = self.tanh(hsv)

        return hsv


def get_dataset(args):
    dataset = TrainDataset(args.dataset)
    sampler =  DistributedSampler(dataset, seed=args.seed, drop_last=True)
    loader = data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    return loader

def train(args):
    loader = get_dataset(args)

    device = torch.device("cuda")

    if args.local_rank == 0:
        writer = Path('./rgb2hsv/log')
        if not writer.exists():
            writer.mkdir(parents=True)
        writer = SummaryWriter(str(writer))

    model = Rgb2hsv().to(device)
    opti = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    model.train()

    step = 0

    for epoch in range(args.epoch):
        max_iter = loader.dataset.__len__() // args.batch_size

        for iter, (rgb, hsv) in enumerate(tqdm(loader, ncols=80)):
            rgb, hsv = rgb.to(device), hsv.to(device)

            opti.zero_grad()
            loss = F.l1_loss(model(rgb), hsv)
            loss.backward()
            opti.step()

            if args.local_rank == 0:
                writer.add_scalar("l1", loss.item(), step)
                step += 1

        if args.local_rank == 0:
            print(f"epoch: {epoch + 1} done")
            save(model, epoch + 1)


def save(model, epoch):
    p = Path("./rgb2hsv/")
    p.mkdir(exist_ok=True)

    torch.save({
        "rgb2hsv": model.module.state_dict(),
        "epoch": epoch
    }, str(p / f"rgb2hsv3_{epoch}_epoch.pth"))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument("--dataset", type=Path, default="./dataset/rgb_train")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam optimizer parameter")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam optimizer parameter")
    parser.add_argument("--seed", type=int, default=1, help="Manual random seed")

    args = parser.parse_args()

    set_seed(args.seed)

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl')

    train(args)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    main()