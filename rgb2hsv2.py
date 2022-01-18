from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data as data

from PIL import Image
from torchvision import transforms

import argparse
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

class TrainDataset(data.Dataset):
    def __init__(self, datapath, transform=None):
        self.datalist = list(datapath.glob("*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        p = self.datalist[index]
        img = Image.open(p)
        rgb = img.convert("RGB")
        hsv = img.convert("HSV")

        if self.transform:
            rgb, hsv = self.transform(rgb), self.transform(hsv)

        return rgb, hsv

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

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

        self.res_blocks = nn.Sequential(*[ResNext_block() for _ in range(3)])

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64+64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(32+32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(16+16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
        )

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

        enc1 = self.enc1(rgb)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)

        hsv = self.res_blocks(enc3)

        hsv = self.dec1(torch.cat([hsv, enc3], 1))
        hsv = self.dec2(torch.cat([hsv, enc2], 1))
        hsv = self.dec3(torch.cat([hsv, enc1], 1))

        return hsv


def get_dataset(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset = TrainDataset(args.dataset, transform)
    loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return loader

def train(args):
    loader = get_dataset(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.autograd.set_detect_anomaly(True)

    writer = Path('./rgb2hsv2/log')
    if not writer.exists():
        writer.mkdir(parents=True)
    writer = SummaryWriter(str(writer))

    model = nn.DataParallel(Rgb2hsv()).to(device)
    l1 = nn.L1Loss().to(device)
    # opti = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opti = torch.optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(args.epoch):
        print(f"epoch: {epoch + 1}")

        model.train()

        max_iter = loader.dataset.__len__() // args.batch_size

        for iter, (rgb, hsv) in enumerate(tqdm(loader, ncols=80)):
            if iter >= max_iter:
                break

            rgb, hsv = rgb.to(device), hsv.to(device)

            opti.zero_grad()
            loss = l1(model(rgb), hsv)
            loss.backward()
            opti.step()

            writer.add_scalar("l1", loss.item(), iter+epoch*max_iter)

        save(model, epoch)



def save(model, epoch):
    p = Path("./rgb2hsv2/")

    if not p.exists():
        p.mkdir()

    torch.save({
        "rgb2hsv2": model.state_dict(),
        "epoch": epoch
    }, str(p / f"rgb2hsv2_{epoch}_epoch.pkl"))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="./dataset/rgb_train")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam optimizer parameter")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam optimizer parameter")

    args = parser.parse_args()
    args.dataset = Path(args.dataset)

    train(args)


if __name__ == "__main__":
    main()