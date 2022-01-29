import argparse
import json
import random
import pickle
import shutil
from pathlib import Path

from tqdm import tqdm
import numpy as np
import cv2

def is_white(vect):
    return np.all(vect > 250)

def make_square_by_mirror(img, orig_h, orig_w):
    # img shape should be square
    is_bgr = len(img.shape) == 3
    img_h, img_w = img.shape[:2]

    if orig_h > orig_w:
        h = img_h
        w = int(orig_w * (img_h / orig_h)) - 2
    else:
        w = img_w
        h = int(orig_h * (img_w / orig_w)) - 2

    crop_h = (img_h - h) // 2
    crop_w = (img_w - w) // 2

    img = img[crop_h:crop_h+h, crop_w:crop_w+w]
    diff = max(img_h - img.shape[0], img_w - img.shape[1])

    pad_l = diff // 2
    pad_r = diff - pad_l
    if is_bgr:
        if h > w:
            pad_width = ((0, 0), (pad_l, pad_r), (0, 0))
        else:
            pad_width = ((diff, 0), (0, 0), (0, 0)) # do not reflect bottom part of torso
    else:
        if h > w:
            pad_width = ((0, 0), (pad_l, pad_r))
        else:
            pad_width = ((diff, 0), (0, 0))

    if h != w:
        return np.pad(img, pad_width, mode="symmetric")
    else:
        return img


def crop_all(dataset_path):
    train_base = dataset_path / "train_image_base"
    test_base = dataset_path / "liner_test_base"
    save_train = dataset_path / "rgb_train"
    save_test = dataset_path / "liner_test"
    unknow_resolution = dataset_path / "unknow_resolution"

    save_train.mkdir(exist_ok=True)
    save_test.mkdir(exist_ok=True)
    unknow_resolution.mkdir(exist_ok=True)

    with (dataset_path / "resolutions.json").open() as f:
        resolutions = json.load(f)

    print("cropping train_image_base...")
    for f in tqdm(train_base.iterdir()):
        file_id = int(f.stem)
        if file_id not in resolutions:
            print(f"{file_id} not found in resolutions.json")
            shutil.copy2(f, unknow_resolution)
            continue

        w, h = resolutions[file_id]

        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        cropped_img = make_square_by_mirror(img, h, w)
        cv2.imwrite(str(save_train / (f.with_suffix(".png"))), cropped_img)

    print("cropping liner_test...")
    for f in tqdm(test_base.iterdir()):
        file_id = int(f.stem)
        if file_id not in resolutions:
            print(f"{file_id} not found in resolutions.json")
            w, h = 512, 512
        else:
            w, h = resolutions[file_id]

        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        cropped_img = make_square_by_mirror(img, h, w)
        cv2.imwrite(str(save_test / (f.with_suffix(".png"))), cropped_img)

    benchmark_dir = dataset_path / "benchmark"
    benchmark_dir.mkdir(exist_ok=True)

    img_list = random.sample(save_train.iterdir(), 256)
    for f in img_list:
        shutil.move(f, benchmark_dir / f.name)

    shutil.rmtree(train_base)
    shutil.rmtree(test_base)


def upscale_lanczos_all(image_base, save_path):
    print("upscaling with lanczos...")
    for f in tqdm(image_base.iterdir()):
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        img_up = cv2.resize(img, (768, 768), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(save_path / f.name), img_up)


def upscale_all(dataset_path, image_base, save_path):
    image_base = dataset_path / "rgb_train"
    save_path = dataset_path / "rgb_train_temp"

    save_path.mkdir(exist_ok=True)

    upscale_lanczos_all(image_base, save_path)

    shutil.rmtree(image_base)
    save_path.rename(image_base)


if __name__=="__main__":
    desc = "Seg colorization crop and upscale"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("--dataset_path", type=Path, default="./dataset", help="path to dataset directory")
    parser.add_argument("--crop_only", action="store_true", help="only makes cropped image")
    parser.add_argument("--upscale_only", action="store_true", help="only makes upscaled image")

    args = parser.parse_args()

    if not args.upscale_only:
        crop_all(args.dataset_path)
    if not args.crop_only:
        upscale_all(args.dataset_path)

