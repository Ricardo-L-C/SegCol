import os, argparse
import urllib3
import shutil
import random
from multiprocessing import Pool
from pathlib import Path
from itertools import cycle

import cv2
from tqdm import tqdm

from utils.xdog_blend import get_xdog_image, add_intensity

SKETCHKERAS_URL = "http://github.com/lllyasviel/sketchKeras/releases/download/0.1/mod.h5"

def make_xdog(img):
    s = 0.35 + 0.1 * random.random()
    k = 2 + random.random()
    g = 0.95
    return get_xdog_image(img, sigma=s, k=k, gamma=g, epsilon=-0.5, phi=10**9)


def download_sketchKeras():
    curr_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    save_path = curr_dir / "utils" / "sketchKeras.h5"

    if save_path.exists():
        print("found sketchKeras.h5")
        return

    print("Downloading sketchKeras...")
    http = urllib3.PoolManager()

    with http.request("GET", SKETCHKERAS_URL, preload_content=False) as r, save_path.open("wb") as out_file:
        shutil.copyfileobj(r, out_file)

    print("Finished downloading sketchKeras.h5")


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def xdog_write(path_img):
    path, xdog_path = path_img
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    xdog_img = make_xdog(img)
    cv2.imwrite(str(xdog_path / path.name), xdog_img)


def exec_keras(dataset_path):
    from utils.sketch_keras_util import batch_keras_enhanced

    keras_path = dataset_path / "keras_train"
    keras_test_path = dataset_path / "keras_test"
    keras_path.mkdir(exist_ok=True)
    keras_test_path.mkdir(exist_ok=True)

    download_sketchKeras()

    print("Extracting sketchKeras of rgb_train")
    img_list = (dataset_path / "rgb_train").iterdir()
    for img_16 in tqdm(chunks(img_list, 16)):
        imgs = list(map(lambda x: cv2.imread(str(x)), img_16))
        krs = batch_keras_enhanced(imgs)

        for p, sketch in zip(img_16, krs):
            sketch = add_intensity(sketch, 1.4)
            cv2.imwrite(str(keras_path / p.name), sketch)

    print("Extracting sketchKeras of benchmark")
    bench_list = (dataset_path / "benchmark").iterdir()
    for img_16 in tqdm(chunks(bench_list, 16)):
        imgs = list(map(lambda x: cv2.imread(str(x)), img_16))
        krs = batch_keras_enhanced(imgs)

        for p, sketch in zip(img_16, krs):
            sketch = add_intensity(sketch, 1.4)
            cv2.imwrite(str(keras_test_path / p.name), sketch)


def exec_xdog(dataset_path):
    xdog_path = dataset_path / "xdog_train"
    xdog_path.mkdir(exist_ok=True)

    print("Extracting XDoG with 32 threads")

    img_list = (dataset_path / "rgb_train").iterdir()
    with Pool(32) as p:
        p.map(xdog_write, zip(img_list, cycle([xdog_path])))


if __name__=="__main__":
    desc = "Seg colorization sketch extractor"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("--dataset", type=Path, default="./dataset", help="path to dataset directory")
    parser.add_argument("--xdog_only", action="store_true")
    parser.add_argument("--keras_only", action="store_true")

    args = parser.parse_args()

    if not args.xdog_only:
        exec_keras(args.dataset_path)

    if not args.keras_only:
        exec_xdog(args.dataset_path)
