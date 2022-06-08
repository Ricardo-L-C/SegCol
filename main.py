import argparse
import pprint
import time
from pathlib import Path

import torch
import random
import numpy as np

from tag2pix import tag2pix

root_path = Path(__file__).resolve().parent
dataset_path = root_path / "dataset"
tag_dump_path = root_path / "loader" / "tag_dump.pkl"
pretrain_path = root_path / "model.pth"

def parse_args():
    desc = "Segmentation colorization"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("--model", type=str, default="tag2pix", choices=["tag2pix", "senet", "resnext", "catconv", "catall", "adain", "seadain"], help="Model Types. (default: tag2pix == SECat)")

    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument("--cpu", action="store_true", help="If set, use cpu only")
    parser.add_argument("--test", action="store_true", help="Colorize line arts in test_dir based on `tag_txt`")
    parser.add_argument("--save_freq", type=int, default=10, help="Save network dump by every `save_freq` epoch. if set to 0, save the last result only")

    parser.add_argument("--num_workers", type=int, default=8, help="total thread count of data loader")
    parser.add_argument("--epoch", type=int, default=50, help="The number of epochs to run")
    parser.add_argument("--save_all_epoch", type=int, default=0, help="If nonzero, save network dump by every epoch after this epoch")

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for single GPU")
    parser.add_argument("--input_size", type=int, default=256, help="Width / Height of input image (must be rectangular)")
    parser.add_argument("--data_size", default=0, type=int, help="Total training image count. if 0, use all train data")
    parser.add_argument("--test_image_count", type=int, default=64, help="Total count of colorizing test images")

    parser.add_argument("--data_dir", type=Path, default=dataset_path, help="Path to the train/test data root directory")
    parser.add_argument("--test_dir", type=str, default="liner_test", help="Directory name of the test line art directory. It has to be in the data_dir.")
    parser.add_argument("--tag_txt", type=str, default="tags.txt", help="Text file name of formatted text tag sets (see README). It has to be in the data_dir.")

    parser.add_argument("--result_dir", type=Path, default="./results", help="Path to save generated images and network dump")
    parser.add_argument("--pretrain_dump", type=Path, default=pretrain_path, help="Path of pretrained CIT network dump.")
    parser.add_argument("--tag_dump", type=Path, default=tag_dump_path, help="Path of tag dictionary / metadata pickle file.")
    parser.add_argument("--load", type=str, default="", help="Path to load network weights (if non-empty)")

    parser.add_argument("--lrG", type=float, default=0.0002, help="Learning rate of generator")
    parser.add_argument("--lrD", type=float, default=0.0002, help="Learning rate of discriminator")
    parser.add_argument("--l1_lambda", type=float, default=1000, help="Coefficient of content loss")
    parser.add_argument("--guide_beta", type=float, default=0.9, help="Coefficient of guide decoder")
    parser.add_argument("--adv_lambda", type=float, default=1, help="Coefficient of adversarial loss")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam optimizer parameter")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam optimizer parameter")
    parser.add_argument("--color_space", type=str, default="rgb", choices=["lab", "rgb", "hsv"], help="color space of images")
    parser.add_argument("--layers", type=int, nargs="+", default=[12, 8, 5, 5],
        help="Block counts of each U-Net Decoder blocks of generator. The first argument is count of bottom block.")

    parser.add_argument("--cit_cvt_weight", type=float, nargs="+", default=[1, 1], help="CIT CVT Loss weight. space-separated")
    parser.add_argument("--two_step_epoch", type=int, default=0, help="If nonzero, apply two-step train. (start_epoch to args.auto_two_step_epoch: cit_cvt_weight==[0, 0], after: --cit_cvt_weight)")
    parser.add_argument("--brightness_epoch", type=int, default=0, help="If nonzero, control brightness after this epoch (see Section 4.3.3) (start_epoch to bright_down_epoch: ColorJitter.brightness == 0.2, after: [1, 7])")

    parser.add_argument("--use_relu", action="store_true", help="Apply ReLU to colorFC")
    parser.add_argument("--no_bn", action="store_true", help="Remove every BN Layer from Generator")
    parser.add_argument("--no_guide", action="store_true", help="Remove guide decoder from Generator. If set, Generator will return same G_f: like (G_f, G_f)")
    parser.add_argument("--no_cit", action="store_true", help="Remove pretrain CIT Network from Generator")

    parser.add_argument("--link_color", action="store_true", help="Link color of different tags")
    parser.add_argument("--dual_color_space", action="store_true", help="Use RGB and HSV color space")
    parser.add_argument("--dual_branch", action="store_true", help="Use dual-branch network")
    parser.add_argument("--direct_cat", action="store_true", help="Use direct concatenation")

    parser.add_argument("--seed", type=int, default=-1, help="if positive, apply random seed")

    parser.add_argument("--wgan", action="store_true", help="Use wGAN loss")
    parser.add_argument("--weight_limit", type=float, default=0.01, help="Use wGAN loss")

    args = parser.parse_args()

    return args


def validate_args(args):
    print("validating arguments...")

    assert args.epoch >= 1, "number of epochs must be larger than or equal to one"
    assert args.batch_size >= 1, "batch size must be larger than or equal to one"

    if args.load != "":
        args.load = Path(args.load)
        assert args.load.exists(), "cannot find network dump file"
    assert args.pretrain_dump.exists(), "cannot find pretrained CIT network dump file"
    assert args.tag_dump.exists(), "cannot find tag metadata pickle file"

    assert args.data_dir.exists(), "cannot find train/test root directory"
    assert (args.data_dir / args.tag_txt).exists(), "cannot find formatted text tag file"

    assert args.seed > 0, "Must assign a seed manually"

    if not args.test:
        args.result_dir = args.result_dir / time.strftime('%y%m%d-%H%M%S', time.localtime())

    args.result_dir.mkdir(parents=True, exist_ok=True)

    with open(args.result_dir / "args.txt", "w") as f:
        f.write(pprint.pformat(vars(args)))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()

    if args.local_rank == 0:
        validate_args(args)

    set_seed(args.seed)

    gan = tag2pix(args)

    if args.test:
        gan.test()
        print(" [*] Testing finished!")
    else:
        gan.train()
        print(" [*] Training finished!")


if __name__ == "__main__":
    main()
