from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from tqdm import tqdm

import utils
from loader.dataloader import get_dataset, get_tag_dict, ColorSpace2RGB
from network import Discriminator
from model.pretrained import pretrain_rgb2hsv, se_resnext_half, Vgg16

from torch.utils.tensorboard import SummaryWriter

class tag2pix(object):
    def __init__(self, args):
        if args.model == 'tag2pix':
            from network import Generator
        elif args.model == 'senet':
            from model.GD_senet import Generator
        elif args.model == 'resnext':
            from model.GD_resnext import Generator
        elif args.model == 'catconv':
            from model.GD_cat_conv import Generator
        elif args.model == 'catall':
            from model.GD_cat_all import Generator
        elif args.model == 'adain':
            from model.GD_adain import Generator
        elif args.model == 'seadain':
            from model.GD_seadain import Generator
        else:
            raise Exception('invalid model name: {}'.format(args.model))

        self.args = args
        self.epoch = args.epoch
        self.batch_size = args.batch_size

        self.input_size = args.input_size
        self.color_revert = ColorSpace2RGB(args.color_space)
        self.layers = args.layers
        [self.cit_weight, self.cvt_weight] = args.cit_cvt_weight

        self.load_dump = (args.load is not "")

        self.load_path = Path(args.load)

        self.l1_lambda = args.l1_lambda
        self.guide_beta = args.guide_beta
        self.adv_lambda = args.adv_lambda
        self.save_freq = args.save_freq

        self.two_step_epoch = args.two_step_epoch
        self.brightness_epoch = args.brightness_epoch
        self.save_all_epoch = args.save_all_epoch

        self.iv_dict, self.cv_dict, self.id_to_name = get_tag_dict(args.tag_dump)

        cvt_class_num = len(self.cv_dict.keys())
        cit_class_num = len(self.iv_dict.keys())
        self.class_num = cvt_class_num + cit_class_num

        self.start_epoch = 1
        self.end_epoch = self.epoch

        self.result_path = args.result_dir

        self.device = torch.device('cuda', args.local_rank)
        torch.cuda.set_device(args.local_rank)

        #### load dataset
        if not args.test:
            torch.distributed.init_process_group(backend='nccl')
            self.train_data_loader, self.test_data_loader = get_dataset(args)

            if self.args.local_rank == 0:
                self.test_images = self.get_test_data(self.test_data_loader, args.test_image_count)

                logger_path = self.result_path / "logger"
                logger_path.mkdir(exist_ok=True)

                self.logger = SummaryWriter(str(logger_path))

        else:
            self.test_data_loader = get_dataset(args)


        ##### initialize network
        self.net_opt = {
            'guide': not args.no_guide,
            'relu': args.use_relu,
            'bn': not args.no_bn,
            'cit': not args.no_cit
        }

        if self.net_opt['cit']:
            self.ResNeXT = se_resnext_half(dump_path=args.pretrain_dump, num_classes=cit_class_num, input_channels=1)
        else:
            self.ResNeXT = nn.Sequential()
        self.ResNeXT.to(self.device, non_blocking=True)

        self.D_input_dim = 3

        if self.args.dual_color_space:
            self.rgb2hsv = pretrain_rgb2hsv().to(self.device, non_blocking=True)
            self.D_input_dim = 6

        self.vgg16 = Vgg16().to(self.device, non_blocking=True)

        self.G = Generator(input_size=args.input_size, layers=args.layers, cv_class_num=cvt_class_num, iv_class_num=cit_class_num, net_opt=self.net_opt, dual_branch=self.args.dual_branch, direct_cat=self.args.direct_cat, link=self.args.link_color).to(self.device, non_blocking=True)
        self.D = Discriminator(input_dim=self.D_input_dim, output_dim=1, input_size=self.input_size, cv_class_num=cvt_class_num, iv_class_num=cit_class_num, link=self.args.link_color).to(self.device, non_blocking=True)

        for param in self.ResNeXT.parameters():
            param.requires_grad = False
        if args.test:
            for param in self.G.parameters():
                param.requires_grad = False
            for param in self.D.parameters():
                param.requires_grad = False

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        #if args.load != Path():
        #    self.load(args.load)

        if not self.args.test:
            self.G = torch.nn.parallel.DistributedDataParallel(self.G, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True, broadcast_buffers=False)
            self.D = torch.nn.parallel.DistributedDataParallel(self.D, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True, broadcast_buffers=False)

        print("device: ", self.device)

    def train(self):
        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1).to(self.device, non_blocking=True), torch.zeros(self.batch_size, 1).to(self.device, non_blocking=True)

        if self.args.local_rank == 0:
            self.print_params()
            print('training start!!')

        self.D.train()

        step = 0
        max_iter = self.train_data_loader.dataset.__len__() // self.batch_size

        for epoch in range(self.start_epoch, self.end_epoch + 1):
            if self.args.local_rank == 0:
                print("EPOCH: {}".format(epoch))

            self.G.train()

            if epoch == self.brightness_epoch:
                print('changing brightness ...')
                self.train_data_loader.dataset.enhance_brightness(self.input_size)


            for iter, (original_, sketch_, skeleton_, iv_tag_, cv_tag_, link_) in enumerate(tqdm(self.train_data_loader, ncols=80)):

                sketch_, original_, skeleton_, iv_tag_, cv_tag_, link_ = sketch_.to(self.device, non_blocking=True), original_.to(self.device, non_blocking=True), skeleton_.to(self.device, non_blocking=True), iv_tag_.to(self.device, non_blocking=True), cv_tag_.to(self.device, non_blocking=True), link_.to(self.device, non_blocking=True)

                with torch.no_grad():
                    feature_tensor = self.ResNeXT(sketch_)

                # update D network
                self.D_optimizer.zero_grad()
                D_loss = 0

                if self.args.link_color:
                    cv_tag_ = link_

                if self.args.direct_cat:
                    direct_cat = torch.cat((sketch_, skeleton_), 1)
                    G_f = self.G(direct_cat, feature_tensor, cv_tag_, link_, 1)
                else:
                    G_f = self.G(sketch_, feature_tensor, cv_tag_, link_, 1)

                if self.args.dual_color_space:
                    hsv_gt = self.rgb2hsv(original_)
                    gt = torch.cat([original_, hsv_gt], 1)

                    hsv_f = self.rgb2hsv(G_f)
                    G_f = torch.cat([G_f, hsv_f], 1)
                else:
                    gt = original_

                if self.two_step_epoch == 0 or epoch >= self.two_step_epoch:
                    D_real, CIT_real, CVT_real = self.D(gt)

                    CIT_real_loss = F.binary_cross_entropy(CIT_real, iv_tag_) if self.net_opt['cit'] else 0
                    CVT_real_loss = F.binary_cross_entropy(CVT_real, cv_tag_)
                    C_real_loss = self.cvt_weight * CVT_real_loss + self.cit_weight * CIT_real_loss

                    D_f_fake, CIT_f_fake, CVT_f_fake = self.D(G_f)

                    CIT_f_fake_loss = F.binary_cross_entropy(CIT_f_fake, iv_tag_) if self.net_opt['cit'] else 0
                    CVT_f_fake_loss = F.binary_cross_entropy(CVT_f_fake, cv_tag_)
                    C_fake_loss = self.cvt_weight * CVT_f_fake_loss + self.cit_weight * CIT_f_fake_loss

                    D_loss += C_real_loss + C_fake_loss
                else:
                    D_real = self.D(gt, 1)
                    D_f_fake = self.D(G_f, 1)

                D_real_loss = F.binary_cross_entropy(D_real, self.y_real_)
                D_fake_loss = F.binary_cross_entropy(D_f_fake, self.y_fake_)

                D_loss += self.adv_lambda * (D_real_loss + D_fake_loss)

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()
                G_loss = 0

                if self.args.dual_branch:
                    G_f, G_g, G_s = self.G(sketch_, feature_tensor, cv_tag_, link_)
                    skeleton_l1_loss = F.l1_loss(G_s, skeleton_)
                    G_loss += skeleton_l1_loss
                    if self.args.local_rank == 0:
                        self.logger.add_scalar("skeleton_l1_loss", skeleton_l1_loss.item(), step)
                elif self.args.direct_cat:
                    direct_cat = torch.cat((sketch_, skeleton_), 1)
                    G_f, G_g = self.G(direct_cat, feature_tensor, cv_tag_, link_)
                else:
                    G_f, G_g = self.G(sketch_, feature_tensor, cv_tag_, link_)

                if not self.args.no_guide:
                    guide_l1_loss = F.l1_loss(G_g, original_)
                    G_loss += guide_l1_loss * self.guide_beta

                    if self.args.local_rank == 0:
                        self.logger.add_scalar("guide_l1_loss", guide_l1_loss.item(), step)

                if self.args.dual_color_space:
                    perceptual_g, perceptual_gt = self.vgg16(G_f), self.vgg16(original_)
                    rgb_perceptual_loss = F.mse_loss(perceptual_g, perceptual_gt)

                    hsv_fake, hsv_real = self.rgb2hsv(G_f), self.rgb2hsv(original_)
                    hsv_l1_loss = F.l1_loss(hsv_fake, hsv_real)

                    G_loss += rgb_perceptual_loss + hsv_l1_loss

                    if self.args.local_rank == 0:
                        self.logger.add_scalar("rgb_perceptual_loss", rgb_perceptual_loss.item(), step)
                        self.logger.add_scalar("hsv_l1_loss", hsv_l1_loss.item(), step)

                    G_f = torch.cat([G_f, hsv_fake], 1)
                else:
                    rgb_l1_loss = F.l1_loss(G_f, original_)
                    G_loss += rgb_l1_loss

                    if self.args.local_rank == 0:
                        self.logger.add_scalar("rgb_l1_loss", rgb_l1_loss.item(), step)

                G_loss *= self.l1_lambda

                if self.two_step_epoch == 0 or epoch >= self.two_step_epoch:
                    D_f_fake, CIT_f_fake, CVT_f_fake = self.D(G_f)

                    CIT_fake_loss = F.binary_cross_entropy(CIT_f_fake, iv_tag_) if self.net_opt['cit'] else 0
                    CVT_fake_loss = F.binary_cross_entropy(CVT_f_fake, cv_tag_)

                    C_fake_loss = self.cvt_weight * CVT_fake_loss + self.cit_weight * CIT_fake_loss
                    G_loss += C_fake_loss

                    if self.args.local_rank == 0:
                        self.logger.add_scalar("C_fake_loss", C_fake_loss.item(), step)
                else:
                    D_f_fake = self.D(G_f, 1)

                D_fake_loss = F.binary_cross_entropy(D_f_fake, self.y_real_)
                G_loss += D_fake_loss

                G_loss.backward()
                self.G_optimizer.step()

                if self.args.local_rank == 0:
                    self.logger.add_scalar("G_loss", G_loss.item(), step)
                    self.logger.add_scalar("D_fake_loss", D_fake_loss.item(), step)


                if self.args.local_rank == 0 and epoch == 1 and (iter + 1)  == 100:
                    self.visualize_results(epoch)

                if self.args.local_rank == 0 and (iter + 1) % 500 == 0:
                    self.visualize_results(epoch)
                    print("Epoch: [{:2d}] [{:4d}/{:4d}] D_loss: {:.8f}, G_loss: {:.8f}".format(epoch, (iter + 1), max_iter, D_loss.item(), G_loss.item()))

                step += 1

            if self.args.local_rank == 0:
                self.visualize_results(epoch)

                if epoch >= self.save_all_epoch > 0:
                    self.save(epoch)
                elif self.save_freq > 0 and epoch % self.save_freq == 0:
                    self.save(epoch)

        if self.args.local_rank == 0:
            print("Training finish!... save training results")

    def test(self):
        self.load_test(self.args.load)

        self.D.eval()
        self.G.eval()

        load_path = self.load_path
        result_path = self.result_path / load_path.stem

        result_path.mkdir(exist_ok=True)

        with torch.no_grad():
            for sketch_, skeleton_, index_, _, cv_tag_, link_ in tqdm(self.test_data_loader, ncols=80):
                sketch_, skeleton_, cv_tag_, link_ = sketch_.to(self.device, non_blocking=True), skeleton_.to(self.device, non_blocking=True), cv_tag_.to(self.device, non_blocking=True), link_.to(self.device, non_blocking=True)

                with torch.no_grad():
                    feature_tensor = self.ResNeXT(sketch_)

                if self.args.direct_cat:
                    direct_cat = torch.cat((sketch_, skeleton_), 1)
                    G_f = self.G(direct_cat, feature_tensor, cv_tag_, link_, 1)
                else:
                    G_f = self.G(sketch_, feature_tensor, cv_tag_, link_, 1)
                G_f = self.color_revert(G_f.cpu())

                for ind, result in zip(index_.cpu().numpy(), G_f):
                    save_path = result_path / f'{ind}.png'
                    if save_path.exists():
                        for i in range(100):
                            save_path = result_path / f'{ind}_{i}.png'
                            if not save_path.exists():
                                break
                    img = Image.fromarray(result)
                    img.save(save_path)

    def visualize_results(self, epoch):
        self.result_path.mkdir(exist_ok=True)

        with torch.no_grad():
            self.G.eval()

            G_f, G_g, G_s = [], [], []

            for i, (_, sketch, skeleton_, _, cv_tag, link) in enumerate(self.test_data_loader):
                sketch, skeleton_, cv_tag, link = sketch.to(self.device, non_blocking=True), skeleton_.to(self.device, non_blocking=True), cv_tag.to(self.device, non_blocking=True), link.to(self.device, non_blocking=True)

                feature_tensor = self.ResNeXT(sketch)

                if self.args.dual_branch:
                    f, g, s = self.G(sketch, feature_tensor, cv_tag, link)
                    G_s.append(s.cpu())
                elif self.args.direct_cat:
                    direct_cat = torch.cat((sketch, skeleton_), 1)
                    f, g = self.G(direct_cat, feature_tensor, cv_tag, link)
                else:
                    f, g = self.G(sketch, feature_tensor, cv_tag, link)
                G_f.append(f.cpu())
                G_g.append(g.cpu())

                if i >= 63:
                    break

            G_f = self.color_revert(torch.cat(G_f, 0))
            G_g = self.color_revert(torch.cat(G_g, 0))
            if self.args.dual_branch:
                G_s = self.color_revert(torch.cat(G_s, 0))

            image_frame_dim = int(np.ceil(np.sqrt(64)))

            utils.save_images(G_f[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim], self.result_path / 'tag2pix_epoch{:03d}_G_f.png'.format(epoch))
            utils.save_images(G_g[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim], self.result_path / 'tag2pix_epoch{:03d}_G_g.png'.format(epoch))
            if self.args.dual_branch:
                utils.save_images(G_s[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim], self.result_path / 'tag2pix_epoch{:03d}_G_s.png'.format(epoch))

    def save(self, save_epoch):
        torch.save({
            'G' : self.G.module.state_dict(),
            'D' : self.D.module.state_dict(),
            'G_optimizer' : self.G_optimizer.state_dict(),
            'D_optimizer' : self.D_optimizer.state_dict(),
            'finish_epoch' : save_epoch,
            'result_path' : str(self.result_path)
            }, str(self.result_path / 'tag2pix_{}_epoch.pkl'.format(save_epoch)))

        print("============= save success =============")
        print("epoch from {} to {}".format(self.start_epoch, save_epoch))
        print("save result path is {}".format(str(self.result_path)))

    def load_test(self, checkpoint_path):
        checkpoint = torch.load(str(checkpoint_path), map_location=torch.device("cpu"))
        self.G.load_state_dict(checkpoint['G'])

    def load(self, checkpoint_path):
        checkpoint = torch.load(str(checkpoint_path), map_location=torch.device("cpu"))
        self.G.load_state_dict(checkpoint['G'])
        self.D.load_state_dict(checkpoint['D'])
        self.G_optimizer.load_state_dict(checkpoint['G_optimizer'])
        self.D_optimizer.load_state_dict(checkpoint['D_optimizer'])
        self.start_epoch = checkpoint['finish_epoch'] + 1

        self.end_epoch = self.args.epoch + self.start_epoch - 1

        print("============= load success =============")
        print("epoch start from {} to {}".format(self.start_epoch, self.end_epoch))
        print("previous result path is {}".format(checkpoint['result_path']))


    def get_test_data(self, test_data_loader, count):
        test_count = 0
        original_, sketch_, iv_tag_, cv_tag_, link_ = [], [], [], [], []
        for orig, sket, _, ivt, cvt, link in test_data_loader:
            original_.append(orig)
            sketch_.append(sket)
            iv_tag_.append(ivt)
            cv_tag_.append(cvt)
            link_.append(link)

            test_count += len(orig)
            if test_count >= count:
                break

        original_ = torch.cat(original_, 0)
        sketch_ = torch.cat(sketch_, 0)
        iv_tag_ = torch.cat(iv_tag_, 0)
        cv_tag_ = torch.cat(cv_tag_, 0)
        link_ = torch.cat(link_, 0)

        image_frame_dim = int(np.ceil(np.sqrt(len(original_))))

        original_ = original_.cpu()
        sketch_np = sketch_.data.numpy().transpose(0, 2, 3, 1)
        original_np = self.color_revert(original_)

        self.save_tag_tensor_name(iv_tag_, cv_tag_, self.result_path / "test_image_tags.txt")
        utils.save_images(original_np[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim], self.result_path / 'tag2pix_original.png')
        utils.save_images(sketch_np[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim], self.result_path / 'tag2pix_sketch.png')

        return original_, sketch_, iv_tag_, cv_tag_, link_


    def save_tag_tensor_name(self, iv_tensor, cv_tensor, save_file_path):
        '''iv_tensor, cv_tensor: batched one-hot tag tensors'''
        iv_dict_inverse = {tag_index: tag_id for (tag_id, tag_index) in self.iv_dict.items()}
        cv_dict_inverse = {tag_index: tag_id for (tag_id, tag_index) in self.cv_dict.items()}

        with open(save_file_path, 'w') as f:
            f.write("CIT tags\n")

            for tensor_i, batch_unit in enumerate(iv_tensor):
                tag_list = []
                f.write(f'{tensor_i} : ')

                for i, is_tag in enumerate(batch_unit):
                    if is_tag:
                        tag_name = self.id_to_name[iv_dict_inverse[i]]
                        tag_list.append(tag_name)
                        f.write(f"{tag_name}, ")
                f.write("\n")

            f.write("\nCVT tags\n")

            for tensor_i, batch_unit in enumerate(cv_tensor):
                tag_list = []
                f.write(f'{tensor_i} : ')

                for i, is_tag in enumerate(batch_unit):
                    if is_tag:
                        tag_name = self.id_to_name[cv_dict_inverse[i]]
                        tag_list.append(self.id_to_name[cv_dict_inverse[i]])
                        f.write(f"{tag_name}, ")
                f.write("\n")

    def print_params(self):
        params_cnt = [0, 0, 0]
        for param in self.G.parameters():
            params_cnt[0] += param.numel()
        for param in self.D.parameters():
            params_cnt[1] += param.numel()
        for param in self.ResNeXT.parameters():
            params_cnt[2] += param.numel()
        print(f'Parameter #: G - {params_cnt[0]} / D - {params_cnt[1]} / Pretrain - {params_cnt[2]}')
