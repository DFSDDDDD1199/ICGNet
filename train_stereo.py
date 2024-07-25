
from __future__ import print_function, division

import os

import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
from core.igev_stereo import IGEVStereo
from evaluate_stereo import *
import core.stereo_datasets as datasets
import torch.nn.functional as F
from core import KeypointLoss, DistillLoss, MatchingLoss
from core.superpoint import SuperPoint
from core.matching import Matching
from torchvision.transforms.functional import rgb_to_grayscale
from torchvision.transforms import Grayscale
import cv2
from PIL import Image


ckpt_path = './checkpoints/from_scratch_no_assign'
log_path = './checkpoints/from_scratch_no_assign'

try:
    from torch.cuda.amp import GradScaler
except:
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


def sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, loss_gamma=0.9, max_disp=192):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    disp_loss = 0.0
    mag = torch.sum(disp_gt**2, dim=1).sqrt()
    valid = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()


    disp_loss += 1.0 * F.smooth_l1_loss(disp_init_pred[valid.bool()], disp_gt[valid.bool()], size_average=True)
    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
        disp_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((disp_preds[-1] - disp_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return disp_loss, metrics


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    # other_params = [param for name, param in model.named_parameters() if 'superpoint' not in name]
    # superpoint_params = [param for name, param in model.named_parameters() if 'superpoint' in name]

    # optimizer = optim.AdamW([
    # {'params': other_params, 'lr': args.lr},  # Set the learning rate for part A
    # # {'params': superpoint_params, 'lr': 0.1 * args.lr},  # Set the learning rate for part B``
    # ],weight_decay=args.wdecay, eps=1e-8)

    # stereo_param = []
    # for name, param in model.named_parameters():
    #     if not 'sparse_matching_decoder' in name and not 'descriptor_extractor' in name:
    #         stereo_param.append(param)

    # optimizer = optim.Adam([
    #     {'params': stereo_param, 'lr': args.lr},
    #     {'params': model.module.descriptor_extractor.parameters(), 'lr': args.lr_matching},
    #     {'params': model.module.sparse_matching_decoder.parameters(), 'lr': args.lr_matching}
    # ], lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wdecay, eps=1e-8)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, [args.lr], args.num_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    # scheduler = optim.lr_scheduler.ConstantLR(optimizer, 0.5, total_iters=0)
    return optimizer, scheduler


class Logger:
    SUM_FREQ = 100
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir=log_path)

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=log_path)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=log_path)

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    model = nn.DataParallel(IGEVStereo(args))
    print("Parameter Count: %d" % count_parameters(model))

    sp_model = SuperPoint(config={
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': 1024
            })
    sp_model = nn.DataParallel(sp_model)
    sp_model.cuda()
    sp_model.requires_grad_(False)

    matching_model = Matching(config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.000,
                'max_keypoints': 1024
            },
            'superglue': {
                'weights': 'indoor',
                'sinkhorn_iterations': 100,
                'match_threshold': 0.2,
            }
        }).eval()
    matching_model = nn.DataParallel(matching_model)
    matching_model.cuda()
    matching_model.requires_grad_(False)

    # sift = cv2.SIFT_create(nfeatures=1024)

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    logger = Logger(model, scheduler)

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        model.cuda()
        checkpoint = torch.load(args.restore_ckpt, map_location='cuda')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer = optimizer
        scheduler = scheduler
        start_batch = int(args.restore_ckpt.split('/')[-1].split('_')[0])
        total_steps = start_batch
        logging.info(f"Done loading checkpoint from batch {start_batch}")
    model.cuda()
    model.train()
    model.module.freeze_bn() # We keep BatchNorm frozen

    validation_frequency = 20000

    scaler = GradScaler(enabled=args.mixed_precision)

    should_keep_training = True
    global_batch_num = 0

    keypoint_loss = nn.DataParallel(KeypointLoss())
    distill_loss = (DistillLoss())
    matching_loss = nn.DataParallel(MatchingLoss())

    while should_keep_training:

        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            image1, image2, disp_gt, valid = [x.cuda() for x in data_blob]
            image1_gray, image2_gray = rgb_to_grayscale(image1) / 255.0, rgb_to_grayscale(image2) / 255.0
            
            # get matching result from superpoint + superglue
            pred = matching_model({'image0': image1_gray, 'image1': image2_gray})
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            kpts0_scores, kpts1_scores = pred['scores0'], pred['scores1']
            batched_matches, conf = pred['matches0'], pred['matching_scores0']
            ref_matching_cost = pred['cost']
            ref_matching_score = pred['scores']
            bin_score = matching_model.module.superglue.bin_score
            input_others = {
                'kpts0': kpts0,
                'kpts1': kpts1,
                'kpts0_scores': kpts0_scores,
                'kpts1_scores': kpts1_scores,
            }


            assert model.training
            disp_init_pred, disp_preds, others = model(image1, image2, iters=args.train_iters, input_others=input_others)
            assert model.training
     
            keypoint_est_left = others['keypoint_est_left']
            keypoint_label_left = sp_model({'image': rgb_to_grayscale(image1)/255.0}, return_heatmap=True).unsqueeze(1).long()
            
            # SIFT as detector
            # keypoint_label_left = np.zeros_like(keypoint_est_left.detach().cpu().numpy())[:, 0:1]
            # for i in range(len(keypoint_est_left)):
            #     sift_keypoints_left, _ = sift.detectAndCompute(rgb_to_grayscale(image1)[i].squeeze().to(torch.uint8).detach().cpu().numpy(), None)
            #     for kp in sift_keypoints_left:
            #         keypoint_label_left[i][0][int(kp.pt[1])][int(kp.pt[0])] = 1
            # keypoint_label_left = torch.from_numpy(keypoint_label_left).cuda()

            loss, metrics = sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, max_disp=args.max_disp)
            loss_keypoint = keypoint_loss(keypoint_est_left, keypoint_label_left)
            loss_distill = distill_loss(others['matching_scores'], ref_matching_score)
            loss_matching = matching_loss(others['matching_scores'], kpts0, kpts1, disp_gt.squeeze(1))
            loss = loss + 100 * loss_keypoint.mean() + 0.5 * loss_distill.mean() + 0.5 * loss_matching.mean()
            
            logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
            logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            global_batch_num += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            logger.push(metrics)

            if total_steps % validation_frequency == validation_frequency - 1:
                save_path = Path(ckpt_path + '/%d_%s.pth' % (total_steps + 1, args.name))
                logging.info(f"Saving file {save_path.absolute()}")
                torch.save({'model':model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, save_path)
                results = validate_sceneflow(model.module, iters=args.valid_iters)
                logger.write_dict(results)
                model.train()
                model.module.freeze_bn()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

        if len(train_loader) >= 10000:
            save_path = Path(ckpt_path + '/%d_epoch_%s.pth.gz' % (total_steps + 1, args.name))
            logging.info(f"Saving file {save_path}")
            torch.save(model.state_dict(), save_path)

    print("FINISHED TRAINING")
    logger.close()
    PATH = ckpt_path + '/%s.pth' % args.name
    torch.save({'model':model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='igev-stereo', help="name your experiment")
    parser.add_argument('--restore_ckpt', default=None, help="")
    parser.add_argument('--mixed_precision', type=bool, default=True, help='use mixed precision')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=600000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[320, 736], help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=22, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during validation forward pass')

    # Architecure choices
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=[0, 1.4], help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.2, 0.4], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    args = parser.parse_args()

    torch.manual_seed(666)
    np.random.seed(666)
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    Path(ckpt_path).mkdir(exist_ok=True, parents=True)

    train(args)
