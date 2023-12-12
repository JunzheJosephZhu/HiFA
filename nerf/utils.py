import os
import glob
import tqdm
import math
import imageio
import random
import warnings
import tensorboardX
from torchvision.utils import save_image
import numpy as np

import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

import trimesh
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver

def eff_distloss_native(w, m, interval):
    '''
    Efficient O(N) realization of distortion loss.
    There are B rays each with N sampled points.
    w:        Float tensor in shape [B,N]. Volume rendering weights of each point.
    m:        Float tensor in shape [B,N]. Midpoint distance to camera of each point.
    interval: Scalar or float tensor in shape [B,N]. The query interval of each point.
    '''
    loss_uni = (1/3) * (interval * w.pow(2)).sum(dim=-1).mean()
    wm = (w * m)
    w_cumsum = w.cumsum(dim=-1)
    wm_cumsum = wm.cumsum(dim=-1)
    loss_bi_0 = wm[..., 1:] * w_cumsum[..., :-1]
    loss_bi_1 = w[..., 1:] * wm_cumsum[..., :-1]
    loss_bi = 2 * (loss_bi_0 - loss_bi_1).sum(dim=-1).mean()
    return loss_bi + loss_uni

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None, flip_z=False):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device))
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0: # subsample some rays
        N = min(N, H*W)

        if error_map is None:
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])

    zs = - torch.ones_like(i)
    if flip_z:
        zs = -zs
    xs = - (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    # directions = safe_normalize(directions)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results

def get_rays_syncdreamer(poses, intrinsics, H, W, N=-1, error_map=None, flip_z=False):
    '''
    poses: world2cam(=pose of world in camera)
    compared to get rays: flip z axis of camera coord, switch yz of world. No adding 0.5
    '''
    image_num = poses.shape[0]
    h, w = H, W
    coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1)[:, :, (1, 0)]  # h,w,2
    coords = coords.float()[None, :, :, :].repeat(image_num, 1, 1, 1)  # imn,h,w,2
    coords = coords.reshape(image_num, h * w, 2)
    coords = torch.cat([coords, torch.ones(image_num, h * w, 1, dtype=torch.float32)], 2)  # imn,h*w,3
    coords = coords.to(poses.device)

    # imn,h*w,3 @ imn,3,3 => imn,h*w,3
    rays_d = coords @ torch.inverse(intrinsics).permute(0, 2, 1)
    R, t = poses[:, :, :3], poses[:, :, 3:]
    rays_d = rays_d @ R
    rays_d = F.normalize(rays_d, dim=-1)
    rays_o = -R.permute(0,2,1) @ t # imn,3,3 @ imn,3,1
    rays_o = rays_o.permute(0, 2, 1).repeat(1, h*w, 1) # imn,h*w,3

    results = {}
    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results



def adjust_text_embeddings(embeddings, azimuth, opt):
    text_z_list = []
    weights_list = []
    K = 0
    for b in range(azimuth.shape[0]):
        text_z_, weights_ = get_pos_neg_text_embeddings(embeddings, azimuth[b], opt)
        K = max(K, weights_.shape[0])
        text_z_list.append(text_z_)
        weights_list.append(weights_)

    # Interleave text_embeddings from different dirs to form a batch
    text_embeddings = []
    for i in range(K):
        for text_z in text_z_list:
            # if uneven length, pad with the first embedding
            text_embeddings.append(text_z[i] if i < len(text_z) else text_z[0])
    text_embeddings = torch.stack(text_embeddings, dim=0) # [B * K, 77, 768]

    # Interleave weights from different dirs to form a batch
    weights = []
    for i in range(K):
        for weights_ in weights_list:
            weights.append(weights_[i] if i < len(weights_) else torch.zeros_like(weights_[0]))
    weights = torch.stack(weights, dim=0) # [B * K]
    return text_embeddings, weights

def get_pos_neg_text_embeddings(embeddings, azimuth_val, opt):
    if azimuth_val >= -90 and azimuth_val < 90:
        if azimuth_val >= 0:
            r = 1 - azimuth_val / 90
        else:
            r = 1 + azimuth_val / 90
        start_z = embeddings['front']
        end_z = embeddings['side']
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['front'], embeddings['side']], dim=0)
        if r > 0.8:
            front_neg_w = 0.0
        else:
            front_neg_w = math.exp(-r * opt.front_decay_factor) * opt.negative_w
        if r < 0.2:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-(1-r) * opt.side_decay_factor) * opt.negative_w

        weights = torch.tensor([1.0, front_neg_w, side_neg_w])
    else:
        if azimuth_val >= 0:
            r = 1 - (azimuth_val - 90) / 90
        else:
            r = 1 + (azimuth_val + 90) / 90
        start_z = embeddings['side']
        end_z = embeddings['back']
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['side'], embeddings['front']], dim=0)
        front_neg_w = opt.negative_w 
        if r > 0.8:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-r * opt.side_decay_factor) * opt.negative_w / 2

        weights = torch.tensor([1.0, side_neg_w, front_neg_w])
    return text_z, weights.to(text_z.device)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


class Trainer():
    def __init__(self, 
		         argv, # command line args
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 guidance, # guidance network
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):
        self.argv = argv
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        from taming.modules.losses.vqperceptual import LPIPS  # TODO: taming dependency yes/no?

        self.perceptual_loss = LPIPS().eval().to(self.device)


        # guide model
        self.guidance = guidance

        # text prompt
        if self.guidance is not None:
            
            for p in self.guidance.parameters():
                p.requires_grad = False

            # self.prepare_text_embeddings()
            
            with torch.no_grad():
                print('prepare embedding!!')
                self.prepare_embeddings()

        else:
            raise NotImplementedError

        # try out torch 2.0
        if torch.__version__[0] == '2':
            self.model = torch.compile(self.model)
            self.guidance = torch.compile(self.guidance)
    
        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        self.optimizer = optimizer(model) if optimizer is not None else None
        if lr_scheduler is None:
            if self.optimizer is not None:
                self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
            else:
                self.lr_scheduler = None
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None

        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.init_path = os.path.join(self.opt.init_with, f'checkpoints')

            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)
        
        self.log(f'[INFO] Cmdline: {self.argv}')
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)
    @torch.no_grad()
    def prepare_embeddings(self):
        
        self.embeddings = {'full':{}, 'pooled': {}}
        self.embeddings['full']['default'], self.embeddings['pooled']['default'] = self.guidance.get_text_embeds([self.opt.text])
        self.embeddings['full']['uncond'], self.embeddings['pooled']['uncond'] = self.guidance.get_text_embeds([self.opt.negative])

        for d in ['front', 'side', 'back']:
            # print('front side back')
            if self.opt.dir_text:
                self.embeddings['full'][d], self.embeddings['pooled'][d] = self.guidance.get_text_embeds([f"{self.opt.text}, {d} view"])
            else:
                self.embeddings['full'][d], self.embeddings['pooled'][d] = self.guidance.get_text_embeds([f"{self.opt.text}"])

        del self.guidance.pipe.text_encoder
        
        if self.guidance.name == 'sd' and self.guidance.is_xl:
            del self.guidance.pipe.text_encoder_2
        torch.cuda.empty_cache()

    def __del__(self):
        return 
        # if self.log_ptr: 
        #     self.log_ptr.close()


    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    ### ------------------------------	

    def train_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']
        y = torch.zeros(B) # Dummy to keep GAN training structure in tact

        # use normal rendering for shape optimization only
        if self.global_step < (self.opt.latent_iter_ratio * self.opt.iters):
            ambient_ratio = 1.0
            shading = 'normal'
            if self.opt.encode_normal:
                as_latent = False
            else:
                as_latent = True
            bg_color = None
        else:
            if self.global_step <= (self.opt.albedo_iter_ratio * self.opt.iters):
                # albedo shading
                ambient_ratio = 1.0
                shading = 'albedo'
            else:
                # random shading
                ambient_ratio = 0.1 + 0.9 * random.random()
                rand = random.random()
                if rand > 0.8:
                    shading = 'textureless'
                else:
                    shading = 'lambertian'

            as_latent = False

            # random background
            rand = random.random()
            if self.opt.bg_radius > 0 and (rand > 0.5 or data["rgb"] is not None):
                bg_color = None # use bg_net
            else:
                bg_color = torch.rand(3).to(self.device) # single color random bg
        start = time.time()

        
        outputs = self.model.render(rays_o, rays_d, staged=False, perturb=True, bg_color=bg_color, ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True, **vars(self.opt))
        # print("rendering took", time.time() - start)
        pred_rgb = outputs['image'].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # [1, 3, H, W] or [1, 4, H, W]
        pred_depth = outputs['depth'].reshape(B, 1, H, W)
        pred_mask = outputs["weights_sum"].reshape(B, 1, H, W)

        if as_latent:
            # abuse normal & mask as latent code for faster geometry initialization (ref: fantasia3D)
            pred_rgb = torch.cat([outputs['image'], outputs['weights_sum'].unsqueeze(-1)], dim=-1).reshape(B, H, W, 4).permute(0, 3, 1, 2).contiguous() # [B, 4, H, W]
        else:
            pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]

        # find guidance
        # print('data azimuth: ', data['azimuth'], data['azimuth'].shape)


        if data["rgb"] is not None: # if there is gt data from syncdreamer. This happens randomly from the dataloader if using img-to-3d
            if self.opt.anneal_gt > 0:
                percentage = float(self.global_step) / self.opt.iters / self.opt.anneal_gt
                if percentage >= 1: # if lambda_gt is annealed to zero, use SDS
                    use_sds = True
                else:
                    loss_gt_coeff = 1 - min(percentage, 1)
                    use_sds = False
            else:
                loss_gt_coeff = 1
                use_sds = False
        else: # else use sds
            use_sds = True

        if not use_sds:
            loss = 0
            if self.opt.lambda_gt > 0:
                loss_rgb = self.opt.lambda_gt * F.mse_loss(pred_rgb, data["rgb"], reduction='sum')
                loss += loss_rgb
                
            if self.opt.perceptual_weight > 0:
                loss_percept = self.perceptual_loss(pred_rgb.contiguous() *2 - 1,  data["rgb"].contiguous() *2 - 1).item() * (self.opt.h * self.opt.w)
                loss += self.opt.perceptual_weight * loss_percept
            
            # loss latent 
            if self.opt.lambda_latent > 0:
                # size    
                pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
                rgb_512 = F.interpolate(data["rgb"], (512, 512), mode='bilinear', align_corners=False)
                latent_pred = self.guidance.encode_latent(pred_rgb_512, requires_grad=True)
                latent_gt = self.guidance.encode_latent(rgb_512, requires_grad=False)
                loss_latent = self.opt.lambda_latent * F.mse_loss(latent_pred, latent_gt, reduction='sum') * self.guidance.size_scale

                loss += loss_latent
            
            loss_mask = self.opt.lambda_mask * F.mse_loss(pred_mask, data["mask"], reduction='sum')
            # loss = loss_rgb + loss_mask
            loss = loss + loss_mask
            loss = loss * loss_gt_coeff
            # loss_recon = loss_rgb.item()
            loss_recon = loss.item()
        else:
            if self.guidance.name == 'sd':
                azimuth = data['azimuth'] # [-180, 180]
                text_z = [self.embeddings['full']['uncond']] * azimuth.shape[0]
                text_z_comp, weights = adjust_text_embeddings(self.embeddings["full"], azimuth, self.opt)
                text_z.append(text_z_comp)
                text_z = torch.cat(text_z, dim=0)

                text_z2 = [self.embeddings['pooled']['uncond']] * azimuth.shape[0]
                text_z2_comp, _ = adjust_text_embeddings(self.embeddings["pooled"], azimuth, self.opt)
                text_z2.append(text_z2_comp)
                text_z2 = torch.cat(text_z2, dim=0)

                loss, loss_recon = self.guidance.train_step(text_z, text_z2, 
                                                            weights, 
                                                            pred_rgb, 
                                                            outputs, 
                                                            self.opt.guidance_scale, 
                                                            self.global_step, shading=="normal")
            else:
                # TODO: Not sure if it's True with IF
                azimuth = data['azimuth'] # [-180, 180]
                text_z = [self.embeddings['full']['uncond']] * azimuth.shape[0]
                # if self.opt.perpneg:
                text_z_comp, weights = adjust_text_embeddings(self.embeddings["full"], azimuth, self.opt)
                text_z.append(text_z_comp)
                text_z = torch.cat(text_z, dim=0)                

                loss, loss_recon = self.guidance.train_step_perpneg(text_z, 
                                                                    weights,
                                                                    pred_rgb, 
                                                                    outputs,
                                                                    guidance_scale=self.opt.guidance_scale, 
                                                                    global_step=self.global_step,
                                                                    grad_scale=1)        
        # regularizations
        if self.opt.lambda_distortion > 0:
            nears, fars = outputs["bounds"]
            g_tf, g_tn = 1 / fars, 1 / nears
            zval = outputs["z_vals"]
            g_t = 1 / zval
            s = (g_t - g_tn) / (g_tf - g_tn)
            weights = outputs["weights"]
            loss_distortion = eff_distloss_native(weights[..., :-1], (s[..., :-1] + s[..., 1:]) / 2, \
                                s[..., 1:] - s[..., :-1])
            loss = loss + self.opt.lambda_distortion * loss_distortion

        if self.opt.lambda_zvar > 0:
            zval = outputs["z_vals"]
            weights = outputs["weights"]
            weights_sum = weights.sum(-1, keepdim=True)
            weights_sum_mask = (weights_sum > 0.5).float()
            weights_normalized = weights / weights_sum.clamp(min=1e-5)
            depth = (zval * weights_normalized).sum(-1, keepdim=True)
            zvar = ((zval - depth) ** 2 * weights_normalized).sum(-1, keepdim=True)
            loss_zvar = (zvar * weights_sum_mask.detach()).sum()
            lambda_zvar = self.opt.lambda_zvar * min(1, 2 * self.global_step / self.opt.iters)
            loss = loss + lambda_zvar * loss_zvar

        if self.opt.lambda_monotonic > 0:
            '''
            Original modification from HiFA
            Basically, the intution is that for a solid object, each ray should only have one ray-surface intersection.
            So we want the nerf rendering's PDF function to be first monotinically increasing, then monotonically decreasing.
            To do so, we first find the peak in PDF of each ray(peak_idx), then calculate the mask for subset of points before/after the peak.
            For each of those subset of points, we penalize the PDF function for not being monotonically increasing/decreasing.
            '''
            pdf = outputs["pdf"] # [N, 96]
            peak_idx = torch.argmax(pdf, dim=-1) # [N]
            pdf_leftpad = torch.cat([torch.zeros_like(pdf[:, :1]), pdf[:, :]], dim=-1) # [N, 97]
            pdf_rightpad = torch.cat([pdf[:, :], torch.zeros_like(pdf[:, :1])], dim=-1) # [N, 97]
            delta = pdf_rightpad - pdf_leftpad
            idx = torch.arange(delta.size(-1)).reshape(1, -1).expand_as(delta).to(delta.device)
            left_mask = idx <= peak_idx.unsqueeze(-1)
            right_mask = idx > peak_idx.unsqueeze(-1)
            left_decrease = torch.clamp(-delta * left_mask, 0)
            right_increase = torch.clamp(delta * right_mask, 0)
            loss_monotonic = left_decrease.mean() + right_increase.mean()
            lambda_monotonic = self.opt.lambda_monotonic# * min(1, 2 * self.global_step / self.opt.iters)
            loss = loss + lambda_monotonic * loss_monotonic

        if self.opt.lambda_opacity > 0:
            loss_opacity = (outputs['weights_sum'] ** 2).mean()
            loss = loss + self.opt.lambda_opacity * loss_opacity

        if self.opt.lambda_zentropy > 0:
            '''
            Original modification from HiFA
            Entropy of the pdf modeled by the weight function
            The implementation in stable-dreamfusion treats PMF as a binary distribution, and penalizes its binary entropy.
            The goal of the binary entropy loss is to encourage most point's alpha weight to be 0.
            But PMF is just PDF * interval. PDF is a continuous distribution, while PMF is discrete. 
            Since interval is random, the binary entropy is dependent on the sampling strategy. 
            So we fix this by calculating the differential entropy of the continous variable(the PDF)
            '''
            weights = outputs['weights']
            pdf = outputs['pdf'].clamp(min=1e-5)
            weights_sum = weights.sum(-1, keepdim=True)
            weights_normalized = weights / weights_sum.clamp(min=1e-5)
            zentropy = (-weights_normalized * torch.log(pdf)).sum(-1, keepdim=True)
            loss_zentropy = (zentropy * weights_sum.detach()).mean()
            lambda_zentropy = self.opt.lambda_zentropy * min(1, 2 * self.global_step / self.opt.iters)
            loss = loss + lambda_zentropy * loss_zentropy

        if self.opt.lambda_bentropy > 0:
            alphas = outputs['weights'].clamp(1e-5, 1 - 1e-5)
            # alphas = alphas ** 2 # skewed entropy, favors 0 over 1
            loss_bentropy = (- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).mean()
            lambda_bentropy = self.opt.lambda_bentropy * min(1, 2 * self.global_step / self.opt.iters)
            loss = loss + lambda_bentropy * loss_bentropy

        if self.opt.lambda_bentropy_sum > 0:
            alphas = outputs['weights_sum'].clamp(1e-5, 1 - 1e-5)
            # alphas = alphas ** 2 # skewed entropy, favors 0 over 1
            loss_bentropy_sum = (- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).mean()
            lambda_bentropy_sum = self.opt.lambda_bentropy_sum# * min(1, 2 * self.global_step / self.opt.iters)  
            loss = loss + lambda_bentropy_sum * loss_bentropy_sum

        if self.opt.lambda_orient > 0 and 'loss_orient' in outputs:
            loss_orient = outputs['loss_orient']
            loss = loss + self.opt.lambda_orient * loss_orient

        # Register the backward hook to print the output gradient of each layer
        # def print_grad(grad):
        #     print('L531 grad', len(grad), grad[0].max(), grad[0].min())
        # for name, layer in self.model.named_modules():
        #     layer.register_backward_hook(lambda module, grad_input, grad_output: print_grad(grad_output))
        #     # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        
        return pred_rgb, pred_depth, loss, loss_recon
    
    def post_train_step(self):

        if self.opt.backbone == 'grid' and self.opt.lambda_tv > 0:

            lambda_tv = min(1.0, self.global_step / 1000) * self.opt.lambda_tv
            # unscale grad before modifying it!
            # ref: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
            self.scaler.unscale_(self.optimizer)
            self.model.encoder.grad_total_variation(lambda_tv, None, self.model.bound)
        
        # clip grad
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)

    def eval_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None
        
        if self.opt.microfacet:
            shading = "albedo"
            ambient_ratio = 0.0
            light_d = safe_normalize(rays_o[0][0])

        outputs = self.model.render(rays_o, rays_d, staged=True, perturb=False, bg_color=None, light_d=light_d, ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True, **vars(self.opt))
        pred_rgb = outputs['image'].reshape(B, H, W, -1).permute(0, 3, 1, 2)
        pred_rgb = pred_rgb.permute(0, 2, 3, 1) # [B, H, W, C]
        pred_depth = outputs['depth'].reshape(B, H, W)

        # dummy 
        loss = torch.zeros([1], device=pred_rgb.device, dtype=pred_rgb.dtype)

        return pred_rgb, pred_depth, loss

    def test_step(self, data, i=0, bg_color=None, perturb=False):  
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(rays_o.device)
        else:
            bg_color = torch.ones(3, device=rays_o.device) # [3]
        if self.opt.bg_test:
            bg_color = None

        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None

        ######### for inspecting geometry
        # shading = "textureless"
        # ambient_ratio = 0.2
        # bg_color = torch.zeros(3, device=rays_o.device)
        # light_d = safe_normalize(rays_o[0][0])
        ######### end for inspecting geometry
        if self.opt.microfacet:
            shading = "albedo"
            ambient_ratio = 0.0
            light_d = safe_normalize(rays_o[0][0])
            
        # outputs = self.model.render(rays_o, rays_d, staged=True, perturb=False, bg_color=None, light_d=light_d, ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True, **vars(self.opt))
        outputs = self.model.render(rays_o, rays_d, staged=True, perturb=perturb, light_d=light_d, ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True, bg_color=bg_color, **vars(self.opt))
        pred_rgb = outputs['image'].reshape(B, H, W, -1).permute(0, 3, 1, 2)
        pred_rgb = pred_rgb.permute(0, 2, 3, 1) # [B, H, W, C]
        pred_depth = outputs['depth'].reshape(B, H, W)
        pred_mask = outputs['weights_sum'].reshape(B, H, W)
        if self.opt.return_normal:
            pred_normal =  outputs['normals'].reshape(B, H, W, -1)
        else:
            pred_normal = None
        return pred_rgb, pred_depth, pred_mask, pred_normal


        '''
        visualization code for kernel smoothing
        z_mtx = outputs["z_vals"].reshape(H, W, -1).detach().cpu()
        weight_mtx = outputs["weights"].reshape(H, W, -1).detach().cpu()

        pdf_mtx = outputs["pdf"].reshape(H, W, -1).detach().cpu()
    
        ############################################################
        sigma_mtx = outputs["sigmas"].reshape(H, W, -1).detach().cpu()
        xcoord, ycoord = 249, 276


        z_vals_mid = z_mtx[ycoord, xcoord][..., :-1] / 2 + z_mtx[ycoord, xcoord][..., 1:] / 2
        pdf_mid = pdf_mtx[ycoord, xcoord][..., :-1] / 2  + pdf_mtx[ycoord, xcoord][..., 1:] / 2        

    
        x_0 = 2.85
        x_1 = 2.92
        -----------------------
        plot the piecewise constant integral
        if i  == 9:  
        plt.close()
        plt.figure(figsize=(8, 6))
        plt.stem(z_vals_mid.numpy(), pdf_mtx[ycoord, xcoord][1:].numpy(), basefmt="", markerfmt="o", linefmt="C1-")
        line2 = plt.scatter(z_mtx[ycoord, xcoord], pdf_mtx[ycoord, xcoord], marker="x", color="red", s=100, linewidths=3)
        
        line3 = plt.plot(pdf_mtx[ycoord, xcoord][1:-1], z_vals_mid, color='#1f77b4')
        line3 = plt.plot(z_mtx[ycoord, xcoord], pdf_mtx[ycoord, xcoord], color="#1f77b4")[0]
        
        line1 = plt.stem(z_vals_mid.numpy(), pdf_mtx[ycoord, xcoord][:-1].numpy(), basefmt="k-", markerfmt="o", linefmt="C1-")
        plt.legend([line1, line2, line3], ['bin boundary', 'coarse samples', 'pdf'])
        plt.xlabel('z_coordinate', fontsize=18)
        plt.ylabel('pdf', fontsize=18)  
        plt.xlim(3.015, 3.055)
        plt.xlim(2.015, 5.055)
        np.save('figs/coarse_nokernel_z.npy', z_mtx[ycoord, xcoord].numpy())
        np.save('figs/coarse_nokernel_pdf.npy', pdf_mtx[ycoord, xcoord].numpy())
        
        np.save('figs/coarse_z.npy', z_mtx[ycoord, xcoord].numpy())
        np.save('figs/coarse_pdf.npy', pdf_mtx[ycoord, xcoord].numpy())
        np.save('figs/gt_z.npy', z_mtx[ycoord, xcoord].numpy())
        np.save('figs/gt_pdf.npy', pdf_mtx[ycoord, xcoord].numpy())
        np.save('figs/gt_weights.npy', (weight_mtx)[ycoord, xcoord].numpy())
        plt.xlim(x_0, x_1)
        plt.ylim(-5, 180)
        plt.savefig(f"figs/bins_naive_{xcoord}_{ycoord}.png")

        -----------------------
        # plot density
        plt.close()
        plt.figure(figsize=(8, 6))
        line1 = plt.scatter(z_mtx[ycoord, xcoord], sigma_mtx[ycoord, xcoord], marker="x", color="red", s=50, linewidths=2)
        line2 = plt.stairs(sigma_mtx[ycoord, xcoord][:-1], z_mtx[ycoord, xcoord], color="#1f77b4")
        plt.xlabel('z_coordinate', fontsize=18)
        plt.ylabel(r'$\sigma$', fontsize=18)  
        # plt.xlim(3.015, 3.055)
        # plt.xlim(2.015, 5.055)
        plt.xlim(x_0, x_1)
        plt.ylim(-20, 500)
        
        plt.savefig(f"figs/{i}_density_{self.opt.num_steps}_{self.opt.upsample_steps}_{xcoord}_{ycoord}")

        # -----------------------
        # plot fine samples
        plt.close()
        plt.figure(figsize=(8, 6))
        line1 = plt.scatter(z_mtx[ycoord, xcoord], pdf_mtx[ycoord, xcoord], marker="x", color="red", s=50, linewidths=2)
        line2 = plt.plot(z_mtx[ycoord, xcoord], pdf_mtx[ycoord, xcoord], color="#1f77b4")[0]
        plt.xlabel('z_coordinate', fontsize=18)
        plt.ylabel('pdf', fontsize=18)  
        # plt.xlim(3.015, 3.055)
        plt.xlim(x_0, x_1)
        plt.ylim(-5, 160)
        np.save('figs/fine_z.npy', z_mtx[ycoord, xcoord].numpy())
        np.save('figs/fine_pdf.npy', pdf_mtx[ycoord, xcoord].numpy())
        
        plt.savefig(f"figs/{i}_pdf_{self.opt.num_steps}_{self.opt.upsample_steps}_{xcoord}_{ycoord}")

        
        -----------------------
        plot weights
        plt.close()
        plt.figure(figsize=(8, 6))
        line1 = plt.stem(z_mtx[ycoord, xcoord].numpy(), weight_mtx[ycoord, xcoord].numpy())
        plt.xlabel('z_coordinate', fontsize=18)
        plt.ylabel('weights', fontsize=18)     
        plt.xlim(3.015, 3.055)
        plt.ylim(-0.02, 0.2)
        plt.savefig(f"figs/weights_{i}_{self.opt.num_steps}_{self.opt.upsample_steps}_{xcoord}_{ycoord}")

        -----------------------
        save rendering
        plt.imsave(f"figs/{i}_render_{self.opt.num_steps}_{self.opt.upsample_steps}_{xcoord}_{ycoord}.png", cv2.rectangle(pred_rgb[0].cpu().numpy(), (xcoord - 3, ycoord - 3), (xcoord + 3, ycoord + 3), (1, 0, 0), 1))
        plt.imsave(f"figs/{i}_zoomed_{self.opt.num_steps}_{self.opt.upsample_steps}_{xcoord}_{ycoord}.png", cv2.rectangle(pred_rgb[0].cpu().numpy(), (xcoord - 3, ycoord - 3), (xcoord + 3, ycoord + 3), (1, 0, 0), 1)[ycoord - 50: ycoord + 50, xcoord - 50: xcoord + 50])
    '''


            

    def generate_point_cloud(self, loader):

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        all_points = []
        all_normals = []

        with torch.no_grad():

            for i, data in enumerate(loader):

                data['shading'] = 'normal' # to get normal as color
                
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, preds_mask = self.test_step(data)

                pred_mask = preds_mask[0].detach().cpu().numpy().reshape(-1) # [H, W], bool
                pred_depth = preds_depth[0].detach().cpu().numpy().reshape(-1, 1) # [N, 1]

                normals = preds[0].detach().cpu().numpy() * 2 - 1 # normals in [-1, 1]
                normals = normals.reshape(-1, 3) # shape [N, 3]

                rays_o = data['rays_o'][0].detach().cpu().numpy() # [N, 3]
                rays_d = data['rays_d'][0].detach().cpu().numpy() # [N, 3]
                points = rays_o + pred_depth * rays_d

                if pred_mask.any():
                    all_points.append(points[pred_mask])
                    all_normals.append(normals[pred_mask])

                pbar.update(loader.batch_size)
        
        points = np.concatenate(all_points, axis=0)
        normals = np.concatenate(all_normals, axis=0)
            
        return points, normals


    def save_mesh(self, loader=None, save_path=None):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'mesh')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(save_path, exist_ok=True)

        if loader is None: # mcubes
            self.model.export_mesh(save_path, resolution=self.opt.mcubes_resolution, decimate_target=self.opt.decimate_target)
        else: # poisson (TODO: not working currently...)
            points, normals = self.generate_point_cloud(loader)
            self.model.export_mesh(save_path, points=points, normals=normals, decimate_target=self.opt.decimate_target)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):
        '''
        returns false if doesnt need training
        '''

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        start_t = time.time()
        if self.epoch == max_epochs:
            return False

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

        end_t = time.time()

        self.log(f"[INFO] training takes {(end_t - start_t)/ 60:.4f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()
        return True

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True):
        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []
            all_preds_mask = []
            all_preds_normal = []
        with torch.no_grad():
            for i, data in enumerate(loader):
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, preds_mask, pred_normal = self.test_step(data, i)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)
                if self.opt.return_normal:
                    pred_normal= pred_normal[0].detach().cpu().numpy()
                    pred_normal = (pred_normal * 255).astype(np.uint8)
                
                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-6)
                pred_depth = (pred_depth * 255).astype(np.uint8)

                pred_mask = preds_mask[0].detach().cpu().numpy()
                pred_mask = (pred_mask * 255).astype(np.uint8)

                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                    all_preds_mask.append(pred_mask)
                    if self.opt.return_normal:
                        all_preds_normal.append(pred_normal)
                else:
                    if self.opt.return_normal:
                        cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_normal.png'), cv2.cvtColor(pred_normal, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    
                    # np.save(os.path.join(save_path, f'rendered_rgb_{str(i).zfill(5)}.npy'), pred)
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_mask.png'), pred_mask)

                # Write cam-2pose
                # print(data['c2w'].size(), data['intrinsics']) 
                # np.savetxt(os.path.join(save_path,f'{name}_{i:04d}_c2w.txt'), data['c2w'][0].cpu().numpy())
                
                # import json 
                
                # json_data = {'name': f'{name}_{i:04d}.png', 
                #              'c2w': data['c2w'][0].cpu().numpy().tolist(),
                #              'intrinsics': data['intrinsics'].tolist()}
                
                # with open(os.path.join(save_path,'c2w.json' ), 'a') as fp:
                #     json.dump(json_data, fp, sort_keys=False, indent=4)
                #     fp.write('\n')
                
                # np.save(os.path.join(save_path, f'c2w_{str(i).zfill(5)}.npy'), data['c2w'][0].cpu().numpy())
                
                pbar.update(loader.batch_size)

        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)
            all_preds_mask = np.stack(all_preds_mask, axis=0)
            if self.opt.return_normal:
                all_preds_normal = np.stack(all_preds_normal, axis=0)

            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_mask.mp4'), all_preds_mask, fps=25, quality=8, macro_block_size=1)
            if self.opt.return_normal:
                imageio.mimwrite(os.path.join(save_path, f'{name}_normal.mp4'), all_preds_normal, fps=25, quality=8, macro_block_size=1)

                from moviepy.editor import VideoFileClip, clips_array
                clip0 = VideoFileClip(os.path.join(save_path, f'{name}_rgb.mp4'))
                clip1 = VideoFileClip(os.path.join(save_path, f'{name}_normal.mp4'))
                combined = clips_array([[clip0, clip1]])
                combined.write_videofile(os.path.join(save_path, f'{name}_rgb_normal.mp4'))

        self.log(f"==> Finished Test.")
    
    # [GUI] train text step.
    def train_gui(self, train_loader, step=16):

        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        
        loader = iter(train_loader)

        for _ in range(step):
            
            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
            
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                pred_rgbs, pred_depths, loss, loss_recon = self.train_step(data)
         
            # self.scaler.scale(loss).backward()
            loss.backward()
            
            self.post_train_step()
            self.optimizer.step()
            # self.scaler.step(self.optimizer)
            # self.scaler.update()
            
            if self.scheduler_update_every_step:
                self.lr_scheduler.step()
                if self.global_step > self.opt.gan_iters:
                    self.lr_scheduler_d.step()

            total_loss += loss.detach()

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
                self.lr_scheduler_d.step()
            else:
                self.lr_scheduler.step()
                self.lr_scheduler_d.step()


        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }
        
        return outputs

    
    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, W, H, bg_color=None, spp=1, downscale=1, light_d=None, ambient_ratio=1.0, shading='albedo'):
        
        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1)

        # from degree theta/phi to 3D normalized vec
        light_d = np.deg2rad(light_d)
        light_d = np.array([
            np.sin(light_d[0]) * np.sin(light_d[1]),
            np.cos(light_d[0]),
            np.sin(light_d[0]) * np.cos(light_d[1]),
        ], dtype=np.float32)
        light_d = torch.from_numpy(light_d).to(self.device)

        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
            'light_d': light_d,
            'ambient_ratio': ambient_ratio,
            'shading': shading,
        }
        
        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed!
                preds, preds_depth, _ = self.test_step(data, bg_color=bg_color, perturb=False if spp == 1 else spp)

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        outputs = {
            'image': preds[0].detach().cpu().numpy(),
            'depth': preds_depth[0].detach().cpu().numpy(),
        }

        return outputs

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training {self.workspace} Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:    
            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
                    
            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                pred_rgbs, pred_depths, loss, loss_recon = self.train_step(data)

            # loss.backward()
            start = time.time()
            self.scaler.scale(loss).backward()
            
            self.post_train_step()
            # self.optimizer.step()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # print('backprop took', time.time() - start)

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                # if self.report_metric_at_train:
                #     for metric in self.metrics:
                #         metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)
                    self.writer.add_scalar("train/loss_recon", loss_recon, self.global_step)
                    # self.writer.add_scalar("train/total_loss", total_loss, self.global_step)
                    
                    # for name, param in self.model.named_parameters():
                    #     fake_distribution = []
                    #     if param.grad is not None:
                    #         # print('name & grad', name, param.grad)
                    #         fake_distribution += [param.grad.mean()]
                    #         self.writer.add_histogram('gradients/%s' % name,  torch.mean(torch.stack(fake_distribution)), self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
                self.lr_scheduler_d.step()
            else:
                self.lr_scheduler.step()
                self.lr_scheduler_d.step()


        self.log(f"==> Finished Epoch {self.epoch}.")


    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate {self.workspace} at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            for data in loader:    
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, loss = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size
                    
                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)
                
                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    # save image
                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                    save_path_depth = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_depth.png')

                    #self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-6)
                    pred_depth = (pred_depth * 255).astype(np.uint8)
                    
                    cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_depth, pred_depth)

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)


        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{name}.pth"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = os.path.join(self.ckpt_path, self.stats["checkpoints"].pop(0))
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, os.path.join(self.ckpt_path, file_path))

        else:    
            if len(self.stats["results"]) > 0:
                # always save best since loss cannot reflect performance.
                if True:
                    # self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    # self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                checkpoint_list = sorted(glob.glob(f'{self.init_path}/*.pth'))
                if checkpoint_list:
                    checkpoint = checkpoint_list[-1]
                    self.log(f"[INFO] Initializing from previous trial at {checkpoint}")
                else:
                    self.log("[WARN] No checkpoint found, model randomly initialized.")
                    return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        if checkpoint.startswith(self.init_path):
            self.global_step = 0
        else:
            self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
        
        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")
