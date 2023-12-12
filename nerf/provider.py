import os
import cv2
import glob
import json
import tqdm
import random
import numpy as np
import scipy
from scipy.spatial.transform import Slerp, Rotation
from skimage.io import imread, imsave
from skimage.transform import resize
from PIL import Image

import trimesh

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils import get_rays, get_rays_syncdreamer, safe_normalize
import pickle

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)
    
DIR_COLORS = np.array([
    [255, 0, 0, 255], # front
    [0, 255, 0, 255], # side
    [0, 0, 255, 255], # back
    [255, 255, 0, 255], # side
    [255, 0, 255, 255], # overhead
    [0, 255, 255, 255], # bottom 
], dtype=np.uint8)

def visualize_poses(poses, dirs, size=0.1):
    # poses: [B, 4, 4], dirs: [B]

    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=1)
    objects = [axes, sphere]

    for pose, dir in zip(poses, dirs):
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
        segs = trimesh.load_path(segs)

        # different color for different dirs
        segs.colors = DIR_COLORS[[dir]].repeat(len(segs.entities), 0)

        objects.append(segs)

    trimesh.Scene(objects).show()

def get_view_direction(thetas, phis, overhead, front):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis
    res[(phis < front)] = 0
    res[(phis >= front) & (phis < np.pi)] = 1
    res[(phis >= np.pi) & (phis < (np.pi + front))] = 2
    res[(phis >= (np.pi + front))] = 3
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res


def rand_poses(size, device, radius_range=[1, 1.5], theta_range=[0, 120], phi_range=[0, 360], return_dirs=False, angle_overhead=30, angle_front=60, jitter=False, uniform_sphere_rate=0.5):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''

    theta_range = np.array(theta_range) / 180 * np.pi
    phi_range = np.array(phi_range) / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

    if random.random() < uniform_sphere_rate:
        unit_centers = F.normalize(
            torch.stack([
                (torch.rand(size, device=device) - 0.5) * 2.0,
                torch.rand(size, device=device),
                (torch.rand(size, device=device) - 0.5) * 2.0,
            ], dim=-1), p=2, dim=1
        )
        thetas = torch.acos(unit_centers[:,1])
        phis = torch.atan2(unit_centers[:,0], unit_centers[:,2])
        phis[phis < 0] += 2 * np.pi
        centers = unit_centers * radius.unsqueeze(-1)
    else:
        thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
        phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]
        phis[phis < 0] += 2 * np.pi

        centers = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.cos(thetas),
            radius * torch.sin(thetas) * torch.cos(phis),
        ], dim=-1) # [B, 3]

    targets = 0

    # jitters
    if jitter:
        centers = centers + (torch.rand_like(centers) * 0.2 - 0.1)
        targets = targets + torch.randn_like(centers) * 0.2

    # lookat
    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))

    if jitter:
        up_noise = torch.randn_like(up_vector) * 0.02
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None

    # back to degree
    thetas = thetas / np.pi * 180
    phis = phis / np.pi * 180

    return poses, dirs, thetas, phis, radius


def circle_poses(device, radius=torch.tensor([3.2]), theta=torch.tensor([60]), phi=torch.tensor([0]), return_dirs=False, angle_overhead=30, angle_front=60):

    theta = theta / 180 * np.pi
    phi = phi / 180 * np.pi
    angle_overhead = angle_overhead / 180 * np.pi
    angle_front = angle_front / 180 * np.pi

    centers = torch.stack([
        radius * torch.sin(theta) * torch.sin(phi),
        radius * torch.cos(theta),
        radius * torch.sin(theta) * torch.cos(phi),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = safe_normalize(centers)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(len(centers), 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(len(centers), 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(theta, phi, angle_overhead, angle_front)
    else:
        dirs = None

    return poses, dirs
    

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


class BackgroundRemoval:
    def __init__(self, device='cuda'):
        from carvekit.api.high import HiInterface
        print("Loading BackgroundRemoval...")
        self.interface = HiInterface(
            object_type="object",  # Can be "object" or "hairs-like".
            batch_size_seg=5,
            batch_size_matting=1,
            device=device,
            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
            matting_mask_size=2048,
            trimap_prob_threshold=231,
            trimap_dilation=30,
            trimap_erosion_iters=5,
            fp16=True,
        )

    @torch.no_grad()
    def __call__(self, image):
        # image: [H, W, 3] array in [0, 255].
        image = Image.fromarray(image)
        image = self.interface([image])[0]
        image = np.array(image)
        return image


class NeRFDataset:
    def __init__(self, opt, device, type='train', H=256, W=256, size=100):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test

        self.H = H
        self.W = W
        self.size = size

        self.training = self.type in ['train', 'all']
        
        self.cx = self.H / 2
        self.cy = self.W / 2
            
        if self.opt.gt_image_rate > 0:
            mask_predictor = BackgroundRemoval()
            
            self.K, self.azs, self.els, self.dists, self.poses = read_pickle(f'camera-16.pkl')
            self.images_info = {'images': [] ,'masks': [], 'Ks': [], 'poses':[]}
            img = imread(self.opt.image_path)
            
            # New: change to 17
            # for index in range(17):    
            for index in range(16):            
                rgb = np.copy(img[:,index*256:(index+1)*256,:])
                # New: resize to [H, W]
                # rgb = resize(rgb, (self.H, self.W), order=1).astype(np.uint8)
                # predict mask
                masked_image = mask_predictor(rgb)
                mask = masked_image[:,:,3].astype(np.float32)/255
                rgb = rgb.astype(np.float32)/255
                K, pose = np.copy(self.K), self.poses[index]
                
                self.images_info['images'].append(torch.from_numpy(rgb.astype(np.float32))) # h,w,3
                self.images_info['masks'].append(torch.from_numpy(mask.astype(np.float32))) # h,w
                self.images_info['Ks'].append(torch.from_numpy(K.astype(np.float32)))
                self.images_info['poses'].append(torch.from_numpy(pose.astype(np.float32)))
            del mask_predictor
        # self.epoch = 0
        # [debug] visualize poses
        # poses, dirs = rand_poses(100, self.device, radius_range=self.opt.radius_range, return_dirs=self.opt.dir_text, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front, jitter=self.opt.jitter_pose, uniform_sphere_rate=1)
        # visualize_poses(poses.detach().cpu().numpy(), dirs.detach().cpu().numpy())


    def collate(self, index):
        # if index[0] == 0:
        #     self.epoch += 1

        # torch.manual_seed(index[0] // 1 + self.epoch * 100)
        # random.seed(index[0] // 1 + self.epoch * 100)
        B = len(index) # always 1

        rgb = None
        mask = None
        use_gt = False
        if self.training:
            rand = random.random()
            if rand < self.opt.gt_image_rate:
                assert self.opt.h == self.opt.w == 256, "we currently dont support img-to-3d experiments other than 256x256 resolution"
                # randomly sample an image
                img_index = random.randint(0, 15)
                rgb = np.copy(self.images_info['images'][img_index])
                mask = np.copy(self.images_info['masks'][img_index])
                K = np.copy(self.images_info['Ks'][img_index])
                # switch y and z axies
                pose = np.copy(self.images_info['poses'][img_index])
                pose[:, 0], pose[:, 1], pose[:, 2] = -pose[:, 1].copy(), pose[:, 2].copy(), pose[:, 0].copy()
                intrinsics = torch.from_numpy(K).unsqueeze(0).repeat(B, 1, 1).to(self.device)
                poses = torch.from_numpy(pose).unsqueeze(0).repeat(B, 1, 1).to(self.device)
                # pose_inv = np.concatenate([R.T, -R.T @ t.reshape(3, 1)], axis=1)
                focal = K[0,0]
                self.cx = K[0,2]
                self.cy = K[1,2]
                use_gt = True
                thetas, phis, radius, dirs = torch.zeros(B).to(self.device), torch.zeros(B).to(self.device), torch.zeros(B).to(self.device), torch.zeros(B).to(self.device)
            else:
                # random pose on the fly
                poses, dirs, thetas, phis, radius = rand_poses(B, self.device, radius_range=self.opt.radius_range, theta_range=self.opt.theta_range, phi_range=self.opt.phi_range, return_dirs=True, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front, jitter=self.opt.jitter_pose, uniform_sphere_rate=self.opt.uniform_sphere_rate)

                # random focal
                fov = random.random() * (self.opt.fovy_range[1] - self.opt.fovy_range[0]) + self.opt.fovy_range[0]
                focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
                intrinsics = np.array([focal, focal, self.cx, self.cy])
        else:
            # # circle pose
            thetas = torch.FloatTensor([self.opt.default_polar]).to(self.device)
            phis = torch.FloatTensor([(index[0] / self.size) * 360]).to(self.device)
            radius = torch.FloatTensor([self.opt.default_radius]).to(self.device)
            poses, dirs = circle_poses(self.device, radius=radius, theta=thetas, phi=phis, return_dirs=True, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front)
            
            # random pose on the fly
            # poses, dirs, thetas, phis, radius = rand_poses(B, self.device, radius_range=self.opt.radius_range, theta_range=self.opt.theta_range, phi_range=self.opt.phi_range, return_dirs=True, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front, jitter=self.opt.jitter_pose, uniform_sphere_rate=self.opt.uniform_sphere_rate)

            # fixed focal
            fov = self.opt.default_fovy

            focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
            intrinsics = np.array([focal, focal, self.cx, self.cy])

        # projection = torch.tensor([
        #     [2*focal/self.W, 0, 0, 0],
        #     [0, -2*focal/self.H, 0, 0],
        #     [0, 0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)],
        #     [0, 0, -1, 0]
        # ], dtype=torch.float32, device=self.device).unsqueeze(0)

        # mvp = projection @ torch.inverse(poses) # [1, 4, 4]

        # sample a low-resolution but full image
        if use_gt:
            rays = get_rays_syncdreamer(poses, intrinsics, self.H, self.W, -1)
        else:
            rays = get_rays(poses, intrinsics, self.H, self.W, -1)


        # delta polar/azimuth/radius to default view
        delta_polar = thetas - self.opt.default_polar
        delta_azimuth = phis - self.opt.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360 # range in [-180, 180]
        delta_radius = radius - self.opt.default_radius

        data = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'], #.to(torch.float16),
            'rays_d': rays['rays_d'], #.to(torch.float16),
            'dir': dirs,
            # 'mvp': mvp,
            'polar': delta_polar,
            'azimuth': delta_azimuth,
            'radius': delta_radius,
            # 'c2w': pose_spherical(delta_azimuth, delta_polar, delta_radius)
            'c2w': poses,
            'intrinsics': intrinsics,
            'rgb': torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).unsqueeze(0) if rgb is not None else None,
            'mask': torch.from_numpy(mask).to(self.device).unsqueeze(0).unsqueeze(0) if mask is not None else None
        }

        return data

    def dataloader(self):
        # loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate, shuffle=False, num_workers=0)
        return loader
