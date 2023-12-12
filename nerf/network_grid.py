import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from activation import trunc_exp
from nerf.bsdf import bsdf_pbr
from .renderer import NeRFRenderer

import numpy as np
from encoding import get_encoder
from diffusers.models.vae import DiagonalGaussianDistribution

from .utils import safe_normalize

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)
    
    def forward(self, x, start_layer=0):
        for l in range(start_layer, self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x


class NeRFNetwork(NeRFRenderer):
    def __init__(self, 
                 opt,
                 num_layers=1,
                 hidden_dim=32,
                 num_layers_bg=2,
                 hidden_dim_bg=16,
                 ):
        
        super().__init__(opt)
        # num_layers=2
        # hidden_dim=64

        self.return_normal = self.opt.return_normal
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.encoder, self.in_dim = get_encoder('hashgrid', input_dim=3, log2_hashmap_size=19, desired_resolution=2048 * self.bound, interpolation='smoothstep')
        self.sh_encoder, self.sh_dim = get_encoder("sphere_harmonics", input_dim=3, degree=4)

        self.sigma_net = MLP(self.in_dim, hidden_dim + 1, hidden_dim, num_layers, bias=True)
        self.rgb_net = MLP(self.hidden_dim + self.sh_dim, 3, hidden_dim, num_layers=2, bias=True) # TODO: increase number of layers
        if self.opt.microfacet:
            self.arm_net = MLP(self.in_dim, 3, hidden_dim, num_layers, bias=True)

        self.normal_net = MLP(self.in_dim, 3, hidden_dim, num_layers, bias=True)

        self.density_activation = trunc_exp if self.opt.density_activation == 'exp' else F.softplus

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg   
            self.hidden_dim_bg = hidden_dim_bg
            
            # use a very simple network to avoid it learning the prompt...
            self.encoder_bg, self.in_dim_bg = get_encoder('frequency', input_dim=3, multires=4)
            self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)
            
        else:
            self.bg_net = None

        self.conv_deep = nn.Sequential(
                        nn.Conv2d(hidden_dim, hidden_dim, 1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(hidden_dim, 8, 1),
                        )

    def smoothen(self, x, B, conv_type):
        assert conv_type == "deep"
        H, W = self.opt.h, self.opt.w
        C = x.shape[-1]
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        x = self.conv_deep(x)
        x = x.permute(0, 2, 3, 1)
        return x

    # add a density blob to the scene center
    def density_blob(self, x):
        # x: [B, N, 3]
        
        d = (x ** 2).sum(-1)
        # g = self.opt.blob_density * torch.exp(- d / (self.opt.blob_radius ** 2))
        g = self.opt.blob_density * (1 - torch.sqrt(d) / self.opt.blob_radius)

        return g

    def common_forward(self, x, dir=None):
        with torch.no_grad():
            sh_enc = self.sh_encoder(dir)
            rand = random.random()
            # if self.opt.albedo and not self.opt.microfacet:
            if self.opt.albedo and not self.opt.microfacet and (rand < self.opt.dir_rate or not self.training):
                sh_enc = sh_enc
            else:
                sh_enc = torch.zeros_like(sh_enc)
        # sigma
        # enc = torch.utils.checkpoint.checkpoint(self.encoder, x, self.bound, use_reentrant=False)
        enc = self.encoder(x, bound=self.bound)

        # h = torch.utils.checkpoint.checkpoint(self.sigma_net, enc)
        h = self.sigma_net(enc)

        sigma = self.density_activation(h[..., 0] + self.density_blob(x))

        # compute albedo
        h = torch.addmm(self.rgb_net.net[0].bias, h[..., 1:], self.rgb_net.net[0].weight[:, :self.in_dim].T)
        h = torch.addmm(h, sh_enc, self.rgb_net.net[0].weight[:, self.in_dim:].T)
        if self.rgb_net.num_layers > 1:
            h = F.relu(h, inplace=True)
        h = self.rgb_net(h, start_layer=1)
        albedo = torch.sigmoid(h)
        return sigma, albedo, enc
    
        # ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
    def finite_difference_normal(self, x, epsilon=1e-2):
        # x: [N, 3]
        dx_pos = self.density((x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
        dx_neg = self.density((x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
        dy_pos = self.density((x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
        dy_neg = self.density((x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
        dz_pos = self.density((x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
        dz_neg = self.density((x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
        
        normal = torch.stack([
            0.5 * (dx_pos - dx_neg) / epsilon, 
            0.5 * (dy_pos - dy_neg) / epsilon, 
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return -normal

    def normal(self, x):
        normal = self.finite_difference_normal(x)
        normal = safe_normalize(normal)
        # remove normal range from [-1,1] to [0,1]
        # normal = (normal + 1) / 2
        
        normal = torch.nan_to_num(normal)
        return normal
    
    def forward(self, x, d, l=None, ratio=1, shading='albedo'):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)
        sigma, albedo, enc = self.common_forward(x, d)
        
        if self.opt.microfacet:
            arm = torch.sigmoid(self.arm_net(enc))
            # arm = torch.cat([torch.zeros_like(arm[..., 0:1]), arm[..., 1:2], torch.zeros_like(arm[..., 2:3])], dim=-1) # diffuse only
            normal = self.normal(x)
            view_pos = x + d
            light_pos = x + l
            bsdf = bsdf_pbr(albedo, arm, x, normal, view_pos, light_pos, 0.08, 0)
            # TODO: maybe add ambient shading
            color = bsdf * np.pi
        else:
            if shading == 'albedo':
                # normal = None
                if self.opt.lambda_orient > 0:
                    normal = self.normal(x)
                else:
                    normal = None
                color = albedo
            
            else: # lambertian shading
                # normal = self.normal_net(enc)
                normal = self.normal(x)

                lambertian = ratio + (1 - ratio) * (normal @ l).clamp(min=0) # [N,]

                if shading == 'textureless':
                    color = lambertian.unsqueeze(-1).repeat(1, albedo.size(-1))
                elif shading == 'normal':
                    color = (normal + 1) / 2
                else: # 'lambertian'
                    color = albedo * lambertian.unsqueeze(-1)

        # TODO: add a flag of return_normal
        if self.return_normal:
            normal = self.normal(x)
            normal = (normal + 1) / 2
        return sigma, color, normal

      
    def density(self, x, dir=None):
        # x: [N, 3], in [-bound, bound]
        
        enc = self.encoder(x, bound=self.bound)
        h = self.sigma_net(enc)
        sigma = self.density_activation(h[..., 0] + self.density_blob(x))
        return {
            'sigma': sigma,
        }


    def background(self, d):

        h = self.encoder_bg(d) # [N, C]
        
        h = self.bg_net(h)

        rgbs = torch.sigmoid(h)

        return rgbs

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr * 10},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.normal_net.parameters(), 'lr': lr},
            {'params': self.conv_deep.parameters(), 'lr': lr},
            {'params': self.rgb_net.parameters(), 'lr': lr},
        ]        

        if self.bg_radius > 0:
            # params.append({'params': self.encoder_bg.parameters(), 'lr': lr * 10})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        if self.opt.microfacet:
            params.append({'params': self.arm_net.parameters(), 'lr': lr})
        return params