import torch
import argparse
import sys
from nerf.provider import NeRFDataset
from nerf.utils import *

# from nerf.gui import NeRFGUI
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler
import huggingface_hub
import os
# pip install carvekit-colab

if __name__ == '__main__':
    huggingface_hub.login("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")
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default=None, help="text prompt")
    parser.add_argument('--negative', default='', type=str, help="negative text prompt")
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --dir_text")
    parser.add_argument('-O2', action='store_true', help="equals --backbone vanilla --dir_text")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--bg_test', action='store_true', help="test mode with background net enabled")
    parser.add_argument('--write_image', action='store_true', help="write image to disk")
    parser.add_argument('--eval_interval', type=int, default=1, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--guidance', type=str, default='stable-diffusion', help='choose from [stable-diffusion, clip, if]')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no_sds', action="store_true")
    parser.add_argument('--loss_type', type=str, default="l2")
    parser.add_argument('--smoothen', type=str, default="")
    parser.add_argument('--guidance_scale', type=float, default=100.0)
    parser.add_argument('--rgb_guidance_scale', type=float, default=100.0, help="deprecated")
    parser.add_argument('--progressive', action="store_true", help='deprecatd')
    parser.add_argument('--max_percent', type=float, default=0.98, help="minimum percent of the timestep to anneal")
    parser.add_argument('--min_percent', type=float, default=0.3, help="minimum percent of the timestep to anneal")
    parser.add_argument('--depthreg_schedule', type=str, choices=['anneal', 'constant', 'quadratic'], default='anneal')
    parser.add_argument('--denoising_steps', type=int, default=4, help="number of DDIM/DPMSolver steps")
    parser.add_argument('--dir_rate', type=float, default=0.5, help="deprecated; if smaller than 1, randomly turn off direction input to NeRF to regularize direction dependency")
    parser.add_argument('--eta', type=float, default=1.0, help="eta for ddim")
    parser.add_argument('--timestep_scheduling', type=str, default='sqrt', choices=['sqrt', 'linear', 'random', 'cosine', 'cubed'])
    parser.add_argument('--if_upsample', action="store_true", help="use upsampling model for IF upsampling")
    parser.add_argument('--if_upsample_guidance', type=float, default=20.0)
    parser.add_argument('--if_upsample_steps', type=int, default=5)
    parser.add_argument('--if_upsample_eta', type=float, default=1.0, help="eta for if upsample ddim")
    parser.add_argument('--if_upsample_h', type=int, default=256)
    parser.add_argument('--if_upsample_w', type=int, default=256)
    parser.add_argument('--if_upsample_noise_level', type=int, default=250)
    parser.add_argument('--sds_weight', default="constant", choices=["constant", "noisematch", "sdslike", "strongerlower"])
    parser.add_argument('--image_path', default="gt_images/firekeeper/0.png")
    parser.add_argument('--gt_image_rate', default=0, type=float, help="during image-to-3d training, ratio of training iterations to use image recon loss vs sds loss")
    parser.add_argument('--clip_grad', action="store_true")

    parser.add_argument('--lambda_gt', default=0.1, type=float)
    parser.add_argument('--lambda_latent', default=1.0, type=float)
    parser.add_argument('--perceptual_weight', default=1.0, type=float)
    
    parser.add_argument('--lambda_mask', default=1.0, type=float)
    parser.add_argument('--anneal_gt', default=0, type=float, help='linearly anneal gt before to this percentage of training process, then use no gt. Set between 0 and 1')
    parser.add_argument('--learned_embeds_path', default=None, type=str)


    parser.add_argument('--scheduler', type=str, default='ddim', choices=['ddim', 'solver'])
    parser.add_argument('--use_pearson', action="store_true")
    parser.add_argument('--interp_size', type=int, default=512)

    parser.add_argument('--save_mesh', action='store_true', help="export an obj mesh with texture")
    parser.add_argument('--mcubes_resolution', type=int, default=512, help="mcubes resolution for extracting mesh")
    parser.add_argument('--decimate_target', type=int, default=1e5, help="target face number for mesh decimation")
    parser.add_argument('--init_with', type=str, default='', help="ckpt to init with")
    parser.add_argument('--use_scaler', action="store_true", help="deprecated")

    ## perp neg options
    parser.add_argument('--negative_w', type=float, default=-2, help="The scale of the weights of negative prompts. A larger value will help to avoid the Janus problem, but may cause flat faces. Vary between 0 to -4, depending on the prompt")
    parser.add_argument('--front_decay_factor', type=float, default=2, help="decay factor for the front prompt")
    parser.add_argument('--side_decay_factor', type=float, default=10, help="decay factor for the side prompt")

    ### training options
    parser.add_argument('--iters', type=int, default=10000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-3, help="max learning rate")
    parser.add_argument('--sigma_scheduler', action="store_true")
    parser.add_argument('--warm_iters', type=int, default=500, help="training iters")
    parser.add_argument('--min_lr', type=float, default=1e-3, help="minimal learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=128, help="num steps sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=64, help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
    parser.add_argument('--albedo', action='store_true', help="only use albedo shading to train, overrides --albedo_iters. Useless when microfacet is enabled")
    parser.add_argument('--albedo_iter_ratio', type=float, default=0, help="training iters that only use albedo shading")
    parser.add_argument('--latent_iter_ratio', type=float, default=0.0, help="number of steps to train with fantasia3d style shape initialization")
    parser.add_argument('--encode_normal', action="store_true", help="When true, if using fantasia3d shape initialization, encode normal map with vae")
    parser.add_argument('--jitter_pose', action='store_true', help="add jitters to the randomly sampled camera poses")
    parser.add_argument('--uniform_sphere_rate', type=float, default=0.5, help="likelihood of sampling camera location uniformly on the sphere surface area")
    parser.add_argument('--microfacet', action="store_true", help="Experimental feature: let MLP output full microfacet BRDF distribution, and shade volumetrically")
    parser.add_argument('--sampling_strategy', default="boxcar", choices=["trapezoidal", "naive", "boxcar"], help="kernel smoothing. Paper mentions the boxcar strategy. Can also treat the PDF as a piecewise linear function, resulting in the trapezoidal filter")
    parser.add_argument('--return_normal', action="store_true", help="if true, return normal map")

    # model options
    parser.add_argument('--bg_radius', type=float, default=1.4, help="if positive, use a background model at sphere(bg_radius)")
    parser.add_argument('--density_activation', type=str, default='softplus', choices=['softplus', 'exp'], help="density activation function")
    parser.add_argument('--density_thresh', type=float, default=1.5, help="threshold for density grid to be occupied")
    parser.add_argument('--blob_density', type=float, default=5, help="max (center) density for the density blob")
    parser.add_argument('--blob_radius', type=float, default=0.2, help="control the radius for the density blob")
    # network backbone
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--backbone', type=str, default='grid', choices=['grid', 'vanilla'], help="nerf backbone")
    parser.add_argument('--optim', type=str, default='adan', choices=['adan', 'adam', 'sgd'], help="optimizer")
    parser.add_argument('--sd_version', type=str, default='5.1', choices=['1.5', '2.0', '2.1', '5.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    # rendering resolution in training, decrease this if CUDA OOM.
    parser.add_argument('--w', type=int, default=512, help="render width for NeRF in training")
    parser.add_argument('--h', type=int, default=512, help="render height for NeRF in training")
    
    ### dataset options
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--dt_gamma', type=float, default=0, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.1, help="minimum near distance for camera")
    parser.add_argument('--dir_text', action='store_true', help="use dir_text")
    parser.add_argument('--suppress_face', action='store_true', help="deprecated")
    parser.add_argument('--angle_overhead', type=float, default=30, help="[0, angle_overhead] is the overhead region")
    parser.add_argument('--angle_front', type=float, default=60, help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")

    ### view options
    parser.add_argument('--radius_range', type=float, nargs='*', default=[3.0, 3.5], help="training camera radius range")
    parser.add_argument('--theta_range', type=float, nargs='*', default=[45, 105], help="training camera fovy range")
    parser.add_argument('--phi_range', type=float, nargs='*', default=[-180, 180], help="training camera fovy range")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[10, 30], help="training camera fovy range")

    parser.add_argument('--default_radius', type=float, default=3.2, help="radius for the default view")
    parser.add_argument('--default_polar', type=float, default=70, help="polar for the default view")
    parser.add_argument('--default_azimuth', type=float, default=0, help="azimuth for the default view")
    parser.add_argument('--default_fovy', type=float, default=20, help="fovy for the default view")

    ### regularizations
    parser.add_argument('--lambda_zentropy', type=float, default=1e-2, help="z-entropy. Original from HiFA. Similar to z-variance. This is the entropy of the pdf modeled by the weight function. See utils.py for details")
    parser.add_argument('--lambda_bentropy', type=float, default=0, help="loss for binary entropy of 3d sample point alpha. Corresponds to lambda_entropy in stable-dreamfusion")
    parser.add_argument('--lambda_bentropy_sum', type=float, default=1e-4, help="loss for binary entropy of 2d sample point alpha")
    parser.add_argument('--lambda_opacity', type=float, default=0, help="loss scale for alpha value")
    parser.add_argument('--lambda_orient', type=float, default=0, help="loss scale for orientation")
    parser.add_argument('--lambda_tv', type=float, default=0, help="loss scale for total variation")
    parser.add_argument('--lambda_gan', type=float, default=0, help="loss scale for gan")
    parser.add_argument('--lambda_zvar', type=float, default=3, help="z-variance loss. Original from HiFA. See paper for details.")
    parser.add_argument('--lambda_depthreg', type=float, default=0.1, help="loss scale for regression of depth")
    parser.add_argument('--lambda_rgb', type=float, default=0.1, help="loss scale for rgb")
    parser.add_argument('--lambda_sds', type=float, default=1.0, help="loss scale for sds in latent space")
    parser.add_argument('--lambda_monotonic', type=float, default=1.0, help="monotonicity loss. Original from HiFA. See utils.py for details")
    parser.add_argument('--lambda_distortion', type=float, default=0.0, help="mipnerf360 distortion loss")
    parser.add_argument('--intensity_only', action="store_true")
    parser.add_argument('--color_bias', type=str, default='', choices=['', 'grayscale'])

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=512, help="GUI width")
    parser.add_argument('--H', type=int, default=512, help="GUI height")
    parser.add_argument('--radius', type=float, default=3, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=60, help="default GUI camera fovy")
    parser.add_argument('--light_theta', type=float, default=60, help="default GUI light direction in [0, 180], corresponding to elevation [90, -90]")
    parser.add_argument('--light_phi', type=float, default=0, help="default GUI light direction in [0, 360), azimuth")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")
    
    opt = parser.parse_args()

    if opt.O:
        opt.fp16 = True
        opt.dir_text = True
        opt.cuda_ray = True

    elif opt.O2:
        # only use fp16 if not evaluating normals (else lead to NaNs in training...)
        if opt.albedo:
            opt.fp16 = True
        opt.dir_text = True
        opt.backbone = 'vanilla'

    if opt.albedo:
        opt.albedo_iter_ratio = 1

    if opt.backbone == 'vanilla':
        from nerf.network import NeRFNetwork
    elif opt.backbone == 'grid':
        from nerf.network_grid import NeRFNetwork
    else:
        raise NotImplementedError(f'--backbone {opt.backbone} is not implemented!')

    print(opt)

    seed_everything(opt.seed)
    model = NeRFNetwork(opt)#.to(torch.float16)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Set up automatic token replacement for prompt
    if '<token>' in opt.text or '<token>' in opt.negative:
        if opt.learned_embeds_path is None:
            raise ValueError('--learned_embeds_path must be specified when using <token>') 
        import torch
        tmp = list(torch.load(opt.learned_embeds_path, map_location='cpu').keys())
        if len(tmp) != 1:
            raise ValueError('Something is wrong with the dict passed in for --learned_embeds_path') 
        token = tmp[0]
        opt.text = opt.text.replace('<token>', token)
        opt.negative = opt.negative.replace('<token>', token)
        print(f'Prompt after replacing <token>: {opt.text}')
        print(f'Negative prompt after replacing <token>: {opt.negative}')

    
    if opt.test:
        if opt.guidance == 'stable-diffusion':
            from nerf.sd import StableDiffusion
            guidance = StableDiffusion(device, opt.loss_type, opt.sd_version, opt.hf_key, opt.no_sds, opt)
        elif opt.guidance == 'clip':
            from nerf.clip import CLIP
            guidance = CLIP(device)
        elif opt.guidance == 'if':
            from nerf.if_utils import IF 
            
            # guidance = IF(device)
            guidance = IF(device, opt=opt)
        else:
            raise NotImplementedError(f'--guidance {opt.guidance} is not implemented.')
        
        
        trainer = Trainer(' '.join(sys.argv), 
                          'df', opt, model, guidance, 
                          optimizer=None, device=device, 
                          workspace=opt.workspace, 
                          fp16=opt.fp16, use_checkpoint=opt.ckpt)

        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()
        
        else:
            test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=16 if opt.write_image else 100).dataloader()
            if opt.save_mesh:
                # a special loader for poisson mesh reconstruction, 
                # loader = NeRFDataset(opt, device=device, type='test', H=128, W=128, size=100).dataloader()
                trainer.save_mesh()
            trainer.test(test_loader, write_video=not opt.write_image)
    else:
        optimizer = {}
        scheduler = {}
        train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=100).dataloader()

        if opt.optim == 'adan':
            from optimizer import Adan
            # Adan usually requires a larger LR
            optimizer = lambda model: Adan(model.get_params(5 * opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
        elif opt.optim == 'adam': # adam
            optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
        elif opt.optim == 'sgd':
            optimizer = lambda model: torch.optim.SGD(model.get_params(opt.lr), momentum=0)

        warm_up_with_cosine_lr = lambda iter: iter / opt.warm_iters if iter <= opt.warm_iters \
            else max(0.5 * ( math.cos((iter - opt.warm_iters) /(opt.iters - opt.warm_iters) * math.pi) + 1), 
                        opt.min_lr / opt.lr)
        alphas_cumprod = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="scheduler").alphas_cumprod
        sigma_schedule = lambda iter: (1 - alphas_cumprod[int((opt.iters - iter) / opt.iters * 960) + 20]) ** 0.5

        if opt.backbone == 'vanilla':
            scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, warm_up_with_cosine_lr)
        else:
            if opt.sigma_scheduler:
                scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, sigma_schedule)
            else:
                scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, warm_up_with_cosine_lr)
        

        if opt.guidance == 'stable-diffusion':
            from nerf.sd import StableDiffusion
            guidance = StableDiffusion(device, 
                                       opt.loss_type, 
                                       opt.sd_version,
                                       opt.hf_key, 
                                       opt.no_sds, opt)
        elif opt.guidance == 'clip':
            from nerf.clip import CLIP
            guidance = CLIP(device)
        elif opt.guidance == 'if':
            from nerf.if_utils import IF
            guidance = IF(device, opt=opt)
            
        else:
            raise NotImplementedError(f'--guidance {opt.guidance} is not implemented.')

        trainer = Trainer(' '.join(sys.argv), 'df', opt, model, guidance, 
                          device=device, workspace=opt.workspace, optimizer=optimizer, 
                          ema_decay=None, fp16=opt.fp16, lr_scheduler=scheduler, 
                          use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval, 
                          scheduler_update_every_step=True)

        if opt.gui:
            trainer.train_loader = train_loader # attach dataloader to trainer

            gui = NeRFGUI(opt, trainer)
            gui.render()
        
        else:
            valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=6).dataloader()
            max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
            ran_anything = trainer.train(train_loader, valid_loader,  max_epoch)
            if ran_anything:
                test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=6 if opt.write_image else 100).dataloader()
                try:
                    trainer.save_mesh()
                except:
                    warnings.warn("failed to save mesh")
                trainer.test(test_loader, write_video=not opt.write_image)