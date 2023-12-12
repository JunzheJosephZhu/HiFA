from transformers import logging
from diffusers import IFPipeline, DDPMScheduler

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import custom_bwd, custom_fwd
from .perpneg_utils import weighted_perpendicular_aggregator
# from perpneg_utils import weighted_perpendicular_aggregator

from diffusers import DDIMScheduler, DPMSolverMultistepScheduler
from diffusers import StableDiffusionDepth2ImgPipeline, DiffusionPipeline
from diffusers.pipelines.deepfloyd_if.pipeline_if_superresolution import IFSuperResolutionPipeline

from torchvision.utils import save_image
from torchmetrics import PearsonCorrCoef
import diffusers

import numpy as np

    
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


class IF(nn.Module):
    # def __init__(self, device, vram_O, t_range=[0.02, 0.98]):
    def __init__(self, device, vram_O=True, opt=None):
        super().__init__()
        self.device = device
        self.name = 'if'
        self.opt=opt
        self.fp16 = opt.fp16
        
        print(f'[INFO] loading DeepFloyd IF-I-XL...')

        model_key = "DeepFloyd/IF-I-XL-v1.0"

        is_torch2 = torch.__version__[0] == '2'

        # Create model
        pipe = IFPipeline.from_pretrained(model_key, variant="fp16", torch_dtype=torch.float16).to(device)

        if self.opt.if_upsample:
            model_key_stage2 = "DeepFloyd/IF-II-L-v1.0"
            self.stage2 = IFSuperResolutionPipeline.from_pretrained(model_key_stage2, text_encoder=None, 
                                                       variant="fp16", torch_dtype=torch.float16).to(device)

        # self.stage_2 = stage_2
        # stage 3
        # safety_modules = {"feature_extractor": stage_1.feature_extractor, "safety_checker": stage_1.safety_checker, "watermarker": stage_1.watermarker}
        # stage_3 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16)
        # stage_3.enable_model_cpu_offload()

        
        if not is_torch2:
            pipe.enable_xformers_memory_efficient_attention()
            self.stage2.enable_xformers_memory_efficient_attention()

        if vram_O:
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            pipe.enable_model_cpu_offload() # could also do enable_sequential_cpu_offload to save more memory
            if self.opt.if_upsample:
                self.stage2.unet.to(memory_format=torch.channels_last)
                self.stage2.enable_attention_slicing(1)
                self.stage2.enable_model_cpu_offload()
        else:
            if self.opt.if_upsample:
                self.stage2.to(device)
                pipe.to(device)

        self.unet = pipe.unet
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler

        self.pipe = pipe
        self.generator = torch.manual_seed(self.opt.seed)
        
        if self.opt.scheduler == "ddim":
            print("===== DDIM Scheduler =====")
            self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
            # self.stage_2.scheduler = DPMSolverMultistepScheduler.from_pretrained(model_key_stage2, subfolder="scheduler")
            # self.stage_2.scheduler.set_timesteps(4)
                
        elif self.opt.scheduler == "solver":
            self.scheduler = DPMSolverMultistepScheduler.from_pretrained(model_key, subfolder="scheduler", solver_order=3)
            
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * self.opt.min_percent)
        self.max_step = int(self.num_train_timesteps * self.opt.max_percent)
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        
        print(f'[INFO] loaded DeepFloyd IF-I-XL!')


        if self.opt.use_pearson:
            self.pearson = PearsonCorrCoef().to(self.device)
        
        
    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # prompt: [str]

        # TODO: should I add the preprocessing at https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deepfloyd_if/pipeline_if.py#LL486C10-L486C28
        prompt = self.pipe._text_preprocessing(prompt, clean_caption=False)
        inputs = self.tokenizer(prompt, padding='max_length', max_length=77, truncation=True, add_special_tokens=True, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings, torch.zeros_like(embeddings)


    # def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, grad_scale=1):

    def train_step(self, text_embeddings, neg_prompt_weights, pred_rgb, outputs,
                   guidance_scale=100, grad_scale=1, global_step=0, is_normal=False):
        
        
        # [0, 1] to [-1, 1] and make sure shape is [64, 64]
        images = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, (images.shape[0],), dtype=torch.long, device=self.device)

        
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(images)
            images_noisy = self.scheduler.add_noise(images, noise, t)
            with torch.cuda.amp.autocast(enabled=self.fp16):
                if self.fp16:
                    images_noisy = images_noisy.half()
                    
                # pred noise
                model_input = torch.cat([images_noisy] * 2)
                model_input = self.scheduler.scale_model_input(model_input, t)
                tt = torch.cat([t] * 2)
                
                # print('text_embeddings', model_input.size(), tt.size(), text_embeddings.size())
                # text_embeddings torch.Size([2, 3, 64, 64]) torch.Size([2]) torch.Size([4, 77, 4096])
                
                noise_pred = self.unet(model_input, tt, encoder_hidden_states=text_embeddings).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
                noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # TODO: how to use the variance here?
            # noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

        # w(t), sigma_t^2
        w = (1 - self.alphas_cumprod[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        targets = (images - grad).detach()
        loss = 0.5 * F.mse_loss(images.float(), targets, reduction='sum') / images.shape[0]
        return loss, loss


    def if_stage2(self, image, weights, B, K, text_embeddings, num_inference_steps, eta):
        height, width = self.opt.if_upsample_h, self.opt.if_upsample_w
        self.stage2.scheduler.config.timestep_spacing = "trailing"
        self.stage2.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.stage2.scheduler.timesteps


        # 5. Prepare intermediate images
        num_channels = self.stage2.unet.config.in_channels // 2
        intermediate_images = self.stage2.prepare_intermediate_images(
            B,
            num_channels,
            height,
            width,
            text_embeddings.dtype,
            self.device,
            None,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.stage2.prepare_extra_step_kwargs(None, eta)

        # 7. Prepare upscaled image and noise level
        image = self.stage2.preprocess_image(image, 1, self.device)
        upscaled = F.interpolate(image, (height, width), mode="bilinear", align_corners=True)

        noise_level = torch.tensor([self.opt.if_upsample_noise_level] * upscaled.shape[0], device=upscaled.device)
        noise = diffusers.utils.randn_tensor(upscaled.shape, generator=None, device=upscaled.device, dtype=upscaled.dtype)
        upscaled = self.stage2.image_noising_scheduler.add_noise(upscaled, noise, timesteps=noise_level)

        noise_level = torch.cat([noise_level] * (1 + K))
        
        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.stage2.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                model_input = torch.cat([intermediate_images, upscaled], dim=1)

                model_input = torch.cat([model_input] * (1 + K))
                model_input = self.stage2.scheduler.scale_model_input(model_input, t)

                # predict the noise residual
                noise_pred = self.stage2.unet(
                    model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    class_labels=noise_level,
                    cross_attention_kwargs=None,
                    return_dict=False,
                )[0]

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred[:B], noise_pred[B:]
                noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1] // 2, dim=1)
                noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1] // 2, dim=1)
                delta_noise_preds = noise_pred_text - noise_pred_uncond.repeat(K, 1, 1, 1)
                noise_pred = noise_pred_uncond + self.opt.if_upsample_guidance * weighted_perpendicular_aggregator(delta_noise_preds, weights, B)

                noise_pred = torch.cat([noise_pred, predicted_variance[:B]], dim=1) # only take predicted variance from prompt without direction

                if self.stage2.scheduler.config.variance_type not in ["learned", "learned_range"]:
                    noise_pred, _ = noise_pred.split(intermediate_images.shape[1], dim=1)

                # compute the previous noisy sample x_t -> x_t-1
                intermediate_images = self.stage2.scheduler.step(
                    noise_pred, t, intermediate_images, **extra_step_kwargs, return_dict=False
                )[0]


        return intermediate_images



    def train_step_perpneg(self, text_embeddings, weights, pred_rgb, outputs,
                           guidance_scale=100, grad_scale=1, global_step=0, is_normal=False):
    
        
        B = pred_rgb.shape[0]
        K = (text_embeddings.shape[0] // B) - 1 # maximum number of prompts        

        # [0, 1] to [-1, 1] and make sure shape is [64, 64]
        images = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, (images.shape[0],), dtype=torch.long, device=self.device)

        # print('t original ',self.min_step, self.max_step + 1, t)
        if self.opt.timestep_scheduling == 'sqrt':
            percentage = np.sqrt(float(global_step) / self.opt.iters) # dont forget to change sigma schedule with this
            t = (self.max_step - self.min_step) * (1 - percentage) + self.min_step 
            t = torch.tensor(t, device=self.device).to(int)
        elif self.opt.timestep_scheduling == 'cubed':
            percentage = np.cbrt(float(global_step) / self.opt.iters) # dont forget to change sigma schedule with this
            t = (self.max_step - self.min_step) * (1 - percentage) + self.min_step 
            t = torch.tensor(t, device=self.device).to(int)
        elif self.opt.timestep_scheduling == 'linear':
            percentage = float(global_step) / self.opt.iters
            t = (self.max_step - self.min_step) * (1 - percentage) + self.min_step 
            t = torch.tensor(t, device=self.device).to(int)
        elif self.opt.timestep_scheduling == 'random':
            percentage = torch.rand((1,), device=self.device)[0]
            t = (20 + 960 * percentage).to(int)
        elif self.opt.timestep_scheduling == 'cosine':
            percentage = np.cos(np.pi - float(global_step) / self.opt.iters * np.pi) / 2 + 0.5
            t = (self.max_step - self.min_step) * (1 - percentage) + self.min_step 
            t = torch.tensor(t, device=self.device).to(int)

        if self.opt.sds_weight == "constant":
            w = 0.05
        elif self.opt.sds_weight == "noisematch":
            w = 0.05 * self.alphas_cumprod[t] ** 0.5 / (1 - self.alphas_cumprod[t]) ** 0.5
        elif self.opt.sds_weight == "sdslike":
            w = 0.05 * self.alphas_cumprod[t] ** 0.5 * (1 - self.alphas_cumprod[t]) ** 0.5
        elif self.opt.sds_weight == "strongerlower":
            w = 0.05 * self.alphas_cumprod[t] ** 1.0 / (1 - self.alphas_cumprod[t]) ** 0.5

        if self.opt.scheduler == "solver":
            self.scheduler.set_timesteps(self.opt.denoising_steps)
            timestep_flipped = self.scheduler.timesteps.flip(-1).to(t.device)
            timestep = timestep_flipped[torch.searchsorted(timestep_flipped, t)]
        elif self.opt.scheduler == "ddim":
            timestep = t
        else:
            raise NotImplementedError
        
        
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(images)
            images_noisy = self.scheduler.add_noise(images, noise, timestep)

            with torch.cuda.amp.autocast(enabled=self.fp16):
                if self.fp16:
                    images_noisy = images_noisy.half()
                    
                # pred noise
                model_input = torch.cat([images_noisy] * (1 + K))
                model_input = self.scheduler.scale_model_input(model_input, timestep)
                tt = torch.cat([timestep.reshape(1)] * (1 + K))
                
                unet_output = self.unet(model_input, tt, encoder_hidden_states=text_embeddings).sample
                noise_pred_uncond, noise_pred_text = unet_output[:B], unet_output[B:]
                noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
                noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)
                # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                delta_noise_preds = noise_pred_text - noise_pred_uncond.repeat(K, 1, 1, 1)
                noise_pred = noise_pred_uncond + guidance_scale * weighted_perpendicular_aggregator(delta_noise_preds, weights, B)
                if self.opt.if_upsample:
                    image_denoised = (images_noisy - (1 - self.alphas_cumprod[t]) ** 0.5 * noise_pred) / self.alphas_cumprod[t] ** 0.5
                    target_upsampled = self.if_stage2(image_denoised, weights, B, K, text_embeddings, self.opt.if_upsample_steps, self.opt.if_upsample_eta)

        if self.opt.if_upsample:
            pred_upsampled = F.interpolate(pred_rgb, (self.opt.if_upsample_h, self.opt.if_upsample_w), mode='bilinear', align_corners=False) * 2 - 1
            
            # loss = 0.5 * F.mse_loss(pred_upsampled, target_upsampled, reduction="sum") / images.shape[0]
            loss = 0.5 * (F.mse_loss(pred_upsampled, target_upsampled, reduction="sum") / images.shape[0] ) / (pred_upsampled.shape[-2] * pred_upsampled.shape[-1])
            return loss, loss

        # w(t), sigma_t^2
        # w = (1 - self.alphas[t])
        
        # w = (1 - self.alphas_cumprod[t])
        # grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        # w_org = (1 - self.alphas_cumprod[t]) # 0.99
        # print('w_org and w: ', w_org, w)
        
        grad = grad_scale * w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        targets = (images - grad).detach()
        
        

        # with torch.no_grad():
        #     neg_embeds,_ = self.get_text_embeds([''])
        #     images_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False) * 2 - 1
        #     targets = self.stage_2(image=targets, prompt_embeds=text_embeddings[:B], 
        #                     negative_prompt_embeds=neg_embeds, 
        #                     generator=self.generator, output_type="pt"
        #                     ).images
            
            # print('targets', targets.size(), targets.max(), targets.min(), images_256.size(), images_256.max(), images_256.min())
        # targets stage 2?
        
        loss = 0.5 * F.mse_loss(images.float(), targets, reduction='sum') / images.shape[0]
        # loss = 0.5 * F.mse_loss(images_256.float(), targets, reduction='sum') / images_256.shape[0]

        return loss, loss


    @torch.no_grad()
    def produce_imgs(self, text_embeddings, height=64, width=64, num_inference_steps=50, guidance_scale=7.5):

        images = torch.randn((1, 3, height, width), device=text_embeddings.device, dtype=text_embeddings.dtype)
        images = images * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            model_input = torch.cat([images] * 2)
            model_input = self.scheduler.scale_model_input(model_input, t)

            # predict the noise residual
            noise_pred = self.unet(model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
            noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

            # compute the previous noisy sample x_t -> x_t-1
            images = self.scheduler.step(noise_pred, t, images).prev_sample

        images = (images + 1) / 2

        return images


    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        pos_embeds = self.get_text_embeds(prompts) # [1, 77, 768]
        neg_embeds = self.get_text_embeds(negative_prompts)
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0) # [2, 77, 768]

        # Text embeds -> img
        imgs = self.produce_imgs(text_embeds, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs

# CUDA_VISIBLE_DEVICES=0 python if_utils.py "a ripe strawberry"
if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    parser.add_argument('-H', type=int, default=64)
    parser.add_argument('-W', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = IF(device, opt.vram_O)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    from PIL import Image
    Image.fromarray(imgs[0]).save('test_if.png')
    
    # visualize image
    plt.imshow(imgs[0])
    plt.show()



